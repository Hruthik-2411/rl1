"""Train an RL agent on FaultDiagEnv.

Examples
--------
    python -m faultdiag.agents.train --algo maskppo --seed 0
    python -m faultdiag.agents.train --algo recppo --obs-mode raw --seed 0
    python -m faultdiag.agents.train --algo maskppo --lam 2.0 --tag lam2 --seed 0

Each run writes to models/<run_name>/:
    model.zip      the trained SB3 model
    meta.json      algorithm, seed, steps and the full env config
    monitor.csv    episode returns during training (for learning curves)
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from faultdiag.env import EnvConfig, FaultDiagEnv

ALGOS = ("maskppo", "recppo", "dqn", "ppo")


def make_vec_env(cfg, n_envs, seed, log_dir, masked, reward_scale=1.0):
    from gymnasium.wrappers import TransformReward
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def thunk(rank):
        def _f():
            env = FaultDiagEnv(cfg)
            # Monitor sits inside the scaling so learning curves show real returns
            env = Monitor(env, os.path.join(log_dir, f"monitor_{rank}") if rank == 0 else None,
                          info_keywords=("success", "query_cost"))
            if reward_scale != 1.0:
                env = TransformReward(env, lambda r: r * reward_scale)
            if masked:
                from sb3_contrib.common.wrappers import ActionMasker
                env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
            env.reset(seed=seed * 1000 + rank)
            return env
        return _f

    return DummyVecEnv([thunk(i) for i in range(n_envs)])


def build_model(algo, venv, seed, tb_dir, args):
    common = dict(seed=seed, verbose=0, tensorboard_log=tb_dir)
    net = dict(net_arch=[128, 128])

    if algo == "maskppo":
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", venv, learning_rate=args.lr, n_steps=1024,
                           batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                           ent_coef=args.ent_coef, clip_range=0.2,
                           policy_kwargs=net, **common)
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO("MlpPolicy", venv, learning_rate=args.lr, n_steps=1024,
                   batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                   ent_coef=args.ent_coef, policy_kwargs=net, **common)
    if algo == "recppo":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO("MlpLstmPolicy", venv, learning_rate=args.lr, n_steps=512,
                            batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                            ent_coef=args.ent_coef,
                            policy_kwargs=dict(net_arch=[128], lstm_hidden_size=128),
                            **common)
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN("MlpPolicy", venv, learning_rate=5e-4, buffer_size=100_000,
                   learning_starts=5_000, batch_size=128, gamma=0.99,
                   train_freq=1, gradient_steps=1, target_update_interval=1_000,
                   exploration_fraction=0.5, exploration_final_eps=0.05,
                   policy_kwargs=net, **common)
    raise ValueError(algo)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=ALGOS, default="maskppo")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--config", default=None, help="YAML file with EnvConfig overrides")
    p.add_argument("--lam", type=float, default=None, help="weight on query cost")
    p.add_argument("--obs-mode", choices=["belief", "raw"], default=None)
    p.add_argument("--no-context", action="store_true", help="hide costs/availability")
    p.add_argument("--no-queries", action="store_true", help="recovery-only agent")
    p.add_argument("--info-bonus", type=float, default=None)
    p.add_argument("--tag", default="", help="extra label for the run name")
    p.add_argument("--out", default="models")
    return p.parse_args(argv)


def config_from_args(args):
    cfg = EnvConfig.from_yaml(args.config) if args.config else EnvConfig()
    over = {}
    if args.lam is not None:
        over["query_cost_weight"] = args.lam
    if args.obs_mode:
        over["obs_mode"] = args.obs_mode
    if args.no_context:
        over["include_context"] = False
    if args.no_queries:
        over["allow_queries"] = False
    if args.info_bonus is not None:
        over["info_gain_bonus"] = args.info_bonus
    return cfg.copy(**over)


def main(argv=None):
    args = parse_args(argv)
    cfg = config_from_args(args)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    label = args.algo + (f"_{args.tag}" if args.tag else "")
    run_name = f"{label}_s{args.seed}"
    run_dir = os.path.join(args.out, run_name)
    os.makedirs(run_dir, exist_ok=True)

    masked = args.algo == "maskppo"
    # DQN and the LSTM agent got stuck on "always escalate" / "never query"
    # with raw rewards of +-100. Training them on rewards scaled to about +-1
    # is more stable. Evaluation always uses the unscaled environment.
    scale = 0.01 if args.algo in ("dqn", "recppo") else 1.0
    venv = make_vec_env(cfg, args.n_envs, args.seed, run_dir, masked, reward_scale=scale)
    model = build_model(args.algo, venv, args.seed, os.path.join(args.out, "tb"), args)

    print(f"training {run_name} for {args.steps:,} steps")
    t0 = time.time()
    model.learn(total_timesteps=args.steps, tb_log_name=run_name, progress_bar=False)
    minutes = (time.time() - t0) / 60

    model.save(os.path.join(run_dir, "model.zip"))
    meta = {"name": run_name, "label": label, "algo": args.algo, "seed": args.seed,
            "steps": args.steps, "train_minutes": round(minutes, 1), "env": cfg.to_dict()}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # quick sanity number so you know the run did something
    from faultdiag.agents.policies import load_policy
    from faultdiag.agents.evaluate import run_episodes
    pol = load_policy(run_dir)
    rows = run_episodes(pol, cfg.copy(info_gain_bonus=0.0), n_episodes=300, seed0=999_000)
    ret = np.mean([r["return"] for r in rows])
    succ = np.mean([r["success"] for r in rows])
    print(f"done in {minutes:.1f} min | quick eval: return {ret:.1f}, success {succ:.2%}")


if __name__ == "__main__":
    main()
