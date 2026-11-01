"""Print a few episodes step by step, to see what a policy actually does.

    python scripts/show_episodes.py greedy_voi --n 5
    python scripts/show_episodes.py models/maskppo_s0 --n 5 --fault network_degradation

Useful for the qualitative part of the report ("for network faults the
agent learned telemetry -> logs -> reroute").
"""
import argparse

from faultdiag.agents.policies import SB3Policy, load_policy
from faultdiag.env import FaultDiagEnv
from faultdiag.env.constants import ACTION_NAMES, FAULT_NAMES
from faultdiag.experiments import scenario_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("policy")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--scenario", default="nominal")
    p.add_argument("--fault", default=None, choices=FAULT_NAMES)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    pol = load_policy(args.policy)
    agent_cfg = pol.train_cfg if isinstance(pol, SB3Policy) else None
    env = FaultDiagEnv(scenario_config(args.scenario, agent_cfg=agent_cfg))
    opts = {"fault": FAULT_NAMES.index(args.fault)} if args.fault else None

    for ep in range(args.n):
        obs, _ = env.reset(seed=args.seed + ep, options=opts)
        pol.reset(env)
        print(f"\n--- episode {ep}: true fault = {FAULT_NAMES[env.fault]}, "
              f"severity = {'high' if env.severity else 'low'}, budget = {env.budget_max:.0f}")
        print(f"    free telemetry said: {FAULT_NAMES[env.last_label[0]]}")
        done, total = False, 0.0
        while not done:
            b = env.belief.marginal
            top = b.argmax()
            a = pol.act(obs, env)
            obs, r, done, _, info = env.step(a)
            total += r
            what = env.history[-1][1]
            if ACTION_NAMES[a].startswith("query"):
                what = f"-> reports {FAULT_NAMES[what]}"
            elif what is not None:
                what = "-> fixed" if what else "-> did not help"
            else:
                what = ""
            print(f"    t={env.t:2d}  belief top: {FAULT_NAMES[top]:20s} ({b[top]:.2f})  "
                  f"action: {ACTION_NAMES[a]:20s} {what}   r={r:6.1f}")
        print(f"    outcome: {info['outcome']}, return {total:.1f}")


if __name__ == "__main__":
    main()
