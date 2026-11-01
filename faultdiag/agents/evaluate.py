"""Evaluate policies on one or more scenarios and write one CSV row per episode.

Examples
--------
    # all baselines + every trained MaskablePPO seed on experiment E1
    python -m faultdiag.agents.evaluate --experiment E1 \
        --methods never_acquire greedy_voi pomcp "models/maskppo_s*" \
        --episodes 1000 --out results/E1.csv

Every method sees exactly the same incidents: episode i always uses
seed `seed0 + i`, so comparisons are paired.
"""
import argparse
import glob
import os
import time

import pandas as pd

from faultdiag.agents.policies import SB3Policy, load_policy
from faultdiag.env import EnvConfig, FaultDiagEnv
from faultdiag.env.constants import SOURCE_NAMES
from faultdiag.experiments import EXPERIMENTS, scenario_config

SLOW_METHODS = {"pomcp"}


def run_episodes(policy, cfg, n_episodes, seed0=100_000, env=None):
    env = env or FaultDiagEnv(cfg)
    rows = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + i)
        policy.reset(env)
        total, done, steps = 0.0, False, 0
        t0 = time.perf_counter()
        while not done:
            a = policy.act(obs, env)
            obs, r, done, _, info = env.step(a)
            total += r
            steps += 1
        ms = (time.perf_counter() - t0) * 1000 / steps
        row = {
            "episode": i,
            "return": total,
            "success": info["success"],
            "outcome": info["outcome"],
            "fault": info["fault"],
            "severity": info["severity"],
            "diag_correct": info["diag_correct"],
            "query_cost": info["query_cost"],
            "n_queries": info["n_queries"],
            "n_repairs": info["n_repairs"],
            "time": info["time"],
            "decision_ms": ms,
        }
        for k, name in enumerate(SOURCE_NAMES):
            row[f"q_{name}"] = info["source_counts"][k]
        rows.append(row)
    return rows


def expand_methods(specs):
    out = []
    for s in specs:
        if any(ch in s for ch in "*?["):
            out.extend(sorted(p for p in glob.glob(s) if os.path.isdir(p)))
        else:
            out.append(s)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=sorted(EXPERIMENTS), default=None)
    p.add_argument("--scenarios", nargs="*", default=None)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--slow-episodes", type=int, default=200,
                   help="episode count for slow planners such as POMCP")
    p.add_argument("--lam", type=float, default=1.0, help="query-cost weight used for scoring")
    p.add_argument("--seed0", type=int, default=100_000)
    p.add_argument("--config", default=None,
                   help="YAML with world settings, e.g. configs/real_test.yaml to test on real recordings")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    base = EnvConfig.from_yaml(args.config) if args.config else EnvConfig()

    scenarios = args.scenarios or EXPERIMENTS[args.experiment or "E1"]
    methods = expand_methods(args.methods)
    if not methods:
        raise SystemExit("no methods matched; did you train the models first?")

    all_rows = []
    for spec in methods:
        policy = load_policy(spec)
        is_rl = isinstance(policy, SB3Policy)
        n = args.slow_episodes if policy.name in SLOW_METHODS else args.episodes
        for sc in scenarios:
            cfg = scenario_config(sc, base=base, agent_cfg=policy.train_cfg if is_rl else None,
                                  query_cost_weight=args.lam, info_gain_bonus=0.0)
            t0 = time.time()
            rows = run_episodes(policy, cfg, n, seed0=args.seed0)
            for r in rows:
                r.update(method=policy.name, run_seed=policy.seed if is_rl else 0,
                         scenario=sc, lam=args.lam, is_rl=is_rl)
            all_rows.extend(rows)
            df = pd.DataFrame(rows)
            print(f"{policy.name:24s} {sc:22s} return {df['return'].mean():7.2f} "
                  f"success {df['success'].mean():6.1%} queries {df['n_queries'].mean():4.2f} "
                  f"({time.time() - t0:.0f}s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(all_rows).to_csv(args.out, index=False)
    print(f"wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
