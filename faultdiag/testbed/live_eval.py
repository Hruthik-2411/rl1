"""Run a policy on real incidents in the Docker testbed (experiment E8).

    python -m faultdiag.testbed.live_eval --methods greedy_voi models/maskppo_s0 \
        --episodes 30 --out results/E8_live.csv

The same FaultDiagEnv bookkeeping is used as in simulation (budget, time
units, reward), only the answers to queries and repairs come from the
containers. Wall-clock time per incident is recorded too.
"""
import argparse
import os
import time

import pandas as pd

from faultdiag.agents.evaluate import run_episodes
from faultdiag.agents.policies import SB3Policy, load_policy
from faultdiag.env import FaultDiagEnv
from faultdiag.env.model import DiagModel
from faultdiag.experiments import scenario_config
from .docker_backend import DockerBackend


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--calibrated", default=None,
                   help="calibrated.json; if given, the agent's belief uses it")
    p.add_argument("--settle", type=float, default=15.0)
    p.add_argument("--out", default="results/E8_live.csv")
    args = p.parse_args(argv)

    backend = DockerBackend(settle=args.settle)
    base_model = DiagModel.from_file(args.calibrated) if args.calibrated else None

    rows = []
    for spec in args.methods:
        pol = load_policy(spec)
        is_rl = isinstance(pol, SB3Policy)
        cfg = scenario_config("nominal", agent_cfg=pol.train_cfg if is_rl else None)
        env = FaultDiagEnv(cfg, backend=backend, base_model=base_model)
        t0 = time.time()
        res = run_episodes(pol, cfg, args.episodes, seed0=500_000, env=env)
        wall = (time.time() - t0) / args.episodes
        for r in res:
            r.update(method=pol.name, run_seed=pol.seed if is_rl else 0,
                     scenario="live_docker", is_rl=is_rl)
        rows.extend(res)
        df = pd.DataFrame(res)
        print(f"{pol.name:20s} success {df['success'].mean():.0%} return {df['return'].mean():.1f} "
              f"queries {df['n_queries'].mean():.2f} | {wall:.0f} s per incident")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
