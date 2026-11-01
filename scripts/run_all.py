"""Full pipeline: train every agent, run every experiment, draw the figures.

    python scripts/run_all.py                    # simulated world, 3 seeds, 1M steps each
    python scripts/run_all.py --quick            # simulated world, ~25 min
    python scripts/run_all.py --real --quick     # REAL Docker recordings (see README section 7)
    python scripts/run_all.py --seeds 0 1 2 3 4
    python scripts/run_all.py --skip-train       # only experiments + figures

--real trains on configs/real_train.yaml (real recorded answers, train split),
tests on configs/real_test.yaml (held-out real recordings), and keeps its
outputs apart in models_real/, results_real/ and figures_real/.

Models that already exist are skipped, so you can stop it and run it again.
Works the same on Windows, Linux and macOS.
"""
import argparse
import os
import subprocess
import sys

PY = sys.executable


def run(args, required=True):
    print("\n$ " + " ".join(args), flush=True)
    res = subprocess.run([PY] + args)
    if res.returncode != 0:
        if required:
            sys.exit(f"step failed: {' '.join(args)}")
        print("  -> step skipped (see message above), carrying on")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--slow-episodes", type=int, default=200)
    p.add_argument("--lams", nargs="+", default=["0", "0.5", "2", "4"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--real", action="store_true", help="train and test on real Docker recordings")
    p.add_argument("--config", default=None, help="training world (YAML)")
    p.add_argument("--eval-config", default=None, help="testing world (YAML)")
    p.add_argument("--models-dir", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--figures-dir", default=None)
    a = p.parse_args()

    if a.quick:
        a.seeds, a.steps, a.episodes, a.slow_episodes, a.lams = [0], 100_000, 200, 50, ["2"]
    if a.real:
        a.config = a.config or "configs/real_train.yaml"
        a.eval_config = a.eval_config or "configs/real_test.yaml"
        suffix = "_real"
        if not os.path.exists("data/real/recordings.jsonl"):
            sys.exit("no real recordings yet: run `python -m faultdiag.testbed.record` first "
                     "(README section 7)")
    else:
        suffix = ""
    M = a.models_dir or f"models{suffix}"
    R = a.results_dir or f"results{suffix}"
    F = a.figures_dir or f"figures{suffix}"

    def train(algo, tag, *extra):
        for s in a.seeds:
            name = f"{algo}_{tag}_s{s}" if tag else f"{algo}_s{s}"
            if os.path.exists(os.path.join(M, name, "model.zip")):
                print(f"skip {name} (already trained)")
                continue
            cmd = ["-m", "faultdiag.agents.train", "--algo", algo, "--seed", str(s),
                   "--steps", str(a.steps), "--out", M]
            if tag:
                cmd += ["--tag", tag]
            if a.config:
                cmd += ["--config", a.config]
            run(cmd + list(extra))

    if not a.skip_train:
        print(f"=== 1. training ({'real recordings' if a.real else 'simulator'}) ===")
        train("maskppo", "")                                  # main method
        train("recppo", "", "--obs-mode", "raw")              # learns its own memory
        train("dqn", "")                                      # value-based comparison
        train("maskppo", "noquery", "--no-queries")           # recovery-only RL
        train("maskppo", "nobelief", "--obs-mode", "raw")     # ablation
        train("maskppo", "noctx", "--no-context")             # ablation
        train("ppo", "nomask")                                # ablation
        train("maskppo", "infogain", "--info-bonus", "10")    # ablation
        for lam in a.lams:
            train("maskppo", f"lam{lam}", "--lam", lam)       # Pareto front

    print("\n=== 2. experiments ===")
    os.makedirs(R, exist_ok=True)
    ev = ["-m", "faultdiag.agents.evaluate", "--episodes", str(a.episodes),
          "--slow-episodes", str(a.slow_episodes)]
    if a.eval_config:
        ev += ["--config", a.eval_config]

    def evaluate(exp, out, methods, *extra):
        run(ev + ["--experiment", exp, "--out", os.path.join(R, out)] + list(extra)
            + ["--methods"] + methods, required=False)

    base = ["oracle", "never_acquire", "always_deep", "random_acquire", "entropy_threshold",
            "greedy_voi", "qmdp", "act_then_measure", "pomcp"]
    main_rl = [f"{M}/maskppo_s*", f"{M}/recppo_s*"]
    sweep = ["never_acquire", "entropy_threshold", "greedy_voi", "act_then_measure"] + main_rl

    evaluate("E1", "E1.csv", base + main_rl + [f"{M}/dqn_s*", f"{M}/maskppo_noquery_s*"])
    evaluate("E3", "E3.csv", sweep)
    evaluate("E3b", "E3b.csv", sweep)
    evaluate("E4", "E4.csv", ["entropy_threshold", "greedy_voi", "act_then_measure"] + main_rl)
    evaluate("E5", "E5.csv", ["never_acquire", "entropy_threshold", "greedy_voi",
                              "act_then_measure", "pomcp"] + main_rl)
    evaluate("E6", "E6.csv", ["never_acquire", "entropy_threshold", "greedy_voi",
                              "act_then_measure", "pomcp"] + main_rl + [f"{M}/dqn_s*"])
    evaluate("E7", "E7.csv", [f"{M}/maskppo_s*", f"{M}/maskppo_nobelief_s*", f"{M}/maskppo_noctx_s*",
                              f"{M}/ppo_nomask_s*", f"{M}/maskppo_infogain_s*", f"{M}/recppo_s*"])
    # E2: one file per lambda; lambda = 1 is the main maskppo run
    evaluate("E2", "E2_lam1.csv", ["greedy_voi", "entropy_threshold:0.15", "entropy_threshold:0.35",
                                   "entropy_threshold:0.6", f"{M}/maskppo_s*"], "--lam", "1")
    for lam in a.lams:
        evaluate("E2", f"E2_lam{lam}.csv", ["greedy_voi", f"{M}/maskppo_lam{lam}_s*"], "--lam", lam)

    print("\n=== 3. figures ===")
    run(["-m", "faultdiag.analysis.plots", "--results", R, "--models", M, "--out", F])
    print(f"\nall done, see {F}/")


if __name__ == "__main__":
    main()
