"""Record real diagnostic data on the Docker testbed.

For every fault (and both severities) this injects the fault into the real
containers, asks each of the four real sources several times, and writes
down exactly what they answered. A second pass injects each fault again
and tries every real repair to see whether it actually fixes the system.

Everything is appended to one JSON-lines file as it happens, so you can
stop and restart without losing data. Some trials are kept aside as a
"test" split that training never sees.

    # start the testbed first:  docker compose up -d --build
    python -m faultdiag.testbed.record --trials 10 --repeats 3 --repair-trials 2
    python -m faultdiag.testbed.record --trials 5 --repair-trials 1          # quicker, ~1 h

At the end it also writes data/real/calibrated.json: the confusion matrices
and repair success rates measured from the TRAIN recordings, which the
agent's belief and the planners use when working with real data.

    python -m faultdiag.testbed.record --calibrate-only    # just rebuild calibrated.json
    python -m faultdiag.testbed.record --selftest          # check the testbed first (~6 min)
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from faultdiag.env.constants import (
    FAULT_NAMES, N_FAULTS, N_SEV, N_SOURCES, N_REPAIRS, NO_FAULT, REPAIR_NAMES, SOURCE_NAMES,
)
from faultdiag.env.model import DiagModel


def split_for(trial, n_trials, test_fraction):
    n_test = int(round(n_trials * test_fraction))
    return "test" if trial >= n_trials - n_test else "train"


def record_observations(be, out, args, rng):
    severities = args.severities
    total = args.trials * N_FAULTS * len(severities)
    done = 0
    t_start = time.time()
    for trial in range(args.trials):
        split = split_for(trial, args.trials, args.test_fraction)
        for f in range(N_FAULTS):
            for sev in severities:
                if f == NO_FAULT and sev == 0 and 1 in severities:
                    continue          # "no fault" has no severity, record it once
                be.start_incident(f, sev, rng)
                answers = [[] for _ in range(N_SOURCES)]
                for _ in range(args.repeats):
                    for k in range(N_SOURCES):
                        answers[k].append(int(be.query(k, rng)))
                be.end_incident()
                rec = {"kind": "obs", "split": split, "fault": f, "severity": sev if f else 1,
                       "trial": trial, "answers": answers, "time": time.strftime("%H:%M:%S")}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                done += 1
                eta = (time.time() - t_start) / done * (total - done) / 60
                said = ", ".join(f"{SOURCE_NAMES[k]}={FAULT_NAMES[answers[k][0]]}"
                                 for k in range(N_SOURCES))
                print(f"[obs {done}/{total}, ~{eta:.0f} min left] {split:5s} "
                      f"{FAULT_NAMES[f]:20s} sev={sev}  {said}", flush=True)


def record_repairs(be, out, args, rng):
    severities = args.severities
    faults = [f for f in range(N_FAULTS) if f != NO_FAULT]
    total = args.repair_trials * len(faults) * len(severities) * N_REPAIRS
    done = 0
    t_start = time.time()
    for trial in range(args.repair_trials):
        split = (split_for(trial, args.repair_trials, args.test_fraction)
                 if args.repair_trials > 1 else "train")
        for f in faults:
            for sev in severities:
                for r in range(N_REPAIRS):
                    be.start_incident(f, sev, rng)
                    fixed = bool(be.repair(r, rng))
                    be.end_incident()
                    rec = {"kind": "repair", "split": split, "fault": f, "severity": sev,
                           "repair": r, "fixed": fixed, "trial": trial}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    done += 1
                    eta = (time.time() - t_start) / done * (total - done) / 60
                    print(f"[repair {done}/{total}, ~{eta:.0f} min left] {split:5s} "
                          f"{FAULT_NAMES[f]:20s} sev={sev} {REPAIR_NAMES[r]:20s} "
                          f"-> {'FIXED' if fixed else 'not fixed'}", flush=True)


# what the testbed is built to do; --selftest checks it on the real containers
EXPECTED_FIX = {1: "scale_out", 2: "kill_leaky_process", 3: "reroute_traffic",
                4: "restart_service", 5: "failover_db"}
RESTART_SHOULD_FIX = {1: False, 2: True, 3: False, 4: True, 5: False}


def selftest(be, rng):
    """About 6 minutes: for every fault, the right repair must fix it and a
    restart must fix only what lives inside the service. Writes no data."""
    print("selftest: each fault at high severity, right repair vs restart\n")
    ok = True
    for f, right in EXPECTED_FIX.items():
        checks = [(REPAIR_NAMES.index(right), True)]
        if right != "restart_service":
            checks.append((REPAIR_NAMES.index("restart_service"), RESTART_SHOULD_FIX[f]))
        for r, want in checks:
            be.start_incident(f, 1, rng)
            said = [FAULT_NAMES[be.query(k, rng)] for k in range(N_SOURCES)]
            fixed = bool(be.repair(r, rng))
            be.end_incident()
            good = fixed == want
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'}  {FAULT_NAMES[f]:20s} {REPAIR_NAMES[r]:20s} "
                  f"-> {'FIXED' if fixed else 'not fixed':9s} (expected "
                  f"{'FIXED' if want else 'not fixed'})  sources: {', '.join(said)}", flush=True)
    print("\nselftest", "PASSED: go ahead and record" if ok else
          "FAILED: paste this output before recording")
    return ok


def calibrate(path, split="train", smoothing=1.0):
    """Estimate confusion matrices and repair success rates from recordings."""
    counts = np.zeros((N_SOURCES, N_SEV, N_FAULTS, N_FAULTS))
    wins = defaultdict(int)
    tries = defaultdict(int)
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") != split:
                continue
            if rec["kind"] == "obs":
                sevs = [0, 1] if rec["fault"] == NO_FAULT else [rec["severity"]]
                for k, labels in enumerate(rec["answers"]):
                    for o in labels:
                        for s in sevs:
                            counts[k, s, rec["fault"], o] += 1
            else:
                key = (rec["fault"], rec["repair"])
                tries[key] += 1
                wins[key] += int(rec["fixed"])

    obs = counts + smoothing
    obs /= obs.sum(axis=-1, keepdims=True)

    success = DiagModel().success.copy()      # simulator guess where nothing was measured
    for (f, r), n in tries.items():
        success[f, r] = min(wins[(f, r)] / n, 0.98)
    success[NO_FAULT] = 0.0
    return obs, success, counts, tries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=10, help="recorded incidents per fault and severity")
    p.add_argument("--repeats", type=int, default=3, help="answers recorded per source per incident")
    p.add_argument("--repair-trials", type=int, default=2, help="attempts per fault/severity/repair")
    p.add_argument("--severities", type=int, nargs="+", default=[0, 1], choices=[0, 1])
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--settle", type=float, default=15.0, help="seconds to let a fault show up")
    p.add_argument("--out", default="data/real")
    p.add_argument("--skip-obs", action="store_true")
    p.add_argument("--skip-repairs", action="store_true")
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--selftest", action="store_true",
                   help="check the testbed behaves as designed (~6 min), record nothing")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rec_path = os.path.join(args.out, "recordings.jsonl")

    if not args.calibrate_only:
        from .docker_backend import DockerBackend
        be = DockerBackend(settle=args.settle)
        rng = np.random.default_rng(0)
        print("checking the testbed is healthy ...")
        if not be.tb.heal():
            raise SystemExit("testbed is not healthy; run `docker compose up -d --build` and "
                             "`python -m faultdiag.testbed.control status` first")
        if args.selftest:
            selftest(be, rng)
            return
        with open(rec_path, "a") as out:
            if not args.skip_obs:
                record_observations(be, out, args, rng)
            if not args.skip_repairs:
                record_repairs(be, out, args, rng)

    obs, success, counts, tries = calibrate(rec_path, "train")
    cal_path = os.path.join(args.out, "calibrated.json")
    with open(cal_path, "w") as f:
        json.dump({"obs": obs.tolist(), "success": success.tolist(),
                   "counts": counts.tolist(), "source": rec_path,
                   "created": time.strftime("%Y-%m-%d %H:%M")}, f, indent=2)

    print(f"\nrecordings: {rec_path}")
    print(f"calibrated model (from train split): {cal_path}")
    print("\nhow often each REAL source named the true fault (train split, both severities):")
    for k in range(N_SOURCES):
        c = counts[k].sum(axis=0)
        acc = np.trace(c) / max(c.sum(), 1)
        per_fault = ", ".join(f"{FAULT_NAMES[f]} {c[f, f] / max(c[f].sum(), 1):.0%}"
                              for f in range(N_FAULTS))
        print(f"  {SOURCE_NAMES[k]:10s} overall {acc:.0%}  ({per_fault})")
    if tries:
        print("\nmeasured repair success (train split):")
        for f in range(1, N_FAULTS):
            row = ", ".join(f"{REPAIR_NAMES[r]} {success[f, r]:.0%}" for r in range(N_REPAIRS))
            print(f"  {FAULT_NAMES[f]:20s} {row}")


if __name__ == "__main__":
    main()