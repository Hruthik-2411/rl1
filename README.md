# Cost-Aware Active Fault Diagnosis and Recovery with RL

Term project: Nallapaneni Vaibhav (135), Palivela Sai Hruthik (141), Yalamarthi Madhav (163).

An agent faces a system incident whose cause is hidden. At each step it either pays to query
one of four information sources (telemetry, health probe, logs, deep diagnostic), each with its
own cost, delay and reliability, or it acts: tries a repair, declares the system healthy, or
escalates to a human. We compare RL agents (MaskablePPO, RecurrentPPO, DQN) against rule-based,
decision-theoretic and planning baselines, stress-test them when sources degrade, fail or get
more expensive, and check the result on a real Docker microservice testbed.

Training and experiments run in plain Python on your laptop, either on the simulator
(sections 3–6) or on real data recorded from a Docker microservice testbed (section 7).

---

## 1. Folder layout

```
faultdiag-rl/
├── faultdiag/
│   ├── env/            simulator: constants, config, model (matrices), belief, env, backends, replay
│   ├── baselines/      oracle, rules, greedy VOI, QMDP, act-then-measure, POMCP
│   ├── agents/         train.py, evaluate.py, policies.py (wraps SB3 models)
│   ├── analysis/       stats.py (IQM + bootstrap CIs), plots.py (all figures)
│   ├── testbed/        control.py (faults/repairs), docker_backend.py, record.py, live_eval.py
│   └── experiments.py  evaluation scenarios E1–E7
├── testbed/            the microservice app: api (Flask), nginx, locust load generator
├── configs/            default.yaml, real_train.yaml, real_test.yaml
├── scripts/            run_all.py (whole pipeline), show_episodes.py
├── tests/              pytest unit tests
└── docker-compose.yml  the real testbed containers (section 7)
```

Outputs appear in `models/`, `results/` and `figures/` as you run things.

---

## 2. Setup (once)

You need Python 3.10, 3.11 or 3.12. Open a terminal in the `faultdiag-rl` folder.

**Windows (PowerShell or cmd)**

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
pytest -q
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
pytest -q
```

All tests should pass. Activate the environment (`.venv\Scripts\activate` or
`source .venv/bin/activate`) every time you open a new terminal.

If you have an NVIDIA GPU you can install the normal CUDA build of torch instead, but it is not
needed: the networks are small and SB3 runs these fine on CPU.

---

## 3. Quick check (about 15 minutes)

```bash
python scripts/run_all.py --quick
```

This trains one seed of every agent for 100k steps, runs all experiments with fewer episodes
and writes figures to `figures/`. The numbers are not final, but it proves the whole pipeline
works on your laptop.

Look at a few episodes to get a feel for the problem:

```bash
python scripts/show_episodes.py greedy_voi --n 3
python scripts/show_episodes.py models/maskppo_s0 --n 3 --fault network_degradation
```

---

## 4. Full pipeline

```bash
python scripts/run_all.py                                 # 3 seeds, 1M steps each
python scripts/run_all.py --seeds 0 1 2 3 4 5 6 7 8 9     # final run for the report
```

Models that already exist are skipped, so you can stop it (Ctrl+C) and start it again.

**Time.** One 1M-step PPO run takes roughly 10–25 minutes on a laptop CPU (DQN is slower).
With 12 agent variants × 3 seeds that is a long night. Split the seeds across the three
laptops, then copy all the `models/` sub-folders onto one machine:

```bash
python scripts/run_all.py --seeds 0 1 2       # laptop 1
python scripts/run_all.py --seeds 3 4 5       # laptop 2
python scripts/run_all.py --seeds 6 7 8 9     # laptop 3
```

On the machine with all models, run the experiments and figures only:

```bash
python scripts/run_all.py --seeds 0 1 2 3 4 5 6 7 8 9 --skip-train
```

Watch training live:

```bash
tensorboard --logdir models/tb
# open http://localhost:6006
```

---

## 5. Individual commands

Train:

```bash
python -m faultdiag.agents.train --algo maskppo --seed 0                 # main method
python -m faultdiag.agents.train --algo recppo --obs-mode raw --seed 0   # learned memory, no belief
python -m faultdiag.agents.train --algo dqn --seed 0
python -m faultdiag.agents.train --algo maskppo --lam 2 --tag lam2 --seed 0
python -m faultdiag.agents.train --help                                  # all options
```

Evaluate (one CSV row per episode, same incidents for every method):

```bash
python -m faultdiag.agents.evaluate --experiment E1 --methods never_acquire greedy_voi pomcp "models/maskppo_s*" --episodes 1000 --out results/E1.csv
```

Baseline names: `oracle never_acquire always_deep random_acquire entropy_threshold greedy_voi qmdp act_then_measure pomcp`.
A parameter can be attached: `entropy_threshold:0.2`, `pomcp:1000` (number of simulations).
Patterns such as `"models/maskppo_s*"` are expanded by the script itself, so they work on Windows too.

Figures and tables:

```bash
python -m faultdiag.analysis.plots --results results --out figures
```

---

## 6. Experiments

| ID | Question | Output |
|----|----------|--------|
| E1 | Which method resolves incidents best overall? | `figures/E1_table.md`, `E1_returns.png` |
| E2 | Trade-off between query cost and success (sweep λ) | `E2_pareto.png` |
| E3 | Effect of the diagnostic budget; E3b: effect of query cost | `E3_budget.png`, `E3b_cost.png` |
| E4 | Does the policy stop trusting a source that degrades? | `E4_reliability.png` |
| E5 | What happens when a source fails mid-incident? | `E5_failure.png` |
| E6 | Generalisation to unseen costs, budgets, reliabilities | `E6_generalisation.png` |
| E7 | Ablations: no belief, no cost context, no masking, info-gain bonus | `E7_ablation.png` |
| E8 | Does it work on real containers? | `results_real/E8_live.csv` (section 7) |

Training curves: `figures/learning_curves.png`.

---

## 7. Training and testing on REAL data from the Docker testbed

Sections 3–6 use the simulator, whose source answers come from hand-set tables. This section
replaces those with **real answers from real systems**: a small shop (nginx gateway → Flask
API → Postgres + Redis, with Locust sending traffic) runs in Docker, faults are injected for
real (stress-ng CPU load, a memory leak, tc/netem network delay, killing the API, stopping
the database), and the four sources run real commands (`docker stats`, the `/health`
endpoint, `docker logs`, a deep check of processes and network rules).

Training directly against the live containers would take weeks (each real incident takes
30–60 s, and RL needs tens of thousands). So the work is split:

1. **Record** real incidents once (a few hours): every answer every real source gave, and
   whether each real repair worked.
2. **Train** on those recordings: every query answer and repair outcome the agent sees is a
   real recorded one (`faultdiag/env/replay.py`). Training is as fast as before.
3. **Test** on held-out real recordings the agent never trained on, and finally on brand-new
   **live** incidents in Docker.

Docker Desktop must be running. All commands are run from the project folder with the
`faultdiag` environment active.

### Step 1: start the testbed and check it

```bash
docker compose up -d --build
python -m faultdiag.testbed.control status        # healthy: True
python -m faultdiag.testbed.control inject cpu_overload
python -m faultdiag.testbed.control status        # healthy: False
python -m faultdiag.testbed.control heal          # healthy
```

### Step 2: record real data

```bash
python -m faultdiag.testbed.record --trials 10 --repeats 3 --repair-trials 2     # ~2.5–3 h
python -m faultdiag.testbed.record --trials 5 --repair-trials 1                  # quicker, ~1–1.5 h
```

- For each fault and severity it injects the fault, waits 15 s for it to show, asks each
  real source 3 times, and saves the answers. Then it tries every real repair on fresh
  injections and saves whether it worked.
- 20% of the trials are kept aside as a **test** split that training never sees.
- Everything is appended to `data/real/recordings.jsonl` as it goes, so you can stop and
  re-run safely (re-running adds more data).
- At the end it writes `data/real/calibrated.json` (real source accuracies and repair success
  rates, from the train split only) and prints how often each real source named the true
  fault. **Save this printout for the report.**

### Step 3: train and test on the real recordings

```bash
python scripts/run_all.py --real --quick
```

This trains all agents on the real recordings (train split), runs experiments E1–E7 on the
held-out real recordings (test split), and writes `models_real/`, `results_real/` and
`figures_real/`. The simulated results in `models/`, `results/` and `figures/` are untouched.
Drop `--quick` for the full run.

In the real-data experiments, budget, cost and source-failure settings (E3, E3b, E5) are
applied exactly as before. In E4 and E6 extra unreliability is added on top of the real answers
(a real answer is replaced by a random one with the given probability).

### Step 4: test live on the real system (E8)

```bash
python -m faultdiag.testbed.live_eval --methods models_real/maskppo_s0 greedy_voi never_acquire --calibrated data/real/calibrated.json --episodes 30 --out results_real/E8_live.csv
docker compose down
```

Each live incident takes 30–60 s (30 incidents × 3 methods ≈ 1 h). These incidents are new:
faults are injected fresh, and every query and repair runs on the containers.

### Optional: does a simulator-trained agent work on real data?

```bash
python -m faultdiag.agents.evaluate --config configs/real_test.yaml --experiment E1 --methods models/maskppo_s0 models_real/maskppo_s0 greedy_voi --episodes 200 --out results_real/sim_vs_real.csv
```

This compares the agent trained on the simulator with the one trained on real recordings, on the
same held-out real incidents (sim-to-real transfer).

### Files

| File | What it is |
|---|---|
| `data/real/recordings.jsonl` | one line per real incident: what each source said, or whether a repair worked |
| `data/real/calibrated.json` | real source accuracies and repair success rates (train split) |
| `configs/real_train.yaml` | train on real recordings (train split) |
| `configs/real_test.yaml` | test on real recordings (test split) |
| `results_real/E8_live.csv` | live incidents on the real system |

The scripts expect the gateway on `http://localhost:8080` and the API on `http://localhost:5001`.
If you changed the ports, set `FD_GATEWAY` and `FD_API` accordingly.

## 8. Troubleshooting

- **`pip install torch` fails**: check `python --version`; torch 2.4.1 needs Python 3.8–3.12.
- **`ModuleNotFoundError: faultdiag`**: you skipped `pip install -e .` or the venv is not active.
- **Training is slow**: use `--steps 500000`, or fewer seeds while developing. The final report should use ≥ 5 seeds.
- **`no methods matched`** in evaluate: the models for that pattern have not been trained yet.
- **Testbed: `docker.errors.DockerException`**: Docker Desktop is not running.
- **Testbed: network fault does nothing**: the api container needs `NET_ADMIN`; it is set in the compose file. Rebuild with `docker compose up -d --build`.
- **Testbed: memory-leak fault crashes the API instead**: lower the leak rate in `faultdiag/testbed/control.py` (`mb = 15`).
- **Recording stops halfway**: just run the same `record` command again; it appends to the existing file.
- **`no recordings for fault ids [...]`** when training on real data: run `record` with more trials.
