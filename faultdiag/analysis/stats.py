"""Summary statistics with bootstrap confidence intervals.

For RL methods we have several training seeds, so the uncertainty that
matters is across seeds: we take the mean of each seed and report the
interquartile mean (IQM) of those with a bootstrap CI, following
Agarwal et al. (2021). Baselines have a single "seed", so their CI comes
from bootstrapping episodes instead.
"""
import numpy as np
import pandas as pd

METRICS = ["return", "success", "diag_correct", "query_cost", "n_queries", "n_repairs", "time"]


def iqm(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 4:
        return float(x.mean())
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean())


def bootstrap_ci(x, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(stat(x)), float(stat(x))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    stats = np.array([stat(x[i]) for i in idx])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def summarise(df, metrics=METRICS, by=("method", "scenario")):
    out = []
    for keys, g in df.groupby(list(by)):
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        seeds = g["run_seed"].unique()
        row["n_seeds"] = len(seeds)
        row["n_episodes"] = len(g)
        for m in metrics:
            vals = g[m].astype(float)
            if len(seeds) >= 3:
                per_seed = vals.groupby(g["run_seed"]).mean().values
                point = iqm(per_seed)
                lo, hi = bootstrap_ci(per_seed, stat=iqm)
            else:
                point = float(vals.mean())
                lo, hi = bootstrap_ci(vals.values)
            row[m] = point
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        out.append(row)
    return pd.DataFrame(out)


def to_markdown(summary, metrics=("return", "success", "diag_correct", "query_cost", "n_queries", "time")):
    lines = ["| method | scenario | " + " | ".join(metrics) + " |",
             "|---|---|" + "---|" * len(metrics)]
    for _, r in summary.iterrows():
        cells = []
        for m in metrics:
            pct = m in ("success", "diag_correct")
            f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.2f}")
            cells.append(f"{f(r[m])} [{f(r[m + '_lo'])}, {f(r[m + '_hi'])}]")
        lines.append(f"| {r['method']} | {r['scenario']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paired_difference(df, a, b, metric="return", scenario=None):
    """Mean per-episode difference a - b on the same incidents, with a CI."""
    d = df if scenario is None else df[df["scenario"] == scenario]
    pa = d[d["method"] == a].groupby("episode")[metric].mean()
    pb = d[d["method"] == b].groupby("episode")[metric].mean()
    diff = (pa - pb).dropna().values
    lo, hi = bootstrap_ci(diff)
    return float(diff.mean()), lo, hi
