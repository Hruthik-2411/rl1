"""Turn the evaluation CSVs in results/ into the figures and tables for the report.

    python -m faultdiag.analysis.plots --results results --out figures

Any experiment whose CSV is missing is skipped, so you can run this
after every stage.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from faultdiag.analysis.stats import summarise, to_markdown
from faultdiag.env.constants import SOURCE_NAMES

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})

# draw the RL methods last so they sit on top
ORDER = ["never_acquire", "always_deep", "random_acquire", "entropy_threshold",
         "qmdp", "act_then_measure", "greedy_voi", "pomcp", "maskppo_noquery",
         "dqn", "recppo", "maskppo", "oracle"]


def _order(methods):
    known = [m for m in ORDER if m in methods]
    return known + sorted(m for m in methods if m not in known)


def _load(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _errbar(ax, x, s, metric, **kw):
    y = s[metric].values
    err = np.vstack([y - s[f"{metric}_lo"].values, s[f"{metric}_hi"].values - y])
    ax.errorbar(x, y, yerr=err, capsize=3, marker="o", ms=4, **kw)


def table_e1(res, out):
    df = _load(os.path.join(res, "E1.csv"))
    if df is None:
        return
    s = summarise(df)
    s["method"] = pd.Categorical(s["method"], _order(s["method"].unique()))
    s = s.sort_values(["scenario", "method"])
    s.to_csv(os.path.join(out, "E1_summary.csv"), index=False)
    with open(os.path.join(out, "E1_table.md"), "w") as f:
        f.write(to_markdown(s))

    nom = s[s["scenario"] == "nominal"] if "nominal" in set(s["scenario"]) else s
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(nom))
    ax.bar(x, nom["return"], color="#8aa9c9")
    ax.errorbar(x, nom["return"], yerr=[nom["return"] - nom["return_lo"], nom["return_hi"] - nom["return"]],
                fmt="none", ecolor="k", capsize=3)
    ax.set_xticks(x, nom["method"], rotation=35, ha="right")
    ax.set_ylabel("mean return per incident")
    ax.set_title("E1: all methods, nominal setting (95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "E1_returns.png"))
    plt.close(fig)


def pareto_e2(res, out):
    files = sorted(glob.glob(os.path.join(res, "E2_lam*.csv")))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # one point per (method family, lambda); RL runs are tagged lam<x>
    df["family"] = df["method"].str.replace(r"_lam[0-9.]+", "", regex=True)
    df["family"] = df["family"].str.replace(r"_[0-9.]+$", "", regex=True)
    s = summarise(df, by=("method", "family", "lam"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for fam, g in s.groupby("family"):
        g = g.sort_values("query_cost")
        axes[0].plot(g["query_cost"], g["success"], marker="o", label=fam)
        axes[1].plot(g["query_cost"], g["time"], marker="o", label=fam)
    axes[0].set_xlabel("mean query cost per incident")
    axes[0].set_ylabel("success rate")
    axes[1].set_xlabel("mean query cost per incident")
    axes[1].set_ylabel("time to resolve (units)")
    axes[0].legend(fontsize=8)
    fig.suptitle("E2: cost vs. performance trade-off (each point = one lambda)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "E2_pareto.png"))
    plt.close(fig)
    s.to_csv(os.path.join(out, "E2_summary.csv"), index=False)


def sweep_plot(res, out, name, pattern, xlabel, fname, title):
    df = _load(os.path.join(res, f"{name}.csv"))
    if df is None:
        return
    df["x"] = df["scenario"].str.extract(pattern).astype(float)
    s = summarise(df, by=("method", "x"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in _order(s["method"].unique()):
        g = s[s["method"] == m].sort_values("x")
        _errbar(axes[0], g["x"], g, "return", label=m)
        axes[1].plot(g["x"], g["n_queries"], marker="o", label=m)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("mean return")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("queries per incident")
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname))
    plt.close(fig)


def reliability_e4(res, out):
    df = _load(os.path.join(res, "E4.csv"))
    if df is None:
        return
    m = df["scenario"].str.extract(r"degrade_(\w+?)_([0-9.]+)")
    df["src"], df["deg"] = m[0], m[1].astype(float)
    qcols = [f"q_{s}" for s in SOURCE_NAMES]
    total = df[qcols].sum(axis=1).replace(0, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, src in zip(axes, SOURCE_NAMES[:3]):
        d = df[df["src"] == src].copy()
        d["share"] = d[f"q_{src}"] / total.loc[d.index]
        for meth in _order(d["method"].unique()):
            g = d[d["method"] == meth].groupby("deg")["share"].mean()
            ax.plot(g.index, g.values, marker="o", label=meth)
        ax.set_title(f"{src} degraded")
        ax.set_xlabel("degradation level")
    axes[0].set_ylabel(f"share of queries sent to the degraded source")
    axes[0].legend(fontsize=7)
    fig.suptitle("E4: does the policy move away from a source that becomes unreliable?")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "E4_reliability.png"))
    plt.close(fig)

    s = summarise(df, by=("method", "src", "deg"))
    s.to_csv(os.path.join(out, "E4_summary.csv"), index=False)


def grouped_bars(res, out, name, fname, title):
    df = _load(os.path.join(res, f"{name}.csv"))
    if df is None:
        return
    s = summarise(df)
    methods = _order(s["method"].unique())
    scen = list(dict.fromkeys(df["scenario"]))
    w = 0.8 / len(scen)
    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 0.9), 3.8))
    for i, sc in enumerate(scen):
        g = s[s["scenario"] == sc].set_index("method").reindex(methods)
        x = np.arange(len(methods)) + i * w
        ax.bar(x, g["return"], width=w, label=sc)
        ax.errorbar(x, g["return"], yerr=[g["return"] - g["return_lo"], g["return_hi"] - g["return"]],
                    fmt="none", ecolor="k", capsize=2, lw=0.8)
    ax.set_xticks(np.arange(len(methods)) + w * (len(scen) - 1) / 2, methods, rotation=35, ha="right")
    ax.set_ylabel("mean return")
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out, fname))
    plt.close(fig)
    s.to_csv(os.path.join(out, f"{name}_summary.csv"), index=False)


def learning_curves(models, out, window=500):
    runs = sorted(glob.glob(os.path.join(models, "*", "monitor_0.monitor.csv")))
    if not runs:
        return
    curves = {}
    for path in runs:
        label = re.sub(r"_s\d+$", "", os.path.basename(os.path.dirname(path)))
        d = pd.read_csv(path, skiprows=1)
        d["steps"] = d["l"].cumsum()
        d["smooth"] = d["r"].rolling(window, min_periods=50).mean()
        curves.setdefault(label, []).append(d)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, ds in sorted(curves.items()):
        grid = np.linspace(0, min(d["steps"].max() for d in ds), 200)
        ys = np.array([np.interp(grid, d["steps"], d["smooth"].bfill()) for d in ds])
        mu = ys.mean(0)
        ax.plot(grid * 8, mu, label=f"{label} ({len(ds)} seeds)")   # monitor sees 1 of 8 envs
        if len(ds) > 1:
            ax.fill_between(grid * 8, ys.min(0), ys.max(0), alpha=0.15)
    ax.set_xlabel("environment steps (all envs)")
    ax.set_ylabel(f"episode return (rolling mean of {window})")
    ax.legend(fontsize=7)
    ax.set_title("Learning curves")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "learning_curves.png"))
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--models", default="models")
    p.add_argument("--out", default="figures")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    table_e1(args.results, args.out)
    pareto_e2(args.results, args.out)
    sweep_plot(args.results, args.out, "E3", r"budget_([0-9.]+)", "diagnostic budget",
               "E3_budget.png", "E3: effect of the diagnostic budget")
    sweep_plot(args.results, args.out, "E3b", r"cost_x([0-9.]+)", "query cost multiplier",
               "E3b_cost.png", "E3b: effect of query cost")
    reliability_e4(args.results, args.out)
    grouped_bars(args.results, args.out, "E5", "E5_failure.png", "E5: source failure")
    grouped_bars(args.results, args.out, "E6", "E6_generalisation.png",
                 "E6: training distribution vs. out-of-distribution")
    grouped_bars(args.results, args.out, "E7", "E7_ablation.png", "E7: ablations")
    learning_curves(args.models, args.out)
    print(f"figures and tables written to {args.out}/")


if __name__ == "__main__":
    main()
