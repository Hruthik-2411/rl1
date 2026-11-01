"""Evaluation scenarios.

A scenario is a set of EnvConfig overrides describing the *world*
(costs, reliabilities, budget, availability). Agents are always trained
on the default randomised config and then evaluated on these.
"""
from faultdiag.env.config import EnvConfig
from faultdiag.env.constants import SOURCE_NAMES, DEEP, PROBE

# a calm, fixed world: nominal costs, no degradation, everything available
NOMINAL = dict(cost_mult_range=[1.0, 1.0], degradation_range=[0.0, 0.0],
               p_unavailable=[0.0, 0.0, 0.0, 0.0], budget_range=[25.0, 25.0])


def _nominal(**extra):
    d = dict(NOMINAL)
    d.update(extra)
    return d


SCENARIOS = {
    "train_dist": {},
    "nominal": _nominal(),
    "ood": dict(cost_mult_range=[2.5, 3.5], degradation_range=[0.4, 0.6],
                budget_range=[5.0, 60.0], p_unavailable=[0.0, 0.2, 0.2, 0.4]),
    "fail_deep_mid": _nominal(fail_source=DEEP, fail_at_step=1),
    "fail_probe_mid": _nominal(fail_source=PROBE, fail_at_step=1),
    "deep_unavailable": _nominal(p_unavailable=[0.0, 0.0, 0.0, 1.0]),
}

for b in [5, 10, 20, 40, 60]:
    SCENARIOS[f"budget_{b}"] = _nominal(budget_range=[float(b), float(b)])

for m in [0.5, 1.0, 2.0, 3.0]:
    SCENARIOS[f"cost_x{m:g}"] = _nominal(cost_mult_range=[m, m])

for k, src in enumerate(SOURCE_NAMES[:3]):          # deep stays reliable
    for d in [0.0, 0.2, 0.4, 0.6, 0.8]:
        deg = [0.0, 0.0, 0.0, 0.0]
        deg[k] = d
        SCENARIOS[f"degrade_{src}_{d:g}"] = _nominal(degradation=deg)

EXPERIMENTS = {
    "E1": ["train_dist", "nominal"],
    "E2": ["nominal"],                                  # run once per lambda
    "E3": [f"budget_{b}" for b in [5, 10, 20, 40, 60]],
    "E3b": [f"cost_x{m:g}" for m in [0.5, 1.0, 2.0, 3.0]],
    "E4": [f"degrade_{s}_{d:g}" for s in SOURCE_NAMES[:3] for d in [0.0, 0.2, 0.4, 0.6, 0.8]],
    "E5": ["nominal", "fail_deep_mid", "fail_probe_mid", "deep_unavailable"],
    "E6": ["train_dist", "ood"],
    "E7": ["train_dist"],
}

# fields that describe the agent, not the world. When an RL model is
# evaluated these come from the config it was trained with.
AGENT_FIELDS = ("obs_mode", "include_context", "belief_knows_degradation",
                "allow_queries", "free_initial_telemetry", "sources_file")


def scenario_config(name, base=None, agent_cfg=None, **extra):
    cfg = (base or EnvConfig()).copy(**SCENARIOS[name])
    if agent_cfg is not None:
        keep = {f: getattr(agent_cfg, f) for f in AGENT_FIELDS}
        # the world model given in the evaluation config wins, so baselines
        # and agents are judged against the same source accuracies
        if base is not None and base.sources_file:
            keep["sources_file"] = base.sources_file
        cfg = cfg.copy(**keep)
    if extra:
        cfg = cfg.copy(**extra)
    return cfg
