"""Helpers shared by the model-based baselines.

The key function is `act_value`: the expected return if, from belief b,
we stop gathering information and only try repairs / declare / escalate.
It is a small exhaustive search (depth 2 by default), which is the
"expected cost of repair" idea from Heckerman et al. (1995) written as a
value instead of a cost.
"""
import numpy as np

from faultdiag.env.constants import (
    N_SOURCES, N_REPAIRS, NO_FAULT, DECLARE_HEALTHY, ESCALATE, repair_action,
)


def valid_queries(env):
    mask = env.action_masks()
    return [k for k in range(N_SOURCES) if mask[k]]


def query_penalty(env, k):
    cfg = env.cfg
    return cfg.query_cost_weight * env.costs[k] + cfg.downtime_cost * cfg.latency[k]


def repair_step_cost(cfg, success, r):
    """Cost of trying repair r, per fault (vector over faults)."""
    useless = (success[:, r] == 0).astype(float)
    return cfg.repair_cost[r] + cfg.downtime_cost * cfg.repair_time + cfg.disruption_penalty * useless


def act_value(belief, cfg, depth=2):
    """Best (value, action) when no more queries are allowed."""
    p_f = belief.marginal
    best_v = -cfg.escalate_cost
    best_a = ESCALATE

    # close the incident
    v = p_f[NO_FAULT] * cfg.reward_fix - (1 - p_f[NO_FAULT]) * cfg.miss_penalty
    if v > best_v:
        best_v, best_a = v, DECLARE_HEALTHY

    succ = belief.model.success
    for r in range(N_REPAIRS):
        p = float(np.dot(p_f, succ[:, r]))
        step_cost = float(np.dot(p_f, repair_step_cost(cfg, succ, r)))
        if depth > 1 and p < 0.999:
            nxt = belief.copy()
            nxt.update_repair(r, False)
            v_fail, _ = act_value(nxt, cfg, depth - 1)
        else:
            v_fail = -cfg.escalate_cost
        v = -step_cost + p * cfg.reward_fix + (1 - p) * v_fail
        if v > best_v:
            best_v, best_a = v, repair_action(r)
    return best_v, best_a


def expected_value_after_query(belief, k, cfg, depth=2):
    """E_o[ act_value(b after seeing o from source k) ]."""
    pred = belief.predictive(k)
    total = 0.0
    for label, p in enumerate(pred):
        if p < 1e-6:
            continue
        v, _ = act_value(belief.posterior_if(k, label), cfg, depth)
        total += p * v
    return total


def expected_entropy_after_query(belief, k):
    pred = belief.predictive(k)
    h = 0.0
    for label, p in enumerate(pred):
        if p < 1e-6:
            continue
        h += p * belief.posterior_if(k, label).entropy()
    return h
