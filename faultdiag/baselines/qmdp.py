"""QMDP and Act-Then-Measure.

QMDP pretends the fault becomes known after this step, so it computes
Q(f, a) for every fault and averages over the belief. It therefore never
values a query for the information it brings: it treats any cheap action
(a query included) as a way to "wait one step and then know everything".
That is the classic weakness of QMDP and makes a useful contrast with
policies that reason about what a source can actually tell them.

Act-Then-Measure (Krale et al., ICAPS 2023) keeps QMDP's choice of
control action but adds a separate decision about measuring. Here the
measuring value is the expected value of perfect information (EVPI)
scaled by how much of the uncertainty a given source is expected to
remove. That scaling is our adaptation for multiple noisy sources.
"""
import numpy as np

from faultdiag.env.constants import (
    N_FAULTS, N_SOURCES, N_REPAIRS, N_ACTIONS, NO_FAULT, DECLARE_HEALTHY,
    ESCALATE, repair_action,
)
from .common import (expected_entropy_after_query, query_penalty,
                     repair_step_cost, valid_queries)
from .rules import Policy


def qmdp_table(env):
    """Q[f, a] assuming the fault is known from the next step on."""
    cfg = env.cfg
    succ = env.belief.model.success
    step = np.stack([repair_step_cost(cfg, succ, r) for r in range(N_REPAIRS)], axis=1)

    # value of a known fault: keep applying the best repair until it works
    V = np.zeros(N_FAULTS)
    for f in range(N_FAULTS):
        if f == NO_FAULT:
            V[f] = cfg.reward_fix
            continue
        best = -cfg.escalate_cost
        for r in range(N_REPAIRS):
            p = succ[f, r]
            if p > 0:
                best = max(best, (p * cfg.reward_fix - step[f, r]) / p)
        V[f] = best

    Q = np.full((N_FAULTS, N_ACTIONS), -1e9)
    for f in range(N_FAULTS):
        for k in range(N_SOURCES):
            Q[f, k] = -query_penalty(env, k) + V[f]
        for r in range(N_REPAIRS):
            p = succ[f, r]
            Q[f, repair_action(r)] = -step[f, r] + p * cfg.reward_fix + (1 - p) * V[f]
        Q[f, DECLARE_HEALTHY] = cfg.reward_fix if f == NO_FAULT else -cfg.miss_penalty
        Q[f, ESCALATE] = -cfg.escalate_cost
    return Q


class QMDP(Policy):
    name = "qmdp"

    def act(self, obs, env):
        q = env.belief.marginal @ qmdp_table(env)
        q[~env.action_masks()] = -np.inf
        return int(np.argmax(q))


class ActThenMeasure(Policy):
    name = "act_then_measure"

    def act(self, obs, env):
        Q = qmdp_table(env)
        b = env.belief.marginal
        control = list(range(N_SOURCES, N_ACTIONS))
        q_ctrl = b @ Q[:, control]
        a_ctrl = control[int(np.argmax(q_ctrl))]

        # value of knowing the fault exactly before choosing the control action
        evpi = float(b @ Q[:, control].max(axis=1) - q_ctrl.max())
        h = env.belief.entropy()
        if h < 1e-6 or evpi <= 0:
            return a_ctrl

        best_k, best_v = None, 0.0
        for k in valid_queries(env):
            frac = (h - expected_entropy_after_query(env.belief, k)) / h
            v = evpi * frac - query_penalty(env, k)
            if v > best_v:
                best_k, best_v = k, v
        return a_ctrl if best_k is None else best_k
