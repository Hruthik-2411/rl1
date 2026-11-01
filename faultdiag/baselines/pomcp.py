"""POMCP (Silver & Veness, 2010) for the fault-diagnosis POMDP.

Monte-Carlo tree search over action/observation histories. Each
simulation samples a hidden (fault, severity) from the current belief and
plays it forward with the agent's model.

Two small changes from the textbook version, both possible because the
hidden state is tiny:
  * we sample the root state straight from the exact belief instead of
    keeping a particle set;
  * at a new leaf, instead of a random rollout we use the value of
    "stop querying and act now" computed on the belief at that leaf
    (common.act_value with depth 1). Random rollouts were far too noisy
    here and made the planner worse than never querying at all.

Plain Python, so it is slow compared to a trained policy. That difference
is part of the story in the report: similar decisions, very different
decision-time cost.
"""
import bisect
import math
import random

import numpy as np

from faultdiag.env.constants import (
    N_FAULTS, N_SEV, N_SOURCES, N_REPAIRS, NO_FAULT, DECLARE_HEALTHY, ESCALATE,
    is_query, is_repair, repair_index,
)
from .common import act_value
from .rules import Policy


class _Node:
    __slots__ = ("n", "children")

    def __init__(self):
        self.n = 0
        self.children = None   # action -> [visits, mean value, {obs: _Node}]


class POMCP(Policy):
    name = "pomcp"

    def __init__(self, n_sims=300, max_depth=8, c_ucb=40.0, seed=0):
        self.n_sims = n_sims
        self.max_depth = max_depth
        self.c = c_ucb
        self.rand = random.Random(seed)

    # ------------------------------------------------------------------
    def reset(self, env):
        model = env.belief.model
        cfg = env.cfg
        # cumulative distributions so sampling is a bisect instead of np.choice
        self.obs_cdf = np.cumsum(model.obs, axis=-1).tolist()
        self.succ = model.success.tolist()
        self.useless = (model.success == 0).tolist()
        self.cfg = cfg

    def act(self, obs, env):
        self.reset(env)
        self.costs = [float(c) for c in env.costs]
        states = list(range(N_FAULTS * N_SEV))
        weights = env.belief.joint.ravel().tolist()

        root = _Node()
        start = (float(env.budget), int(env.t), tuple(bool(a) for a in env.available))
        for _ in range(self.n_sims):
            s = self.rand.choices(states, weights)[0]
            fault, sev = divmod(s, N_SEV)
            self._simulate(fault, sev, start, env.belief, root, 0)

        mask = env.action_masks()
        best_a, best_q = None, -math.inf
        for a, (n, q, _) in root.children.items():
            if n > 0 and mask[a] and q > best_q:
                best_a, best_q = a, q
        return int(best_a)

    # ------------------------------------------------------------------
    def _legal(self, ctx):
        budget, _, avail = ctx
        acts = [k for k in range(N_SOURCES)
                if self.cfg.allow_queries and avail[k] and self.costs[k] <= budget + 1e-9]
        acts += list(range(N_SOURCES, N_SOURCES + N_REPAIRS))
        acts += [DECLARE_HEALTHY, ESCALATE]
        return acts

    def _step(self, fault, sev, ctx, a):
        """Generative model: returns (obs, reward, done, new_ctx)."""
        cfg = self.cfg
        budget, t, avail = ctx
        if is_query(a):
            u = self.rand.random()
            o = min(bisect.bisect_left(self.obs_cdf[a][sev][fault], u), N_FAULTS - 1)
            dt = cfg.latency[a]
            r = -cfg.query_cost_weight * self.costs[a] - cfg.downtime_cost * dt
            new = (budget - self.costs[a], t + dt, avail)
            done = False
        elif is_repair(a):
            ri = repair_index(a)
            fixed = self.rand.random() < self.succ[fault][ri]
            dt = cfg.repair_time
            r = -cfg.repair_cost[ri] - cfg.downtime_cost * dt
            if self.useless[fault][ri]:
                r -= cfg.disruption_penalty
            if fixed:
                r += cfg.reward_fix
            new = (budget, t + dt, avail)
            done = fixed
            o = int(fixed)
        elif a == DECLARE_HEALTHY:
            r = cfg.reward_fix if fault == NO_FAULT else -cfg.miss_penalty
            return None, r, True, ctx
        else:
            return None, -cfg.escalate_cost, True, ctx

        if not done and new[1] >= cfg.max_time:
            r -= cfg.timeout_penalty
            done = True
        return o, r, done, new

    def _simulate(self, fault, sev, ctx, belief, node, depth):
        if node.children is None or depth >= self.max_depth:
            if node.children is None:
                node.children = {a: [0, 0.0, {}] for a in self._legal(ctx)}
            node.n += 1
            return act_value(belief, self.cfg, depth=1)[0]

        # UCB1 over actions, untried actions first
        log_n = math.log(node.n + 1)
        best_a, best_u = None, -math.inf
        for a, (n, q, _) in node.children.items():
            u = math.inf if n == 0 else q + self.c * math.sqrt(log_n / n)
            if u > best_u:
                best_a, best_u = a, u

        o, r, done, new_ctx = self._step(fault, sev, ctx, best_a)
        stats = node.children[best_a]
        if done:
            ret = r
        else:
            nb = belief.copy()
            if is_query(best_a):
                nb.update_label(best_a, o)
            else:
                nb.update_repair(repair_index(best_a), False)
            child = stats[2].get(o)
            if child is None:
                child = stats[2][o] = _Node()
            ret = r + self._simulate(fault, sev, new_ctx, nb, child, depth + 1)

        node.n += 1
        stats[0] += 1
        stats[1] += (ret - stats[1]) / stats[0]
        return ret
