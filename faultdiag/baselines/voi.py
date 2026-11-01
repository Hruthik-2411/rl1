"""Myopic value-of-information policy (decision-theoretic troubleshooting).

At each step compare
    acting now:        V_act(b)
    one more query k:  E_o[V_act(b_o)] - lambda*cost_k - d*latency_k
and do whichever is larger. It looks only one query ahead, so it can
miss plans like "cheap test first, then the decisive one".
"""
from .common import act_value, expected_value_after_query, query_penalty, valid_queries
from .rules import Policy


class GreedyVOI(Policy):
    name = "greedy_voi"

    def __init__(self, depth=2, margin=0.5):
        self.depth = depth
        self.margin = margin

    def act(self, obs, env):
        cfg = env.cfg
        v_now, a_now = act_value(env.belief, cfg, self.depth)
        best_q, best_gain = None, self.margin
        for k in valid_queries(env):
            v_k = expected_value_after_query(env.belief, k, cfg, self.depth) - query_penalty(env, k)
            gain = v_k - v_now
            if gain > best_gain:
                best_q, best_gain = k, gain
        return a_now if best_q is None else best_q
