"""Simple hand-written policies used as reference points."""
import numpy as np

from faultdiag.env.constants import (
    DEEP, NO_FAULT, DECLARE_HEALTHY, repair_action,
)
from .common import act_value, valid_queries


class Policy:
    name = "policy"

    def reset(self, env):
        pass

    def act(self, obs, env):
        raise NotImplementedError


class Oracle(Policy):
    """Cheats: reads the true fault and applies the best repair straight away.
    Gives an upper bound on what any policy could score."""
    name = "oracle"

    def act(self, obs, env):
        f = env.fault
        env.first_guess = f   # it "diagnoses" perfectly by construction
        if f == NO_FAULT:
            return DECLARE_HEALTHY
        succ = env.true_model.success[f]
        return repair_action(int(np.argmax(succ)))


class NeverAcquire(Policy):
    """Acts on the prior plus the free telemetry reading only."""
    name = "never_acquire"

    def act(self, obs, env):
        return act_value(env.belief, env.cfg)[1]


class AlwaysDeep(Policy):
    """Keeps buying the most accurate source until it is confident or broke."""
    name = "always_deep"

    def __init__(self, max_deep=3, confident=0.99):
        self.max_deep = max_deep
        self.confident = confident

    def act(self, obs, env):
        if (DEEP in valid_queries(env) and env.counts[DEEP] < self.max_deep
                and env.belief.marginal.max() < self.confident):
            return DEEP
        return act_value(env.belief, env.cfg)[1]


class RandomAcquire(Policy):
    """Queries a random affordable source while unsure, then acts greedily."""
    name = "random_acquire"

    def __init__(self, p_query=0.7, confident=0.9, seed=0):
        self.p_query = p_query
        self.confident = confident
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        qs = valid_queries(env)
        if qs and env.belief.marginal.max() < self.confident and self.rng.random() < self.p_query:
            return int(self.rng.choice(qs))
        return act_value(env.belief, env.cfg)[1]


class EntropyThreshold(Policy):
    """Query the cheapest source (each at most `per_source` times) while the
    normalised entropy of the belief is above tau."""
    name = "entropy_threshold"

    def __init__(self, tau=0.35, per_source=2):
        self.tau = tau
        self.per_source = per_source

    def act(self, obs, env):
        if env.belief.normalised_entropy() > self.tau:
            qs = [k for k in valid_queries(env) if env.counts[k] < self.per_source]
            if qs:
                return min(qs, key=lambda k: env.costs[k])
        return act_value(env.belief, env.cfg)[1]
