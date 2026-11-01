"""Observation and repair models.

obs[k, s, f, o] = P(source k reports label o | true fault f, severity s)
success[f, r]   = P(repair r fixes fault f)

The numbers are hand-set so that no single cheap source can tell every
fault apart: telemetry mixes up network and database problems, the probe
is blind to resource faults, logs miss network trouble, and the deep
diagnostic sees everything but costs a lot. If the sources were all good
at everything the best policy would be trivial and the project boring.
"""
import json

import numpy as np

from .constants import N_FAULTS, N_SEV, N_SOURCES, N_REPAIRS, NO_FAULT

# rows = true fault, cols = reported label
# order: no_fault, cpu, memory, network, crash, db
TELEMETRY_HIGH = [
    [0.80, 0.05, 0.05, 0.04, 0.03, 0.03],
    [0.05, 0.80, 0.08, 0.03, 0.02, 0.02],
    [0.05, 0.10, 0.75, 0.03, 0.04, 0.03],
    [0.10, 0.03, 0.02, 0.40, 0.05, 0.40],
    [0.10, 0.03, 0.03, 0.14, 0.50, 0.20],
    [0.08, 0.02, 0.02, 0.38, 0.10, 0.40],
]
PROBE_HIGH = [
    [0.85, 0.04, 0.04, 0.02, 0.02, 0.03],
    [0.40, 0.35, 0.15, 0.04, 0.03, 0.03],
    [0.40, 0.15, 0.35, 0.04, 0.03, 0.03],
    [0.15, 0.05, 0.05, 0.50, 0.15, 0.10],
    [0.03, 0.01, 0.01, 0.03, 0.90, 0.02],
    [0.05, 0.02, 0.02, 0.06, 0.10, 0.75],
]
LOGS_HIGH = [
    [0.80, 0.04, 0.04, 0.04, 0.04, 0.04],
    [0.20, 0.50, 0.20, 0.04, 0.03, 0.03],
    [0.10, 0.15, 0.65, 0.03, 0.04, 0.03],
    [0.35, 0.05, 0.05, 0.40, 0.08, 0.07],
    [0.02, 0.02, 0.03, 0.03, 0.85, 0.05],
    [0.02, 0.02, 0.02, 0.04, 0.05, 0.85],
]


def _deep_matrix(acc=0.95):
    off = (1.0 - acc) / (N_FAULTS - 1)
    m = np.full((N_FAULTS, N_FAULTS), off)
    np.fill_diagonal(m, acc)
    return m


# how much a low-severity fault looks like "nothing is wrong" to each source
LOW_SEVERITY_MASK = [0.30, 0.20, 0.20, 0.05]

# rows = fault, cols = repair
# order: restart, scale_out, reroute, kill_leaky, failover_db
SUCCESS = [
    [0.00, 0.00, 0.00, 0.00, 0.00],   # no fault: nothing to fix
    [0.30, 0.90, 0.20, 0.30, 0.00],   # cpu overload
    [0.60, 0.20, 0.00, 0.90, 0.00],   # memory leak
    [0.00, 0.10, 0.85, 0.00, 0.05],   # network degradation
    [0.90, 0.30, 0.40, 0.10, 0.00],   # service crash
    [0.10, 0.00, 0.00, 0.00, 0.90],   # db failure
]


def _normalise_rows(m):
    m = np.asarray(m, dtype=np.float64)
    return m / m.sum(axis=-1, keepdims=True)


def nominal_obs_model():
    high = [np.array(TELEMETRY_HIGH), np.array(PROBE_HIGH),
            np.array(LOGS_HIGH), _deep_matrix()]
    obs = np.zeros((N_SOURCES, N_SEV, N_FAULTS, N_FAULTS))
    for k in range(N_SOURCES):
        hi = _normalise_rows(high[k])
        lo = hi.copy()
        a = LOW_SEVERITY_MASK[k]
        for f in range(N_FAULTS):
            if f == NO_FAULT:
                continue
            lo[f] = (1 - a) * hi[f]
            lo[f, NO_FAULT] += a
        obs[k, 0] = lo
        obs[k, 1] = hi
    return obs


def degrade(obs, degradation):
    """Blend each source's matrices towards uniform by its degradation level."""
    out = obs.copy()
    uniform = np.full(N_FAULTS, 1.0 / N_FAULTS)
    for k, d in enumerate(degradation):
        if d > 0:
            out[k] = (1 - d) * obs[k] + d * uniform
    return out


class DiagModel:
    """Everything a planner needs to know about the world.

    The simulator keeps a 'true' DiagModel (possibly degraded); the agent's
    belief filter and the model-based baselines use a nominal one unless
    told otherwise.
    """

    def __init__(self, obs=None, success=None):
        self.obs = nominal_obs_model() if obs is None else np.asarray(obs, dtype=np.float64)
        self.success = np.asarray(SUCCESS if success is None else success, dtype=np.float64)
        assert self.obs.shape == (N_SOURCES, N_SEV, N_FAULTS, N_FAULTS)
        assert self.success.shape == (N_FAULTS, N_REPAIRS)

    def with_degradation(self, degradation):
        return DiagModel(degrade(self.obs, degradation), self.success)

    def sample_label(self, k, fault, sev, rng):
        return int(rng.choice(N_FAULTS, p=self.obs[k, sev, fault]))

    def sample_repair(self, r, fault, rng):
        return bool(rng.random() < self.success[fault, r])

    def source_accuracy(self, k):
        """Average diagonal mass, i.e. how often the source is right."""
        return float(np.mean([np.trace(self.obs[k, s]) / N_FAULTS for s in range(N_SEV)]))

    # --- calibration file written by testbed/record.py
    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            d = json.load(f)
        obs = np.asarray(d["obs"], dtype=np.float64)
        if obs.ndim == 3:
            # testbed data has no severity split, use it for both levels
            obs = np.stack([obs, obs], axis=1)
        success = d.get("success")
        return cls(obs, success)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"obs": self.obs.tolist(), "success": self.success.tolist()}, f, indent=2)
