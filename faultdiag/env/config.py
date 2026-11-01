"""Environment configuration.

Every knob of the simulator lives in EnvConfig. Ranges given as [lo, hi]
are sampled fresh at every reset (domain randomisation); a range with
lo == hi is simply a fixed value.
"""
from dataclasses import dataclass, field, asdict, replace
from typing import List, Optional
import json

import yaml


@dataclass
class EnvConfig:
    # --- incident generation
    fault_prior: List[float] = field(
        default_factory=lambda: [0.15, 0.17, 0.17, 0.17, 0.17, 0.17])
    p_high_severity: float = 0.5

    # --- information sources (order: telemetry, probe, logs, deep)
    base_cost: List[float] = field(default_factory=lambda: [1.0, 3.0, 5.0, 15.0])
    latency: List[int] = field(default_factory=lambda: [1, 1, 2, 3])
    # multiplier applied to base_cost, sampled per episode
    cost_mult_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
    # degradation d in [0, 1]: the source's confusion matrix is blended
    # with a uniform matrix, M' = (1 - d) M + d U. Sampled per source.
    degradation_range: List[float] = field(default_factory=lambda: [0.0, 0.35])
    # fixed per-source degradation, overrides the range (used in tests)
    degradation: Optional[List[float]] = None
    # probability that a source is unavailable for the whole episode
    p_unavailable: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.05, 0.15])
    # optional mid-episode failure of one source
    fail_source: Optional[int] = None
    fail_at_step: int = 2
    # optional file with confusion matrices measured on the testbed; the
    # agent's belief and the planners use it instead of the hand-set tables
    sources_file: Optional[str] = None
    # optional file of real answers recorded on the Docker testbed. When set,
    # every query answer and repair outcome is replayed from real recordings
    # instead of being drawn from the tables (see faultdiag/env/replay.py)
    replay_file: Optional[str] = None
    replay_split: str = "train"     # "train" or "test" part of the recordings

    # --- diagnostic budget (money for queries only)
    budget_range: List[float] = field(default_factory=lambda: [10.0, 40.0])

    # --- repairs
    # order: restart, scale_out, reroute, kill_leaky, failover_db
    repair_cost: List[float] = field(default_factory=lambda: [5.0, 12.0, 8.0, 5.0, 20.0])
    repair_time: int = 3
    # extra cost when a repair cannot possibly help the true fault
    # (restarting a healthy service, failing over a healthy database ...)
    disruption_penalty: float = 20.0
    max_time: int = 30

    # --- reward
    reward_fix: float = 100.0      # incident resolved (or correctly closed)
    downtime_cost: float = 3.0     # per time unit the incident stays open
    query_cost_weight: float = 1.0  # lambda in the report
    miss_penalty: float = 100.0    # declared healthy while a fault was present
    escalate_cost: float = 60.0    # handing over to a human
    timeout_penalty: float = 80.0  # ran out of time
    invalid_action_penalty: float = 5.0
    info_gain_bonus: float = 0.0   # only used in the ablation

    # --- what the agent gets to see
    obs_mode: str = "belief"        # "belief" or "raw"
    include_context: bool = True    # current costs + availability in the obs
    belief_knows_degradation: bool = False
    allow_queries: bool = True      # False -> recovery-only baseline
    free_initial_telemetry: bool = True

    def copy(self, **overrides):
        return replace(self, **overrides)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
