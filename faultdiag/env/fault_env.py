"""FaultDiagEnv: one episode = one incident.

The agent starts with a free telemetry reading, then at every step either
pays for another source, tries a repair, declares the system healthy or
escalates to a human. The true fault is hidden; it is only used to decide
outcomes and to score the episode afterwards.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .backends import SimBackend
from .belief import Belief
from .config import EnvConfig
from .constants import (
    N_FAULTS, N_SOURCES, N_REPAIRS, N_ACTIONS, NO_FAULT, TELEMETRY,
    DECLARE_HEALTHY, ESCALATE, is_query, is_repair, repair_index,
)
from .model import DiagModel

MAX_COST_FOR_NORM = 30.0
MAX_BUDGET_FOR_NORM = 60.0
MAX_COUNT_FOR_NORM = 5.0


class FaultDiagEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config=None, backend=None, base_model=None):
        super().__init__()
        self.cfg = config if config is not None else EnvConfig()
        if backend is None and self.cfg.replay_file:
            # answers come from real recordings made on the Docker testbed
            from .replay import ReplayBackend
            backend = ReplayBackend(self.cfg.replay_file, split=self.cfg.replay_split)
        self.backend = backend if backend is not None else SimBackend()

        if base_model is not None:
            self.base_model = base_model
        elif self.cfg.sources_file:
            self.base_model = DiagModel.from_file(self.cfg.sources_file)
        else:
            self.base_model = DiagModel()

        self.action_space = spaces.Discrete(N_ACTIONS)
        obs_dim = len(self._build_obs(dummy=True))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng()
        self._done = True

    # ------------------------------------------------------------------
    # episode setup
    # ------------------------------------------------------------------
    def _uniform(self, rng_range):
        lo, hi = rng_range
        return lo if lo == hi else float(self._rng.uniform(lo, hi))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        cfg = self.cfg

        # sample the information environment for this episode
        self.costs = np.array([c * self._uniform(cfg.cost_mult_range) for c in cfg.base_cost])
        self.degradation = np.array([self._uniform(cfg.degradation_range) for _ in range(N_SOURCES)])
        if cfg.degradation is not None:
            self.degradation = np.asarray(cfg.degradation, dtype=np.float64)
        if "degradation" in options:
            self.degradation = np.asarray(options["degradation"], dtype=np.float64)
        self.available = np.array([self._rng.random() >= p for p in cfg.p_unavailable])
        self.budget_max = self._uniform(cfg.budget_range)
        self.budget = self.budget_max

        self.true_model = self.base_model.with_degradation(self.degradation)
        belief_model = self.true_model if cfg.belief_knows_degradation else self.base_model
        self.belief = Belief(belief_model, cfg.fault_prior, cfg.p_high_severity)

        # sample the hidden incident
        prior = np.asarray(cfg.fault_prior) / np.sum(cfg.fault_prior)
        self.fault = int(options.get("fault", self._rng.choice(N_FAULTS, p=prior)))
        self.severity = int(options.get("severity", self._rng.random() < cfg.p_high_severity))

        if isinstance(self.backend, SimBackend):
            self.backend.set_model(self.true_model)
        if hasattr(self.backend, "set_degradation"):
            self.backend.set_degradation(self.degradation)
        self.backend.start_incident(self.fault, self.severity, self._rng)

        # bookkeeping
        self.t = 0
        self.n_steps = 0
        self.counts = np.zeros(N_SOURCES, dtype=np.int64)
        self.last_label = np.full(N_SOURCES, -1, dtype=np.int64)
        self.tried = np.zeros(N_REPAIRS, dtype=np.int64)
        self.query_spent = 0.0
        self.first_guess = None
        self.history = []
        self.free_used = False
        self._done = False

        if cfg.free_initial_telemetry and self.available[TELEMETRY]:
            label = self.backend.query(TELEMETRY, self._rng)
            self.belief.update_label(TELEMETRY, label)
            self.counts[TELEMETRY] += 1
            self.last_label[TELEMETRY] = label
            self.free_used = True
            self.history.append(("free_telemetry", label))

        return self._build_obs(), {}

    # ------------------------------------------------------------------
    # masking
    # ------------------------------------------------------------------
    def action_masks(self):
        mask = np.ones(N_ACTIONS, dtype=bool)
        for k in range(N_SOURCES):
            ok = (self.cfg.allow_queries and self.available[k]
                  and self.costs[k] <= self.budget + 1e-9)
            mask[k] = ok
        return mask

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action):
        assert not self._done, "call reset() first"
        action = int(action)
        cfg = self.cfg
        reward = 0.0
        terminated = False
        outcome = None

        if not self.action_masks()[action]:
            # only reachable for agents that ignore the mask (DQN, RecurrentPPO)
            dt = 1
            reward -= cfg.invalid_action_penalty
            self.history.append(("invalid", action))

        elif is_query(action):
            k = action
            cost = float(self.costs[k])
            dt = cfg.latency[k]
            h_before = self.belief.entropy()
            label = self.backend.query(k, self._rng)
            self.belief.update_label(k, label)
            self.budget -= cost
            self.query_spent += cost
            self.counts[k] += 1
            self.last_label[k] = label
            reward -= cfg.query_cost_weight * cost
            if cfg.info_gain_bonus > 0:
                reward += cfg.info_gain_bonus * (h_before - self.belief.entropy())
            self.history.append((f"query_{k}", label))

        elif is_repair(action):
            r = repair_index(action)
            self._note_first_guess()
            dt = cfg.repair_time
            reward -= cfg.repair_cost[r]
            if self.true_model.success[self.fault, r] == 0:
                reward -= cfg.disruption_penalty
            fixed = self.backend.repair(r, self._rng)
            self.tried[r] = 1
            self.history.append((f"repair_{r}", fixed))
            if fixed:
                reward += cfg.reward_fix
                terminated = True
                outcome = "fixed"
            else:
                self.belief.update_repair(r, False)

        elif action == DECLARE_HEALTHY:
            self._note_first_guess()
            dt = 0
            terminated = True
            if self.fault == NO_FAULT:
                reward += cfg.reward_fix
                outcome = "closed_correctly"
            else:
                reward -= cfg.miss_penalty
                outcome = "missed_fault"
            self.history.append(("declare_healthy", None))

        elif action == ESCALATE:
            self._note_first_guess()
            dt = 0
            terminated = True
            reward -= cfg.escalate_cost
            outcome = "escalated"
            self.history.append(("escalate", None))
        else:
            raise ValueError(action)

        # the incident costs time while it stays open
        reward -= cfg.downtime_cost * dt
        self.t += dt
        self.n_steps += 1

        if not terminated and self.t >= cfg.max_time:
            reward -= cfg.timeout_penalty
            terminated = True
            outcome = "timeout"

        if cfg.fail_source is not None and self.n_steps == cfg.fail_at_step:
            self.available[cfg.fail_source] = False

        info = {}
        if terminated:
            self._done = True
            self.backend.end_incident()
            info = self._episode_info(outcome)

        return self._build_obs(), float(reward), terminated, False, info

    def _note_first_guess(self):
        if self.first_guess is None:
            self.first_guess = self.belief.most_likely()

    def _episode_info(self, outcome):
        success = outcome in ("fixed", "closed_correctly")
        paid = self.counts.copy()
        if self.free_used:
            paid[TELEMETRY] -= 1
        return {
            "outcome": outcome,
            "success": success,
            "fault": self.fault,
            "severity": self.severity,
            "diag_correct": self.first_guess == self.fault,
            "query_cost": self.query_spent,
            "n_queries": int(paid.sum()),
            "source_counts": paid.tolist(),
            "n_repairs": int(sum(1 for h in self.history if h[0].startswith("repair"))),
            "time": self.t,
            "budget_max": self.budget_max,
        }

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def _build_obs(self, dummy=False):
        cfg = self.cfg
        parts = []
        if dummy:
            belief = np.zeros(N_FAULTS)
            ent = 0.0
            budget = budget_max = t = 0.0
            counts = np.zeros(N_SOURCES)
            last = np.full(N_SOURCES, -1)
            tried = np.zeros(N_REPAIRS)
            costs = np.zeros(N_SOURCES)
            avail = np.zeros(N_SOURCES)
        else:
            belief = self.belief.marginal
            ent = self.belief.normalised_entropy()
            budget, budget_max, t = self.budget, self.budget_max, self.t
            counts, last, tried = self.counts, self.last_label, self.tried
            costs, avail = self.costs, self.available

        if cfg.obs_mode == "belief":
            parts.append(belief)
            parts.append([ent])
        elif cfg.obs_mode != "raw":
            raise ValueError(cfg.obs_mode)

        parts.append([budget / max(budget_max, 1e-6) if budget_max else 0.0,
                      budget_max / MAX_BUDGET_FOR_NORM,
                      t / cfg.max_time])
        parts.append(np.minimum(counts / MAX_COUNT_FOR_NORM, 1.0))
        onehots = np.zeros((N_SOURCES, N_FAULTS + 1))
        for k in range(N_SOURCES):
            onehots[k, last[k] if last[k] >= 0 else N_FAULTS] = 1.0
        parts.append(onehots.ravel())
        parts.append(tried)
        if cfg.include_context:
            parts.append(np.minimum(costs / MAX_COST_FOR_NORM, 1.0))
            parts.append(np.asarray(avail, dtype=np.float64))

        obs = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    def render(self):
        from .constants import ACTION_NAMES, FAULT_NAMES
        lines = [f"true fault: {FAULT_NAMES[self.fault]} (sev={self.severity})",
                 f"t={self.t} budget={self.budget:.1f}/{self.budget_max:.1f}",
                 "belief: " + ", ".join(f"{FAULT_NAMES[i]}={p:.2f}"
                                        for i, p in enumerate(self.belief.marginal))]
        for h in self.history:
            lines.append(f"  {h}")
        return "\n".join(lines)


def make_env(config=None, **kw):
    return FaultDiagEnv(config=config, **kw)
