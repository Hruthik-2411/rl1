import numpy as np
import pytest

from faultdiag.env import Belief, DiagModel, EnvConfig, FaultDiagEnv
from faultdiag.env.constants import (
    DEEP, DECLARE_HEALTHY, ESCALATE, N_ACTIONS, N_FAULTS, NO_FAULT, TELEMETRY,
    CPU_OVERLOAD, repair_action, SCALE_OUT,
)


def test_matrices_are_distributions():
    m = DiagModel()
    assert np.allclose(m.obs.sum(axis=-1), 1.0)
    assert (m.obs >= 0).all()
    assert ((m.success >= 0) & (m.success <= 1)).all()


def test_belief_stays_normalised():
    b = Belief(DiagModel(), EnvConfig().fault_prior, 0.5)
    rng = np.random.default_rng(0)
    for _ in range(50):
        b.update_label(rng.integers(4), rng.integers(N_FAULTS))
        assert abs(b.joint.sum() - 1) < 1e-9
        assert (b.joint >= 0).all()


def test_deep_query_concentrates_belief():
    b = Belief(DiagModel(), EnvConfig().fault_prior, 0.5)
    b.update_label(DEEP, CPU_OVERLOAD)
    b.update_label(DEEP, CPU_OVERLOAD)
    assert b.most_likely() == CPU_OVERLOAD
    assert b.marginal[CPU_OVERLOAD] > 0.95


def test_failed_repair_rules_out_faults_it_would_fix():
    b = Belief(DiagModel(), EnvConfig().fault_prior, 0.5)
    before = b.marginal[CPU_OVERLOAD]
    b.update_repair(SCALE_OUT, fixed=False)      # scale-out fixes CPU 90% of the time
    assert b.marginal[CPU_OVERLOAD] < 0.3 * before


def test_masks_respect_budget():
    env = FaultDiagEnv(EnvConfig(budget_range=[2.0, 2.0], cost_mult_range=[1.0, 1.0],
                                 p_unavailable=[0, 0, 0, 0]))
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask[TELEMETRY]          # costs 1
    assert not mask[DEEP]           # costs 15
    assert mask[DECLARE_HEALTHY] and mask[ESCALATE]


def test_unavailable_source_is_masked():
    env = FaultDiagEnv(EnvConfig(p_unavailable=[0, 0, 0, 1.0]))
    env.reset(seed=1)
    assert not env.action_masks()[DEEP]


def test_mid_episode_failure():
    env = FaultDiagEnv(EnvConfig(fail_source=DEEP, fail_at_step=1, p_unavailable=[0, 0, 0, 0],
                                 budget_range=[50, 50]))
    env.reset(seed=2)
    assert env.action_masks()[DEEP]
    env.step(TELEMETRY)
    assert not env.action_masks()[DEEP]


def test_declare_healthy_rewards():
    cfg = EnvConfig()
    env = FaultDiagEnv(cfg)
    env.reset(seed=0, options={"fault": NO_FAULT})
    _, r, done, _, info = env.step(DECLARE_HEALTHY)
    assert done and info["success"] and r == pytest.approx(cfg.reward_fix)

    env.reset(seed=0, options={"fault": CPU_OVERLOAD})
    _, r, done, _, info = env.step(DECLARE_HEALTHY)
    assert done and not info["success"] and r == pytest.approx(-cfg.miss_penalty)


def test_query_reward_matches_hand_calculation():
    cfg = EnvConfig(cost_mult_range=[1.0, 1.0], p_unavailable=[0, 0, 0, 0], budget_range=[30, 30])
    env = FaultDiagEnv(cfg)
    env.reset(seed=3)
    _, r, _, _, _ = env.step(DEEP)
    expected = -cfg.query_cost_weight * 15.0 - cfg.downtime_cost * cfg.latency[DEEP]
    assert r == pytest.approx(expected)
    assert env.budget == pytest.approx(15.0)


def test_useless_repair_costs_extra():
    cfg = EnvConfig()
    env = FaultDiagEnv(cfg)
    env.reset(seed=4, options={"fault": NO_FAULT})
    _, r, done, _, _ = env.step(repair_action(SCALE_OUT))
    expected = -cfg.repair_cost[SCALE_OUT] - cfg.disruption_penalty - cfg.downtime_cost * cfg.repair_time
    assert not done and r == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["belief", "raw"])
def test_random_rollouts(mode):
    env = FaultDiagEnv(EnvConfig(obs_mode=mode))
    rng = np.random.default_rng(0)
    for ep in range(200):
        obs, _ = env.reset(seed=ep)
        assert obs.shape == env.observation_space.shape
        done = False
        while not done:
            a = rng.choice(np.flatnonzero(env.action_masks()))
            obs, r, done, _, info = env.step(a)
            assert np.all(obs >= 0) and np.all(obs <= 1)
        assert info["outcome"] in ("fixed", "closed_correctly", "missed_fault", "escalated", "timeout")


def test_invalid_action_is_penalised_not_crashing():
    env = FaultDiagEnv(EnvConfig(p_unavailable=[0, 0, 0, 1.0]))
    env.reset(seed=0)
    _, r, done, _, _ = env.step(DEEP)
    assert not done and r < 0
    assert N_ACTIONS == 11
