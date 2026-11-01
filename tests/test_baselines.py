import numpy as np
import pytest

from faultdiag.agents.evaluate import run_episodes
from faultdiag.baselines import BASELINES
from faultdiag.env import EnvConfig


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_baseline_runs(name):
    pol = BASELINES[name]() if name != "pomcp" else BASELINES[name](n_sims=50)
    rows = run_episodes(pol, EnvConfig(), n_episodes=20 if name != "pomcp" else 5)
    assert all(np.isfinite(r["return"]) for r in rows)


def test_oracle_beats_never_acquire():
    cfg = EnvConfig()
    oracle = np.mean([r["return"] for r in run_episodes(BASELINES["oracle"](), cfg, 200)])
    never = np.mean([r["return"] for r in run_episodes(BASELINES["never_acquire"](), cfg, 200)])
    assert oracle > never + 20


def test_information_is_worth_paying_for():
    # if this fails, the cost settings make querying pointless and the
    # whole study loses its point
    cfg = EnvConfig()
    voi = np.mean([r["return"] for r in run_episodes(BASELINES["greedy_voi"](), cfg, 300)])
    never = np.mean([r["return"] for r in run_episodes(BASELINES["never_acquire"](), cfg, 300)])
    assert voi > never
