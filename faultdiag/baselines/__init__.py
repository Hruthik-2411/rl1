from .rules import Oracle, NeverAcquire, AlwaysDeep, RandomAcquire, EntropyThreshold
from .voi import GreedyVOI
from .qmdp import QMDP, ActThenMeasure
from .pomcp import POMCP

BASELINES = {
    "oracle": Oracle,
    "never_acquire": NeverAcquire,
    "always_deep": AlwaysDeep,
    "random_acquire": RandomAcquire,
    "entropy_threshold": EntropyThreshold,
    "greedy_voi": GreedyVOI,
    "qmdp": QMDP,
    "act_then_measure": ActThenMeasure,
    "pomcp": POMCP,
}
