"""Backend that replays real answers recorded on the Docker testbed.

testbed/record.py runs real incidents on the containers and writes one
JSON line per recorded incident:

    {"kind": "obs", "split": "train", "fault": 3, "severity": 1, "trial": 0,
     "answers": [[3, 5, 3], [3, 3, 0], [0, 0, 3], [3, 3, 3]]}
        answers[k] = what source k actually reported, in the order it was asked

    {"kind": "repair", "split": "train", "fault": 3, "severity": 1,
     "repair": 2, "fixed": true}

During an episode the backend picks one recorded incident with the same
fault and severity and hands out its real answers in order, so the agent
sees exactly what the real sources said, including how their answers
were correlated within one incident. Repair outcomes are drawn from the
recorded real outcomes for that fault and repair.

Training is still fast, because the slow part (running Docker) happened
once, during recording.
"""
import json
from collections import defaultdict

import numpy as np

from .backends import Backend
from .constants import N_FAULTS, N_SOURCES, NO_FAULT


class ReplayBackend(Backend):
    def __init__(self, path, split="train"):
        self.path = path
        self.split = split
        self.incidents = defaultdict(list)    # (fault, sev) -> [answers, ...]
        self.pool = defaultdict(list)         # (fault, sev, source) -> all labels
        self.repairs = defaultdict(list)      # (fault, sev, repair) -> [bool, ...]
        self.repairs_any_sev = defaultdict(list)
        self.repairs_any_split = defaultdict(list)

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["kind"] == "repair":
                    # kept from every split as a last resort: with few repair
                    # trials, the test split may have none of its own
                    self.repairs_any_split[(rec["fault"], rec["repair"])].append(bool(rec["fixed"]))
                if rec.get("split", "train") != split:
                    continue
                fs = (rec["fault"], rec["severity"])
                if rec["kind"] == "obs":
                    self.incidents[fs].append(rec["answers"])
                    for k, labels in enumerate(rec["answers"]):
                        self.pool[fs + (k,)].extend(labels)
                elif rec["kind"] == "repair":
                    self.repairs[fs + (rec["repair"],)].append(bool(rec["fixed"]))
                    self.repairs_any_sev[(rec["fault"], rec["repair"])].append(bool(rec["fixed"]))

        missing = [f for f in range(N_FAULTS)
                   if not (self.incidents[(f, 0)] or self.incidents[(f, 1)])]
        if missing:
            raise ValueError(f"{path} ({split}) has no recordings for fault ids {missing}; "
                             "run testbed/record.py with more trials")

        self.degradation = np.zeros(N_SOURCES)
        self.fault = None
        self.severity = None
        self.current = None
        self.ptr = None

    # ------------------------------------------------------------------
    def set_degradation(self, degradation):
        # M' = (1 - d) M + d U is the same as: with probability d, replace the
        # real answer by a uniformly random label
        self.degradation = np.asarray(degradation, dtype=np.float64)

    def _key(self, fault, sev):
        # fall back to the other severity if one was not recorded
        return (fault, sev) if self.incidents[(fault, sev)] else (fault, 1 - sev)

    def start_incident(self, fault, severity, rng):
        self.fault = fault
        self.severity = severity
        key = self._key(fault, severity)
        options = self.incidents[key]
        self.current = options[int(rng.integers(len(options)))]
        self.ptr = [0] * N_SOURCES
        self.key = key

    def query(self, source, rng):
        answers = self.current[source]
        if self.ptr[source] < len(answers):
            label = answers[self.ptr[source]]
            self.ptr[source] += 1
        else:
            # asked more often than we recorded: draw from all real answers
            # this source gave for this fault
            label = int(rng.choice(self.pool[self.key + (source,)]))
        if self.degradation[source] > 0 and rng.random() < self.degradation[source]:
            label = int(rng.integers(N_FAULTS))
        return int(label)

    def repair(self, r, rng):
        if self.fault == NO_FAULT:
            return False
        outcomes = (self.repairs.get((self.fault, self.severity, r))
                    or self.repairs_any_sev.get((self.fault, r))
                    or self.repairs_any_split.get((self.fault, r)))
        if not outcomes:
            return False
        return bool(outcomes[int(rng.integers(len(outcomes)))])

    def summary(self):
        n_obs = sum(len(v) for v in self.incidents.values())
        n_rep = sum(len(v) for v in self.repairs.values())
        return f"{self.path} [{self.split}]: {n_obs} recorded incidents, {n_rep} recorded repairs"
