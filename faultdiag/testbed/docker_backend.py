"""Backend that answers the agent's queries with real measurements.

Each source runs real commands against the containers and turns what it
sees into one of the six fault labels, using simple rules an on-call
engineer might apply. The rules are deliberately imperfect, exactly like
the cheap sources in the simulator; testbed/record.py measures how
imperfect they really are.
"""
import json
import time

import requests

from faultdiag.env.backends import Backend
from faultdiag.env.constants import (
    NO_FAULT, CPU_OVERLOAD, MEMORY_LEAK, NETWORK_DEGRADATION, SERVICE_CRASH,
    DB_FAILURE, TELEMETRY, PROBE, LOGS, DEEP,
)
from .control import API, DB, BATCH, API_URL, GATEWAY_URL, Testbed

SETTLE_SECONDS = 15     # let a fault show up in metrics and logs
AFTER_REPAIR_SECONDS = 5


def cpu_percent(stats):
    cpu = stats["cpu_stats"]
    pre = stats["precpu_stats"]
    d_cpu = cpu["cpu_usage"]["total_usage"] - pre["cpu_usage"]["total_usage"]
    d_sys = cpu.get("system_cpu_usage", 0) - pre.get("system_cpu_usage", 0)
    n = cpu.get("online_cpus") or len(cpu["cpu_usage"].get("percpu_usage") or [1])
    return 100.0 * d_cpu / d_sys * n if d_sys > 0 else 0.0


class DockerBackend(Backend):
    def __init__(self, settle=SETTLE_SECONDS):
        self.tb = Testbed()
        self.settle = settle
        self.fault = None
        self.t_start = 0.0

    # ------------------------------------------------------------------
    def start_incident(self, fault, severity, rng):
        if not self.tb.heal():
            # better to stop than to record a fault on top of a broken system
            raise RuntimeError("testbed did not become healthy; check `docker compose ps` "
                               "and `docker compose logs api gateway`")
        self.fault = fault
        self.t_start = time.time()
        self.tb.inject(fault, severity)
        time.sleep(self.settle)

    def end_incident(self):
        self.tb.heal()

    def repair(self, r, rng):
        self.tb.repair(r)
        time.sleep(AFTER_REPAIR_SECONDS)
        # fixed = users are fine AND the cause is really gone
        return self.tb.user_facing_ok() and self.tb.fault_gone(self.fault)

    def query(self, source, rng):
        return {TELEMETRY: self.telemetry, PROBE: self.probe,
                LOGS: self.logs, DEEP: self.deep}[source]()

    # ------------------------------------------------------------------
    # the four sources
    # ------------------------------------------------------------------
    def _gateway_latency(self):
        """Median latency over 3 requests (ms) and whether any failed."""
        lat, failed = [], False
        for _ in range(3):
            t0 = time.time()
            try:
                r = requests.get(f"{GATEWAY_URL}/items", timeout=3)
                failed |= r.status_code >= 500
            except requests.RequestException:
                failed = True
            lat.append((time.time() - t0) * 1000)
        return sorted(lat)[1], failed

    def telemetry(self):
        """Container metrics plus end-to-end latency, what a dashboard shows."""
        if not self.tb.running(API):
            return SERVICE_CRASH
        stats = self.tb.container(API).stats(stream=False)
        # the dashboard shows the whole machine's CPU, not just the service:
        # the service plus the batch job that shares its CPU
        cpu = cpu_percent(stats)
        if self.tb.running(BATCH):
            cpu += cpu_percent(self.tb.container(BATCH).stats(stream=False))
        mem = stats["memory_stats"]
        # same number `docker stats` shows: usage minus reclaimable file cache
        cache = mem.get("stats", {}).get("inactive_file",
                                         mem.get("stats", {}).get("total_inactive_file", 0))
        mem_mb = (mem.get("usage", 0) - cache) / 2**20
        latency, failed = self._gateway_latency()
        if cpu > 85:
            return CPU_OVERLOAD
        if mem_mb > 200:
            return MEMORY_LEAK
        if latency > 400 or failed:
            # can't tell a slow network from a struggling database here
            return NETWORK_DEGRADATION if latency > 400 else DB_FAILURE
        return NO_FAULT

    def probe(self):
        """Hit the service's own /health endpoint."""
        t0 = time.time()
        try:
            r = requests.get(f"{API_URL}/health", timeout=2.5)
            body = r.json()
        except requests.RequestException:
            return SERVICE_CRASH
        except ValueError:
            return SERVICE_CRASH
        ms = (time.time() - t0) * 1000
        if not body.get("db", True):
            return DB_FAILURE
        if ms > 800:
            return NETWORK_DEGRADATION
        if ms > 250:
            return CPU_OVERLOAD
        return NO_FAULT

    def logs(self):
        """Grep the service's JSON logs since this incident started (at most 30 s)."""
        if not self.tb.running(API):
            return SERVICE_CRASH
        # not further back than the incident start, or errors left over from
        # the previous incident (e.g. a stopped database) leak into this one
        since = max(int(time.time()) - 30, int(self.t_start))
        raw = self.tb.container(API).logs(since=since).decode(errors="ignore")
        events = {}
        for line in raw.splitlines():
            try:
                ev = json.loads(line).get("event")
            except ValueError:
                continue
            events[ev] = events.get(ev, 0) + 1
        if events.get("db_error", 0) >= 2:
            return DB_FAILURE
        if events.get("memory_high", 0) >= 1:
            return MEMORY_LEAK
        if events.get("slow_request", 0) >= 3:
            return CPU_OVERLOAD     # slow requests: usually CPU, sometimes network
        return NO_FAULT

    def deep(self):
        """Look at everything, including things only an admin can see."""
        if not self.tb.running(DB):
            return DB_FAILURE
        if not self.tb.running(API):
            return SERVICE_CRASH
        if self.tb.netem_on_link():
            return NETWORK_DEGRADATION
        if self.tb.stress_running(BATCH):
            return CPU_OVERLOAD
        # an admin watches memory over a few seconds: still growing = leak
        if self.tb.leaking(wait=2, grow_mb=5) or (self.tb.rss() or 0) > 250:
            return MEMORY_LEAK
        return NO_FAULT