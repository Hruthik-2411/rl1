"""Inject faults into the testbed, undo them, and run repairs.

Run on its own to try faults by hand:

    python -m faultdiag.testbed.control status
    python -m faultdiag.testbed.control inject cpu_overload
    python -m faultdiag.testbed.control heal

Where each fault lives, and so which repair can remove it:

    cpu_overload         heavy job in the fd-batch container on the API's CPU -> scale_out
    memory_leak          inside the API process                               -> restart, kill_leaky
    network_degradation  delay on the gateway's link (via fd-netctl)          -> reroute
    service_crash        API container killed                                 -> restart
    db_failure           database container stopped                           -> failover_db

A restart only clears what is inside the API container, like in a real
deployment, so it is no longer a fix for everything.
"""
import os
import sys
import time

import docker
import requests

from faultdiag.env.constants import (
    FAULT_NAMES, NO_FAULT, CPU_OVERLOAD, MEMORY_LEAK, NETWORK_DEGRADATION,
    SERVICE_CRASH, DB_FAILURE,
    RESTART_SERVICE, SCALE_OUT, REROUTE_TRAFFIC, KILL_LEAKY_PROCESS, FAILOVER_DB,
)

API = "fd-api"
DB = "fd-db"
BATCH = "fd-batch"        # noisy neighbour on the API's CPU
NETCTL = "fd-netctl"      # lives in the gateway's network namespace
GATEWAY = "fd-gateway"
GATEWAY_URL = os.environ.get("FD_GATEWAY", "http://localhost:8080")
API_URL = os.environ.get("FD_API", "http://localhost:5001")

MEM_LIMIT_OK_MB = 350
LATENCY_OK_MS = 300

# '[s]tress-ng' matches stress-ng but not the grep/pkill command line itself
STRESS_PATTERN = "'[s]tress-ng'"


class Testbed:
    def __init__(self):
        self.client = docker.from_env()
        self.ncpu = int(self.client.info().get("NCPU", 1))
        self._check_netctl()

    # ------------------------------------------------------------------
    def container(self, name):
        c = self.client.containers.get(name)
        c.reload()
        return c

    def running(self, name):
        try:
            return self.container(name).status == "running"
        except docker.errors.NotFound:
            return False

    def sh(self, name, cmd, detach=False):
        """Run a shell command inside a container, ignore failures."""
        try:
            res = self.container(name).exec_run(["sh", "-c", cmd], detach=detach)
            if detach:
                return ""
            return res.output.decode(errors="ignore")
        except (docker.errors.APIError, docker.errors.NotFound):
            return ""

    def _check_netctl(self):
        """fd-netctl must share the gateway's CURRENT network.

        If the gateway was restarted on its own, fd-netctl is left attached
        to the old, dead network and network faults would silently do
        nothing.
        """
        if not (self.running(NETCTL) and self.running(GATEWAY)):
            return
        nets = self.container(GATEWAY).attrs["NetworkSettings"]["Networks"]
        gw_macs = {(n.get("MacAddress") or "").lower() for n in nets.values()}
        mac = self.sh(NETCTL, "cat /sys/class/net/eth0/address").strip().lower()
        if mac and mac not in gw_macs:
            raise SystemExit("fd-netctl is attached to an old gateway network. Run:\n"
                             "    docker compose up -d --force-recreate")

    def wait_healthy(self, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.user_facing_ok():
                return True
            time.sleep(1)
        return False

    def update(self, name, **data):
        """docker update through the raw API.

        docker-compose's `cpus: 1.0` is stored as NanoCpus, and docker-py's
        update() can only send the older cpu_period/cpu_quota, which Docker
        then refuses. Sending the fields ourselves avoids that.
        """
        api = self.client.api
        url = api._url("/containers/{0}/update", self.container(name).id)
        api._result(api._post_json(url, data=data), True)

    def set_cpus(self, n):
        """API CPU limit. 1 = only CPU 0 (shared with the batch job),
        more = CPUs 0..n-1, which gets it out of the batch job's way."""
        n = max(1, min(n, self.ncpu))
        cpuset = "0" if n == 1 else f"0-{n - 1}"
        self.update(API, NanoCpus=int(n * 1e9), CpusetCpus=cpuset)

    def cpus(self):
        nano = self.container(API).attrs["HostConfig"].get("NanoCpus") or 0
        return nano / 1e9

    # ------------------------------------------------------------------
    # faults
    # ------------------------------------------------------------------
    def inject(self, fault, severity=1):
        high = severity == 1
        if fault == NO_FAULT:
            return
        if fault == CPU_OVERLOAD:
            # a heavier job also gets a bigger share of the CPU it shares with the API
            self.update(BATCH, CpuShares=8192 if high else 2048)
            workers = 2 if high else 1
            self.sh(BATCH, f"stress-ng --cpu {workers} --timeout 900s", detach=True)
        elif fault == MEMORY_LEAK:
            mb = 15 if high else 8
            requests.post(f"{API_URL}/admin/leak", params={"mb": mb}, timeout=3)
        elif fault == NETWORK_DEGRADATION:
            delay = "400ms 80ms loss 5%" if high else "180ms 40ms"
            self.sh(NETCTL, f"tc qdisc add dev eth0 root netem delay {delay}")
        elif fault == SERVICE_CRASH:
            self.container(API).kill()
        elif fault == DB_FAILURE:
            self.container(DB).stop(timeout=2)
        else:
            raise ValueError(fault)

    def heal(self):
        """Put everything back to a healthy state."""
        for name in (DB, BATCH, NETCTL):
            if not self.running(name):
                self.container(name).start()
        if not self.running(API):
            self.container(API).start()
            time.sleep(3)
        self.sh(BATCH, f"pkill -f {STRESS_PATTERN}; true")
        self.sh(NETCTL, "tc qdisc del dev eth0 root 2>/dev/null; true")
        self.sh(API, f"tc qdisc del dev eth0 root 2>/dev/null; pkill -f {STRESS_PATTERN}; true")
        try:
            requests.post(f"{API_URL}/admin/free", timeout=3)
        except requests.RequestException:
            pass
        try:
            self.set_cpus(1)        # undo a previous scale_out
        except docker.errors.APIError:
            pass
        if self.wait_healthy(30):
            return True
        # something is stuck (e.g. a run was stopped halfway): restart the
        # service, which clears every fault inside it, and wait once more
        self.container(API).restart(timeout=3)
        return self.wait_healthy(60)

    def stress_running(self, name=BATCH):
        """Is a stress-ng process alive in that container? Zombie entries
        (state Z, already dead) are skipped."""
        out = self.sh(name, f"ps -eo stat=,args= | grep -v '^ *Z' | grep {STRESS_PATTERN}")
        return bool(out.strip())

    def netem_on_link(self):
        return "netem" in self.sh(NETCTL, "tc qdisc show dev eth0")

    # ------------------------------------------------------------------
    # repairs (same order as the simulator's repair actions)
    # ------------------------------------------------------------------
    def repair(self, r):
        if r == RESTART_SERVICE:
            self.container(API).restart(timeout=3)
        elif r == SCALE_OUT:
            # three CPUs instead of the one it shares with the batch job
            if self.running(API):
                self.set_cpus(3)
        elif r == REROUTE_TRAFFIC:
            # stand-in for sending users over a healthy path: drop the
            # degraded qdisc on the gateway link
            self.sh(NETCTL, "tc qdisc del dev eth0 root 2>/dev/null; true")
        elif r == KILL_LEAKY_PROCESS:
            # kills the leaking work inside the service, not other jobs on the machine
            self.sh(API, f"pkill -f {STRESS_PATTERN}; true")
            try:
                requests.post(f"{API_URL}/admin/free", timeout=3)
            except requests.RequestException:
                pass
        elif r == FAILOVER_DB:
            db = self.container(DB)
            if db.status == "running":
                db.restart(timeout=2)
            else:
                db.start()
        else:
            raise ValueError(r)

    # ------------------------------------------------------------------
    def rss(self):
        try:
            return requests.get(f"{API_URL}/health", timeout=3).json().get("rss_mb")
        except (requests.RequestException, ValueError):
            return None

    def leaking(self, wait=3, grow_mb=10):
        """Memory still growing? Two readings of the API's memory, a few s apart."""
        a = self.rss()
        time.sleep(wait)
        b = self.rss()
        if a is None or b is None:
            return None
        return b - a >= grow_mb

    def fault_gone(self, fault):
        """Is the cause of the fault really gone, or only hidden for a moment?

        A memory leak at 8 MB/s does not hurt users in the first seconds, so
        the user-facing check alone would call every repair a success. A real
        engineer would not close the incident while the leak is still growing.
        """
        if fault == CPU_OVERLOAD:
            # fixed when the API no longer has to share its only CPU with the job
            return not self.stress_running(BATCH) or self.cpus() >= 2
        if fault == MEMORY_LEAK:
            return self.leaking() is False
        if fault == NETWORK_DEGRADATION:
            return not self.netem_on_link()
        if fault in (SERVICE_CRASH, DB_FAILURE):
            return self.running(API) and self.running(DB)
        return True

    def user_facing_ok(self, n=5):
        """What 'fixed' means: requests through the gateway succeed and are
        fast, and the service is not about to run out of memory."""
        if not (self.running(API) and self.running(DB)):
            return False
        lat = []
        for _ in range(n):
            try:
                t0 = time.time()
                r = requests.get(f"{GATEWAY_URL}/items", timeout=3)
                lat.append((time.time() - t0) * 1000)
                if r.status_code != 200:
                    return False
            except requests.RequestException:
                return False
        lat.sort()
        if lat[len(lat) // 2] > LATENCY_OK_MS:
            return False
        try:
            mem = requests.get(f"{API_URL}/health", timeout=3).json().get("rss_mb", 0)
        except (requests.RequestException, ValueError):
            return False
        return mem < MEM_LIMIT_OK_MB


def main(argv):
    tb = Testbed()
    if not argv or argv[0] == "status":
        print("api:", tb.running(API), "| db:", tb.running(DB), "| batch:", tb.running(BATCH),
              "| netctl:", tb.running(NETCTL), "| healthy:", tb.user_facing_ok())
    elif argv[0] == "inject":
        tb.inject(FAULT_NAMES.index(argv[1]), severity=int(argv[2]) if len(argv) > 2 else 1)
        print("injected", argv[1])
    elif argv[0] == "heal":
        print("healthy" if tb.heal() else "still unhealthy, check `docker compose logs api gateway`")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])