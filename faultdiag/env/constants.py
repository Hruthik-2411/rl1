"""Names and indices shared by the simulator, the baselines and the Docker testbed.

Keeping them in one place means the action ids used by a trained policy
mean exactly the same thing in the simulator and on the real containers.
"""

# Hidden faults -----------------------------------------------------------
NO_FAULT = 0
CPU_OVERLOAD = 1
MEMORY_LEAK = 2
NETWORK_DEGRADATION = 3
SERVICE_CRASH = 4
DB_FAILURE = 5

FAULT_NAMES = [
    "no_fault",
    "cpu_overload",
    "memory_leak",
    "network_degradation",
    "service_crash",
    "db_failure",
]
N_FAULTS = len(FAULT_NAMES)

# Severity is a second hidden variable. Low-severity faults are harder to
# see from the cheap sources.
SEVERITIES = ["low", "high"]
N_SEV = len(SEVERITIES)

# Information sources -------------------------------------------------------
TELEMETRY = 0
PROBE = 1
LOGS = 2
DEEP = 3

SOURCE_NAMES = ["telemetry", "probe", "logs", "deep"]
N_SOURCES = len(SOURCE_NAMES)

# Recovery actions ---------------------------------------------------------
RESTART_SERVICE = 0
SCALE_OUT = 1
REROUTE_TRAFFIC = 2
KILL_LEAKY_PROCESS = 3
FAILOVER_DB = 4

REPAIR_NAMES = ["restart_service", "scale_out", "reroute_traffic",
                "kill_leaky_process", "failover_db"]
N_REPAIRS = len(REPAIR_NAMES)

# Flat action space ----------------------------------------------------------
#   0..3   query a source
#   4..8   try a repair
#   9      declare the system healthy (close the incident)
#   10     escalate to a human operator
QUERY_ACTIONS = list(range(0, N_SOURCES))
REPAIR_ACTIONS = list(range(N_SOURCES, N_SOURCES + N_REPAIRS))
DECLARE_HEALTHY = N_SOURCES + N_REPAIRS
ESCALATE = DECLARE_HEALTHY + 1
N_ACTIONS = ESCALATE + 1

ACTION_NAMES = (
    [f"query_{s}" for s in SOURCE_NAMES]
    + REPAIR_NAMES
    + ["declare_healthy", "escalate"]
)


def is_query(a):
    return 0 <= a < N_SOURCES


def is_repair(a):
    return N_SOURCES <= a < N_SOURCES + N_REPAIRS


def repair_index(a):
    return a - N_SOURCES


def query_action(source):
    return source


def repair_action(r):
    return N_SOURCES + r
