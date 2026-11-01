"""Where observations and repair outcomes come from.

The environment does the bookkeeping (budget, time, belief, reward); a
backend only answers two questions: "what does source k report?" and
"did repair r work?". SimBackend samples the answer from the model,
testbed.docker_backend.DockerBackend runs real commands on containers.
"""


class Backend:
    def start_incident(self, fault, severity, rng):
        raise NotImplementedError

    def query(self, source, rng):
        raise NotImplementedError

    def repair(self, r, rng):
        raise NotImplementedError

    def end_incident(self):
        pass


class SimBackend(Backend):
    def __init__(self):
        self.model = None
        self.fault = None
        self.severity = None
        self.fixed = False

    def set_model(self, model):
        self.model = model

    def start_incident(self, fault, severity, rng):
        self.fault = fault
        self.severity = severity
        self.fixed = False

    def query(self, source, rng):
        return self.model.sample_label(source, self.fault, self.severity, rng)

    def repair(self, r, rng):
        ok = self.model.sample_repair(r, self.fault, rng)
        if ok:
            self.fixed = True
        return ok
