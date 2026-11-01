"""Exact Bayesian belief over the hidden (fault, severity) pair.

There are only 6 x 2 = 12 hidden states, so we can keep the full
posterior and update it with Bayes' rule. The agent sees the marginal
over faults plus its entropy.
"""
import numpy as np

from .constants import N_FAULTS, N_SEV


class Belief:
    def __init__(self, model, fault_prior, p_high):
        self.model = model
        prior_f = np.asarray(fault_prior, dtype=np.float64)
        prior_f = prior_f / prior_f.sum()
        prior_s = np.array([1.0 - p_high, p_high])
        self.prior = np.outer(prior_f, prior_s)   # shape (N_FAULTS, N_SEV)
        self.joint = self.prior.copy()

    def reset(self):
        self.joint = self.prior.copy()

    def copy(self):
        b = Belief.__new__(Belief)
        b.model = self.model
        b.prior = self.prior
        b.joint = self.joint.copy()
        return b

    # ------------------------------------------------------------------
    def _apply(self, likelihood):
        post = self.joint * likelihood
        z = post.sum()
        if z <= 1e-12:
            # The evidence is impossible under our model (can happen when a
            # source is degraded and we don't know it). Fall back to a
            # softened version of the old belief instead of dividing by 0.
            post = 0.5 * self.joint + 0.5 * self.prior
            z = post.sum()
        self.joint = post / z

    def likelihood_label(self, k, label):
        # obs[k, s, f, o] -> want (f, s)
        return self.model.obs[k, :, :, label].T

    def update_label(self, k, label):
        self._apply(self.likelihood_label(k, label))

    def update_repair(self, r, fixed):
        p = self.model.success[:, r][:, None]           # (N_FAULTS, 1)
        lik = p if fixed else (1.0 - p)
        self._apply(np.repeat(lik, N_SEV, axis=1))

    # ------------------------------------------------------------------
    @property
    def marginal(self):
        return self.joint.sum(axis=1)

    def entropy(self):
        p = self.marginal
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def normalised_entropy(self):
        return self.entropy() / np.log(N_FAULTS)

    def predictive(self, k):
        """P(label | belief) for source k, shape (N_FAULTS,)."""
        # sum over f, s of joint[f, s] * obs[k, s, f, :]
        return np.einsum("fs,sfo->o", self.joint, self.model.obs[k])

    def posterior_if(self, k, label):
        b = self.copy()
        b.update_label(k, label)
        return b

    def most_likely(self):
        return int(np.argmax(self.marginal))
