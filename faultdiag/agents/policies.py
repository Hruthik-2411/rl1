"""Wrap trained SB3 models so they look like the baseline policies.

Every policy (rule, planner or network) exposes reset(env) and
act(obs, env) -> action, which keeps the evaluation loop identical.
"""
import json
import os

import numpy as np

from faultdiag.baselines.rules import Policy
from faultdiag.env import EnvConfig


class SB3Policy(Policy):
    def __init__(self, run_dir, deterministic=True):
        with open(os.path.join(run_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.name = self.meta["label"]
        self.algo = self.meta["algo"]
        self.seed = self.meta["seed"]
        self.train_cfg = EnvConfig.from_dict(self.meta["env"])
        self.deterministic = deterministic
        path = os.path.join(run_dir, "model.zip")

        if self.algo == "maskppo":
            from sb3_contrib import MaskablePPO
            self.model = MaskablePPO.load(path, device="cpu")
        elif self.algo == "recppo":
            from sb3_contrib import RecurrentPPO
            self.model = RecurrentPPO.load(path, device="cpu")
        elif self.algo == "dqn":
            from stable_baselines3 import DQN
            self.model = DQN.load(path, device="cpu")
        elif self.algo == "ppo":
            from stable_baselines3 import PPO
            self.model = PPO.load(path, device="cpu")
        else:
            raise ValueError(self.algo)
        self._lstm = None
        self._start = True

    def reset(self, env):
        self._lstm = None
        self._start = True

    def act(self, obs, env):
        if self.algo == "maskppo":
            a, _ = self.model.predict(obs, action_masks=env.action_masks(),
                                      deterministic=self.deterministic)
        elif self.algo == "recppo":
            a, self._lstm = self.model.predict(obs, state=self._lstm,
                                               episode_start=np.array([self._start]),
                                               deterministic=self.deterministic)
            self._start = False
        elif self.algo == "dqn":
            # trained with a penalty for invalid actions; at test time we
            # simply never pick a masked action
            import torch
            with torch.no_grad():
                t = torch.as_tensor(obs[None], dtype=torch.float32)
                q = self.model.q_net(t).numpy()[0]
            q[~env.action_masks()] = -np.inf
            a = int(np.argmax(q))
        else:
            a, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(a)


def load_policy(spec):
    """`spec` is a baseline name ('greedy_voi'), a baseline with its first
    parameter ('entropy_threshold:0.2', 'pomcp:1000') or a run directory."""
    from faultdiag.baselines import BASELINES
    if spec in BASELINES:
        return BASELINES[spec]()
    if ":" in spec and spec.split(":")[0] in BASELINES:
        name, val = spec.split(":", 1)
        val = float(val)
        pol = BASELINES[name](int(val) if name == "pomcp" else val)
        pol.name = f"{name}_{val:g}"
        return pol
    if os.path.isdir(spec):
        return SB3Policy(spec)
    raise ValueError(f"don't know how to load policy '{spec}'")
