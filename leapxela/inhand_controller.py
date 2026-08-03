"""
In-hand manipulation controller for the LeapXELA hand.

Phases: hold an open preshape while the object settles, smoothstep-close to a
grasp pose, then perturb the grasp targets with Ornstein-Uhlenbeck noise
(clipped to actuator ranges). The 16-D position targets are the actions
logged for JEPA conditioning.
"""

from __future__ import annotations

import mujoco as mj
import numpy as np

PRESHAPE = {
    "rf_mcp": 0.1, "rf_rot": -0.3, "rf_pip": 0.5, "rf_dip": 0.35,
    "mf_mcp": 0.1, "mf_rot": 0.0, "mf_pip": 0.5, "mf_dip": 0.35,
    "if_mcp": 0.1, "if_rot": 0.3, "if_pip": 0.5, "if_dip": 0.35,
    "th_cmc": 1.2, "th_axl": 0.2, "th_mcp": 0.3, "th_ipl": 0.3,
}
GRASP = {
    "rf_mcp": 1.3, "rf_rot": -0.3, "rf_pip": 0.7, "rf_dip": 0.5,
    "mf_mcp": 1.3, "mf_rot": 0.0, "mf_pip": 0.7, "mf_dip": 0.5,
    "if_mcp": 1.3, "if_rot": 0.3, "if_pip": 0.7, "if_dip": 0.5,
    "th_cmc": 1.6, "th_axl": 0.3, "th_mcp": 1.0, "th_ipl": 0.8,
}
# Relative OU amplitude per joint type (suffix of the joint name).
PERTURB_SCALE = {"mcp": 1.0, "rot": 0.5, "pip": 1.0, "dip": 1.0, "cmc": 0.7, "axl": 0.5, "ipl": 1.0}


class OUPerturbation:
    def __init__(self, dim: int, sigma: float, tau: float, rng: np.random.Generator):
        self._sigma = sigma
        self._tau = tau
        self._rng = rng
        self._state = np.zeros(dim)

    def step(self, dt: float) -> np.ndarray:
        self._state += (
            -self._state * dt / self._tau
            + self._sigma * np.sqrt(2.0 * dt / self._tau) * self._rng.standard_normal(self._state.shape)
        )
        return self._state.copy()


class InHandGraspController:
    def __init__(
        self,
        model: mj.MjModel,
        settle_duration: float,
        close_duration: float,
        perturb_sigma: float,
        perturb_tau: float,
        rng: np.random.Generator,
    ):
        self._settle = settle_duration
        self._close = close_duration

        self.joint_names = tuple(
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
        )
        self._ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = model.actuator_ctrlrange[:, 1].copy()
        self._preshape = np.array([PRESHAPE[name] for name in self.joint_names])
        self._grasp = np.array([GRASP[name] for name in self.joint_names])
        self._scale = np.array([PERTURB_SCALE[name.split("_")[1]] for name in self.joint_names])
        self._ou = OUPerturbation(model.nu, perturb_sigma, perturb_tau, rng)

    @property
    def perturb_start(self) -> float:
        return self._settle + self._close

    def compute_ctrl(self, t: float, dt: float) -> np.ndarray:
        if t < self._settle:
            return self._preshape.copy()
        if t < self._settle + self._close:
            u = (t - self._settle) / self._close
            s = 3.0 * u * u - 2.0 * u * u * u
            return self._preshape + s * (self._grasp - self._preshape)
        target = self._grasp + self._scale * self._ou.step(dt)
        return np.clip(target, self._ctrl_lo, self._ctrl_hi)
