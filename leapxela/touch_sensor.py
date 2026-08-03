"""
Taxel sensors for the LeapXELA hand.

`TaxelSensor` is the interface both implementations share:
- `VirtualTaxelSensor` (this module): splats MuJoCo contact wrenches between
  the object and taxel-carrying hand bodies onto nearby taxel sites. Output is
  per-taxel force in the taxel-local frame (x/y shear, z normal, compressive
  positive).
- A future `SlideJointTaxelSensor` over leapXela_pointcloud/robot_with_sensors.xml
  would read the per-taxel 3-slide-joint deflections (uSkin-style displacement)
  from data.qpos and return the same (368, 3) contract.
"""

from __future__ import annotations

import abc
from typing import List

import mujoco as mj
import numpy as np

from leapxela.taxel_layout import N_TAXELS, TaxelLayout


class TaxelSensor(abc.ABC):
    CHANNEL_NAMES = ("shear_x", "shear_y", "normal_z")

    @abc.abstractmethod
    def update(self, data: mj.MjData) -> np.ndarray:
        """Per-taxel force (368, 3) float32 in taxel-local frames."""

    @abc.abstractmethod
    def taxel_positions(self, data: mj.MjData) -> np.ndarray:
        """World-frame taxel positions (368, 3) from forward kinematics."""

    @abc.abstractmethod
    def taxel_frames(self, data: mj.MjData) -> np.ndarray:
        """World-frame taxel rotation matrices (368, 3, 3)."""

    @abc.abstractmethod
    def set_noise_model(self, noise_std: float, gain_std: float, seed: int) -> None:
        """Enable additive noise and fixed per-taxel gain jitter."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources."""


class VirtualTaxelSensor(TaxelSensor):
    def __init__(
        self,
        model: mj.MjModel,
        layout: TaxelLayout,
        contact_geom_names: List[str],
        kernel_sigma: float,
        kernel_cutoff: float,
    ):
        self._model = model
        self._kernel_sigma = kernel_sigma
        self._kernel_cutoff = kernel_cutoff

        self._site_ids = np.array(
            [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, e.site_name) for e in layout.entries]
        )
        if np.any(self._site_ids < 0):
            raise ValueError("Scene model is missing taxel sites; build it with build_scene_xml")

        self._object_geom_ids = set()
        for name in contact_geom_names:
            geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"Contact geom '{name}' not found in model")
            self._object_geom_ids.add(geom_id)

        self._body_taxels = {}
        for flat_idx, entry in enumerate(layout.entries):
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, entry.body)
            self._body_taxels.setdefault(body_id, []).append(flat_idx)
        self._body_taxels = {
            body_id: np.asarray(idx, dtype=np.int64) for body_id, idx in self._body_taxels.items()
        }

        self._noise_std = 0.0
        self._taxel_gain = None
        self._rng = None
        self.last_total_force = np.zeros(3)

    def update(self, data: mj.MjData) -> np.ndarray:
        out = np.zeros((N_TAXELS, 3))
        total_force = np.zeros(3)
        site_pos = data.site_xpos[self._site_ids]
        site_mat = data.site_xmat[self._site_ids].reshape(-1, 3, 3)
        force6 = np.zeros(6)

        for con_idx in range(data.ncon):
            contact = data.contact[con_idx]
            body1 = self._model.geom_bodyid[contact.geom1]
            body2 = self._model.geom_bodyid[contact.geom2]
            if contact.geom1 in self._object_geom_ids and body2 in self._body_taxels:
                hand_body, hand_is_geom2 = body2, True
            elif contact.geom2 in self._object_geom_ids and body1 in self._body_taxels:
                hand_body, hand_is_geom2 = body1, False
            else:
                continue

            mj.mj_contactForce(self._model, data, con_idx, force6)
            # contact.frame rows = (normal, tan1, tan2); the normal points from
            # geom1 into geom2 and the constraint force pushes them apart, so
            # the force on geom2 is +frame^T f, on geom1 it is -frame^T f.
            frame = contact.frame.reshape(3, 3)
            force_world = frame.T @ force6[:3]
            if not hand_is_geom2:
                force_world = -force_world

            taxel_idx = self._body_taxels[hand_body]
            rel = contact.pos - site_pos[taxel_idx]
            # v_local = R^T v_world per taxel (site_xmat maps local -> world)
            local = np.einsum("kji,kj->ki", site_mat[taxel_idx], rel)
            dist_sq = local[:, 0] ** 2 + local[:, 1] ** 2
            weights = np.exp(-dist_sq / (2.0 * self._kernel_sigma**2))
            weights[dist_sq > self._kernel_cutoff**2] = 0.0
            # Keep the splat on the contacted patch: drop taxels whose plane is
            # far from the contact point (other patches on the same body).
            weights[np.abs(local[:, 2]) > self._kernel_cutoff] = 0.0
            weight_sum = weights.sum()
            if weight_sum <= 0.0:
                continue
            weights /= weight_sum

            force_local = np.einsum("kji,j->ki", site_mat[taxel_idx], force_world)
            out[taxel_idx] += weights[:, None] * force_local
            total_force += force_world

        self.last_total_force = total_force
        # Compression positive: pressing on the taxel pushes the hand along the
        # taxel -z axis, so flip the normal channel; shear stays the tangential
        # force applied by the object.
        out[:, 2] *= -1.0
        if self._taxel_gain is not None:
            out *= self._taxel_gain
        if self._noise_std > 0.0:
            out += self._rng.normal(0.0, self._noise_std, out.shape)
        return out.astype(np.float32)

    def taxel_positions(self, data: mj.MjData) -> np.ndarray:
        return data.site_xpos[self._site_ids].copy()

    def taxel_frames(self, data: mj.MjData) -> np.ndarray:
        return data.site_xmat[self._site_ids].reshape(-1, 3, 3).copy()

    def set_noise_model(self, noise_std: float, gain_std: float, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._noise_std = noise_std
        self._taxel_gain = 1.0 + self._rng.normal(0.0, gain_std, (N_TAXELS, 1))

    def close(self) -> None:
        pass
