"""
Canonical LeapXELA 368-taxel layout.

Flat taxel order == hardware taxel IDs 0..367 (leapXelaMap.LEAP_XELA_ID), so
simulation datasets are index-compatible with real XELA streams. Patch base
poses come from the leapXela_pointcloud calibration JSONs; within-patch grids
follow ros_ws/src/xela_description/mjcf/add_touch_point_cloud_to_model.py; the
hardware<->sim correspondence follows
ros_ws/src/xela_point_cloud_representation/src/leap_taxel_map.py.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
_MODEL_DIR = _PACKAGE_DIR.parent
POINTCLOUD_DIR = _PACKAGE_DIR / "leapXela_pointcloud"
ROBOT_XML = POINTCLOUD_DIR / "robot.xml"
TAXEL_MAP_JSON = _PACKAGE_DIR / "leap_sensor_taxel_map.json"
TAXEL_ID_MAP_PY = _PACKAGE_DIR / "leapXelaMap.py"
FINGERTIP_MAGNET_POSE_JSON = _MODEL_DIR / "fingertip_magnet_pose.json"

N_TAXELS = 368
PACK_SHAPE = (26, 31)

# Within-patch grid conventions (add_touch_point_cloud_to_model.py):
# 4x4 taxel (i, j) at ((i+1)*4.25mm, (j+1)*4.70mm, 5mm); 4x6 taxel (i, j) at
# (4.55mm + i*7.25mm, 4.43mm + j*7.25mm, 5mm); both frames flipped by
# euler [pi, 0, pi] (== rotation about y by pi) so taxel +z faces outward.
GRID_4_4_ORIGIN = (4.25e-3, 4.70e-3)
GRID_4_4_PITCH = (4.25e-3, 4.70e-3)
GRID_4_6_ORIGIN = (4.55e-3, 4.43e-3)
GRID_4_6_PITCH = (7.25e-3, 7.25e-3)
GRID_Z = 5.0e-3
GRID_FLIP_QUAT = (0.0, 0.0, 1.0, 0.0)

# Patch -> (finger key, patch key) in leap_sensor_taxel_map.json, following
# get_link_name_from_base_frame_name_* in add_touch_point_cloud_to_model.py
# ("second"=distal 4x4, "third"=middle/proximal 4x4, "fourth"=palm knuckle).
PATCH_TO_HARDWARE: Dict[str, Tuple[str, str]] = {
    "4_4_1": ("MF", "fourth"),
    "4_4_2": ("RF", "fourth"),
    "4_4_3": ("IF", "fourth"),
    "4_4_4": ("RF", "third"),
    "4_4_5": ("RF", "second"),
    "4_4_6": ("MF", "third"),
    "4_4_7": ("MF", "second"),
    "4_4_8": ("IF", "third"),
    "4_4_9": ("IF", "second"),
    "4_4_10": ("TH", "third"),
    "4_4_11": ("TH", "second"),
    "4_6_1": ("Palm", "down_left"),
    "4_6_2": ("Palm", "up_right"),
    "4_6_3": ("Palm", "up_left"),
    "tip_rf": ("RF", "tip"),
    "tip_mf": ("MF", "tip"),
    "tip_if": ("IF", "tip"),
    "tip_th": ("TH", "tip"),
}

FINGERTIP_PATCH_TO_FINGER = {"tip_rf": "rf", "tip_mf": "mf", "tip_if": "if", "tip_th": "th"}


@dataclass(frozen=True)
class TaxelEntry:
    taxel_id: int
    patch: str
    body: str
    site_name: str
    pos: np.ndarray
    quat: np.ndarray


@dataclass(frozen=True)
class TaxelLayout:
    entries: Tuple[TaxelEntry, ...]
    id_grid: np.ndarray
    pack_rows: np.ndarray
    pack_cols: np.ndarray


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_mat(q) @ np.asarray(v, dtype=np.float64)


def _joint_body_map(robot_xml: pathlib.Path) -> Dict[str, str]:
    worldbody = ET.parse(robot_xml).getroot().find("worldbody")
    mapping: Dict[str, str] = {}
    for body in worldbody.iter("body"):
        for joint in body.findall("joint"):
            mapping[joint.get("name")] = body.get("name")
    return mapping


def semantic_body_map(robot_xml: pathlib.Path) -> Dict[str, str]:
    """Map semantic link names used by the calibration JSONs to MJCF body names."""
    joints = _joint_body_map(robot_xml)
    palm = ET.parse(robot_xml).getroot().find("worldbody").find("body").get("name")
    mapping = {"palm": palm}
    for finger in ("rf", "mf", "if"):
        mapping[f"{finger}_px"] = joints[f"{finger}_rot"]
        mapping[f"{finger}_md"] = joints[f"{finger}_pip"]
        mapping[f"{finger}_ds"] = joints[f"{finger}_dip"]
    mapping["th_px"] = joints["th_mcp"]
    mapping["th_ds"] = joints["th_ipl"]
    return mapping


def _load_json(path: pathlib.Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def _load_id_grid(map_py: pathlib.Path) -> np.ndarray:
    spec = importlib.util.spec_from_file_location("leapXelaMap", map_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = module.LEAP_XELA_ID.astype(np.float64)
    return np.where(raw == 6e6, -1.0, raw).astype(np.int64)


def _grid_entries(
    patch: str,
    body: str,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    hw_rows: Dict[str, List[int]],
    origin: Tuple[float, float],
    pitch: Tuple[float, float],
) -> List[TaxelEntry]:
    quat = quat_mul(base_quat, np.asarray(GRID_FLIP_QUAT))
    entries = []
    for row_idx in range(len(hw_rows)):
        ids = hw_rows[str(row_idx + 1)]
        for m, taxel_id in enumerate(ids):
            i, j = row_idx, 3 - m
            local = np.array([origin[0] + i * pitch[0], origin[1] + j * pitch[1], GRID_Z])
            entries.append(
                TaxelEntry(
                    taxel_id=int(taxel_id),
                    patch=patch,
                    body=body,
                    site_name=f"taxel_{int(taxel_id):03d}",
                    pos=base_pos + quat_rotate(base_quat, local),
                    quat=quat,
                )
            )
    return entries


def _fingertip_entries(
    patch: str,
    body: str,
    magnet_pose: Dict,
    finger: str,
    hw_tip: Dict[str, List[int]],
) -> List[TaxelEntry]:
    base = magnet_pose[f"{finger}_pointcloud_base_frame"]
    base_pos = np.asarray(base["pos"], dtype=np.float64)
    base_quat = np.asarray(base["quat"], dtype=np.float64)
    tip_ids: List[int] = []
    for row in range(1, 7):
        tip_ids.extend(hw_tip[str(row)])
    entries = []
    for k, taxel_id in enumerate(tip_ids):
        rel = magnet_pose[str(k + 1)]
        entries.append(
            TaxelEntry(
                taxel_id=int(taxel_id),
                patch=patch,
                body=body,
                site_name=f"taxel_{int(taxel_id):03d}",
                pos=base_pos + quat_rotate(base_quat, np.asarray(rel["pos"], dtype=np.float64)),
                quat=quat_mul(base_quat, np.asarray(rel["quat"], dtype=np.float64)),
            )
        )
    return entries


def build_layout() -> TaxelLayout:
    bodies = semantic_body_map(ROBOT_XML)
    sites_4_4 = _load_json(POINTCLOUD_DIR / "4_4_sites.json")
    sites_4_6 = _load_json(POINTCLOUD_DIR / "4_6_sites.json")
    magnet_pose = _load_json(FINGERTIP_MAGNET_POSE_JSON)
    hw_map = _load_json(TAXEL_MAP_JSON)
    id_grid = _load_id_grid(TAXEL_ID_MAP_PY)

    entries: List[TaxelEntry] = []
    for patch, (finger_key, patch_key) in PATCH_TO_HARDWARE.items():
        hw_patch = hw_map[finger_key][patch_key]
        if patch.startswith("4_4") or patch.startswith("4_6"):
            sites = sites_4_4 if patch.startswith("4_4") else sites_4_6
            base = sites[f"{patch}_base_frame"]
            origin = GRID_4_4_ORIGIN if patch.startswith("4_4") else GRID_4_6_ORIGIN
            pitch = GRID_4_4_PITCH if patch.startswith("4_4") else GRID_4_6_PITCH
            entries.extend(
                _grid_entries(
                    patch=patch,
                    body=bodies[base["parent"]],
                    base_pos=np.asarray(base["pos"], dtype=np.float64),
                    base_quat=np.asarray(base["quat"], dtype=np.float64),
                    hw_rows=hw_patch,
                    origin=origin,
                    pitch=pitch,
                )
            )
        else:
            finger = FINGERTIP_PATCH_TO_FINGER[patch]
            entries.extend(
                _fingertip_entries(
                    patch=patch,
                    body=bodies[f"{finger}_ds"],
                    magnet_pose=magnet_pose,
                    finger=finger,
                    hw_tip=hw_patch,
                )
            )

    entries.sort(key=lambda e: e.taxel_id)
    ids = np.array([e.taxel_id for e in entries])
    if not np.array_equal(ids, np.arange(N_TAXELS)):
        raise ValueError("Taxel IDs do not cover 0..367 exactly once")

    pack_rows = np.zeros(N_TAXELS, dtype=np.int64)
    pack_cols = np.zeros(N_TAXELS, dtype=np.int64)
    rows, cols = np.where(id_grid >= 0)
    grid_ids = id_grid[rows, cols]
    if not np.array_equal(np.sort(grid_ids), np.arange(N_TAXELS)):
        raise ValueError("LEAP_XELA_ID grid does not contain IDs 0..367 exactly once")
    pack_rows[grid_ids] = rows
    pack_cols[grid_ids] = cols

    return TaxelLayout(
        entries=tuple(entries),
        id_grid=id_grid,
        pack_rows=pack_rows,
        pack_cols=pack_cols,
    )


def pack_frames(flat: np.ndarray, layout: TaxelLayout) -> np.ndarray:
    """(T, 368, C) taxel values -> (T, C, 26, 31) hardware image packing."""
    if flat.ndim != 3 or flat.shape[1] != N_TAXELS:
        raise ValueError(f"Expected (T, {N_TAXELS}, C), got {flat.shape}")
    packed = np.zeros((flat.shape[0], flat.shape[2], *PACK_SHAPE), dtype=np.float32)
    packed[:, :, layout.pack_rows, layout.pack_cols] = flat.transpose(0, 2, 1)
    return packed


def patch_indices(layout: TaxelLayout) -> Dict[str, np.ndarray]:
    """Flat taxel indices per patch name."""
    indices: Dict[str, List[int]] = {}
    for flat_idx, entry in enumerate(layout.entries):
        indices.setdefault(entry.patch, []).append(flat_idx)
    return {patch: np.asarray(idx, dtype=np.int64) for patch, idx in indices.items()}
