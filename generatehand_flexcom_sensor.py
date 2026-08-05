import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np


_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from leapxela.taxel_layout import FINGERTIP_MAGNET_POSE_JSON, build_layout

TAXEL_COUNT = 368
# Calibrated taxel positions must be reproduced in-plane to this tolerance.
TAXEL_POSITION_TOLERANCE = 0.0001
PAD_BOX_MIN_HALF_SIZE = 0.002
# Canonical patch name -> the collision geom the membrane is mounted on.
PATCH_TO_GEOM = {
    "4_4_1": "mf_bs_uspa44",
    "4_4_2": "rf_bs_uspa44",
    "4_4_3": "if_bs_uspa44",
    "4_4_4": "rf_px_uspa44",
    "4_4_5": "rf_md_uspa44",
    "4_4_6": "mf_px_uspa44",
    "4_4_7": "mf_md_uspa44",
    "4_4_8": "if_px_uspa44",
    "4_4_9": "if_md_uspa44",
    "4_4_10": "th_px_uspa44",
    "4_4_11": "th_ds_uspa44",
    "4_6_1": "uspa46_3",
    "4_6_2": "uspa46_2",
    "4_6_3": "uspa46_1",
}
GEOM_TO_PATCH = {geom: patch for patch, geom in PATCH_TO_GEOM.items()}


FINGER_TIP_TYPES = ("CoACD", "Box")
FLEX_EDGE_INSET = 0.001
FLEX_CLEARANCE = 0.000
FLEX_RADIUS = 0.0006
FLEX_THICKNESS = 0.001
FLEX_YOUNG = 2.0e4
PROBE_GEOM_NAME = "sensor_probe"
PROBE_BODY_NAME = "sensor_probe_body"
PROBE_RADIUS = 0.004
PROBE_CLEARANCE = 0.002
PROBE_PENETRATION = 0.0004
REST_SETTLE_DURATION = 0.3
REST_FLATNESS_TOLERANCE = 0.0005
TIP_SURFACE_CLEARANCE = 0.0015
FLEX_ANCHOR_SOLREF = (0.01, 1.0)
FLEX_ANCHOR_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
FLEX_VERTEX_DAMPING = 0.05
FLEX_CONTACT_SOLREF = (0.01, 1.0)
TAXEL_SITE_SIZE = 0.0012
# The vertex bodies carry the taxel frame, which puts the slide DOFs on the
# membrane's principal stiffness directions. Newton's cost-improvement test
# then declares victory after a single iteration and leaves the pads badly
# under-converged: an untouched palm patch picks up a 1.5 mm oscillation during
# a curl. tolerance=0 disables that early exit (tightening it does not help --
# 1e-12 still exits after one iteration). 20 iterations is not enough (13 mm of
# lag); 30 passes, 50 is the same speed to within measurement noise, so take
# the margin. Costs ~12% throughput: 852 -> 749 steps/s.
SOLVER_ITERATIONS = 50
SOLVER_TOLERANCE = 0.0
CURL_FRACTION = 0.8
CURL_RAMP_DURATION = 0.4
CURL_HOLD_DURATION = 0.3
CURL_RETURN_DURATION = 0.4
CURL_MAX_STRETCH_RATIO = 3.0
CURL_PEAK_HOME_ERROR = 0.006
CURL_RESIDUAL_HOME_ERROR = 0.0005
TIP_FINGERS = ("if", "mf", "rf", "th")
TIP_TAXEL_COUNT = 30
FINGERTIP_POSE_JSON = FINGERTIP_MAGNET_POSE_JSON
# Physical tip layout: the JSON keys "1".."30" run column-major (six columns
# down the finger, sizes 5/4/6/6/4/5), while the 6x6 hardware canvas of
# leapxela/taxel_tip_pos.md is row-major. This table is the bridge between the
# two and the single source of truth for mesh and heatmap indexing.
TIP_CANVAS_ROWS = (
    (None, None, 10, 16, None, None),
    (1, None, 11, 17, None, 26),
    (2, 6, 12, 18, 22, 27),
    (3, 7, 13, 19, 23, 28),
    (4, 8, 14, 20, 24, 29),
    (5, 9, 15, 21, 25, 30),
)
# Canvas cell of each taxel, indexed by 0-based JSON key.
TIP_GRID_ROWCOL = tuple(
    next(
        (row, column)
        for row, cells in enumerate(TIP_CANVAS_ROWS)
        for column, key in enumerate(cells)
        if key == taxel + 1
    )
    for taxel in range(TIP_TAXEL_COUNT)
)
# Hardware stream index -> 0-based JSON key (canvas read row-major).
TIP_DOC_TO_KEY = tuple(
    key - 1 for cells in TIP_CANVAS_ROWS for key in cells if key is not None
)
TIP_HEATMAP_PITCH = 0.0065
# Correct mesh spans 6-15 mm after the outward offset; a canvas/calibration
# mismatch jumps straight to ~34 mm (edges crossing the whole fingertip).
TIP_MAX_EDGE_LENGTH = 0.020
# Pad-face taxels used as the probe-press target (canvas rows 3-4, cols 2-3:
# the most protruding taxels of the dome).
TIP_PRESS_TAXELS = (12, 13, 18, 19)


def _tip_elements() -> list[int]:
    """42 triangles over the 6x6 canvas, remapped to JSON key order."""
    bridge = [
        (1, 3, 4), (1, 2, 4), (2, 4, 5), (2, 5, 6),
        (3, 7, 8), (3, 8, 4), (4, 8, 9), (4, 9, 5),
        (5, 9, 10), (5, 10, 11), (5, 11, 6), (6, 11, 12),
    ]
    strip = []
    for row_pair in range(3):
        base = 7 + row_pair * 6
        for column in range(5):
            near, far = base + column, base + column + 1
            below_near = base + 6 + column
            below_far = base + 6 + column + 1
            strip.append((near, below_near, below_far))
            strip.append((near, below_far, far))
    return [
        TIP_DOC_TO_KEY[index - 1]
        for triangle in bridge + strip
        for index in triangle
    ]


TIP_ELEMENTS = _tip_elements()


@dataclass(frozen=True)
class SensorDefinition:
    geom_name: str
    count: tuple[int, int, int]
    normal_axis: int
    normal_sign: float
    # Roll the calibrated patch frame 180 deg about its normal. The Palm
    # "up_right" module is calibrated with base quat [0.7071 -0.7071 0 0] where
    # the other two palm 4x6 modules have [0 0 -0.7071 -0.7071] (the hardware
    # workspace's leapXela_pointcloud/4_6_sites.json), so its in-plane axes
    # point the opposite way. Undoing it makes every palm patch present the
    # same axes, at the cost of negating shear_x/shear_y for those taxels
    # relative to the real XELA stream.
    flip_in_plane: bool = False


@dataclass(frozen=True)
class FlexSkin:
    definition: SensorDefinition
    flex_name: str
    parent_body_name: str
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    normal: np.ndarray
    half_u: float
    half_v: float
    taxel_ids: np.ndarray
    vertex_rotations: np.ndarray


@dataclass(frozen=True)
class TipSkin:
    finger: str
    flex_name: str
    parent_body_name: str
    vertex_body_names: tuple[str, ...]
    vertex_rotations: np.ndarray
    grid_rowcol: tuple[tuple[int, int], ...]
    taxel_ids: np.ndarray


SENSOR_DEFINITIONS = (
    SensorDefinition("uspa46_1", (6, 4, 1), 1, -1.0),
    SensorDefinition("uspa46_2", (6, 4, 1), 1, -1.0, flip_in_plane=True),
    SensorDefinition("uspa46_3", (6, 4, 1), 1, -1.0),
    SensorDefinition("rf_bs_uspa44", (4, 4, 1), 1, -1.0),
    SensorDefinition("rf_px_uspa44", (4, 4, 1), 2, 1.0),
    SensorDefinition("rf_md_uspa44", (4, 4, 1), 1, 1.0),
    SensorDefinition("mf_bs_uspa44", (4, 4, 1), 1, -1.0),
    SensorDefinition("mf_px_uspa44", (4, 4, 1), 2, 1.0),
    SensorDefinition("mf_md_uspa44", (4, 4, 1), 1, 1.0),
    SensorDefinition("if_bs_uspa44", (4, 4, 1), 1, -1.0),
    SensorDefinition("if_px_uspa44", (4, 4, 1), 2, 1.0),
    SensorDefinition("if_md_uspa44", (4, 4, 1), 1, 1.0),
    SensorDefinition("th_px_uspa44", (4, 4, 1), 0, -1.0),
    SensorDefinition("th_ds_uspa44", (4, 4, 1), 0, 1.0),
)


def load_base_model(mode: str) -> mj.MjSpec:
    paths = {
        "base_model": _MODEL_DIR / "leapXela_base_model.xml",
        "touchgrid": _MODEL_DIR
        / "robot_touch_sensor_array_binary_touchgrid_generated.xml",
    }
    if mode in FINGER_TIP_TYPES:
        model_path = paths["base_model"]
    elif mode == "touchgrid":
        model_path = paths["touchgrid"]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    print(f"Loaded base model from {model_path}")
    return mj.MjSpec.from_file(model_path.as_posix())


def _is_perimeter_vertex(
    vertex_id: int, count: tuple[int, int, int]
) -> bool:
    column = vertex_id // count[1]
    row = vertex_id % count[1]
    return (
        column == 0
        or column == count[0] - 1
        or row == 0
        or row == count[1] - 1
    )


def _surface_basis(
    geom: mj.MjsGeom, definition: SensorDefinition
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_local = np.zeros(3)
    normal_local[definition.normal_axis] = definition.normal_sign
    tangent_u_local = np.zeros(3)
    tangent_u_local[1 if definition.normal_axis == 0 else 0] = 1.0
    tangent_v_local = np.cross(normal_local, tangent_u_local)

    geom_rotation_flat = np.zeros(9)
    mj.mju_quat2Mat(geom_rotation_flat, np.asarray(geom.quat))
    geom_rotation = geom_rotation_flat.reshape(3, 3)
    return (
        geom_rotation @ tangent_u_local,
        geom_rotation @ tangent_v_local,
        geom_rotation @ normal_local,
    )


def _basis_quaternion(
    tangent_u: np.ndarray, tangent_v: np.ndarray, normal: np.ndarray
) -> list[float]:
    rotation = np.column_stack((tangent_u, tangent_v, normal))
    quaternion = np.zeros(4)
    mj.mju_mat2Quat(quaternion, rotation.ravel())
    return quaternion.tolist()


def add_taxel_site(body: mj.MjsBody, taxel_id: int) -> None:
    """Name a flex vertex body with its hardware taxel id.

    The site carries no rotation of its own: it inherits the vertex body frame,
    which is the taxel frame (+z out of the skin). It exists to give a consumer
    holding only the XML a named handle on the taxel — `data.site_xpos` and
    `data.site_xmat` instead of hunting through `flex_vertbodyid`. The name
    matches `leapxela.taxel_layout.TaxelEntry.site_name`.
    """
    body.add_site(
        name=f"taxel_{taxel_id:03d}",
        size=[TAXEL_SITE_SIZE, 0.0, 0.0],
        group=4,
        rgba=[1.0, 0.0, 0.0, 1.0],
    )


def pad_grid_from_entries(
    entries: list, definition: SensorDefinition
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve a calibrated patch into a regular grid the flex can reproduce.

    Returns (rotation, centre, pitch, taxel_ids, positions) with taxel_ids
    ordered by MuJoCo's grid vertex index (row-major, v fastest).
    """
    positions = np.array([entry.pos for entry in entries], dtype=np.float64)
    rotation = _quat_to_matrix(np.asarray(entries[0].quat, dtype=np.float64))
    if definition.flip_in_plane:
        # 180 deg about the normal: negate the two in-plane columns, which
        # keeps det = +1. It has to happen here rather than downstream, before
        # the grid indices below are derived from these axes -- the vertex
        # bodies are laid out by the same rotation, so a later flip would
        # desync `taxel_ids` from the vertices it labels.
        rotation = rotation @ np.diag([-1.0, -1.0, 1.0])
    centre = positions.mean(axis=0)
    offsets = positions - centre
    height_spread = float(np.ptp(offsets @ rotation[:, 2]))
    if height_spread > TAXEL_POSITION_TOLERANCE:
        raise ValueError(
            f"Patch for '{definition.geom_name}' is not planar "
            f"({height_spread * 1000.0:.3f} mm spread); a grid flex cannot "
            f"reproduce it"
        )

    count = definition.count
    taxel_ids = np.full(count[0] * count[1], -1, dtype=np.int64)
    pitch = np.zeros(2)
    for axis, (size, column) in enumerate(((count[0], 0), (count[1], 1))):
        coordinate = offsets @ rotation[:, column]
        levels = np.unique(np.round(coordinate, 6))
        if len(levels) != size:
            raise ValueError(
                f"Patch for '{definition.geom_name}' has {len(levels)} "
                f"columns along axis {axis}, expected {size}"
            )
        pitch[axis] = float(np.diff(levels).mean())
    for index, entry in enumerate(entries):
        i = int(round((offsets[index] @ rotation[:, 0]) / pitch[0]
                      + (count[0] - 1) / 2.0))
        j = int(round((offsets[index] @ rotation[:, 1]) / pitch[1]
                      + (count[1] - 1) / 2.0))
        taxel_ids[i * count[1] + j] = entry.taxel_id
    if np.any(taxel_ids < 0):
        raise ValueError(
            f"Patch for '{definition.geom_name}' did not fill every grid cell"
        )
    return rotation, centre, pitch, taxel_ids, positions


def add_flex_sensor(
    spec: mj.MjSpec, definition: SensorDefinition, entries: list
) -> FlexSkin:
    geom = spec.geom(definition.geom_name)
    if geom is None:
        raise ValueError(f"Sensor geom '{definition.geom_name}' does not exist")

    parent = geom.parent
    _, _, mount_normal = _surface_basis(geom, definition)
    geom_half_sizes = np.asarray(geom.size[:3], dtype=np.float64)
    half_normal = float(np.dot(np.abs(mount_normal), geom_half_sizes))

    rotation, centre, pitch, taxel_ids, calibrated = pad_grid_from_entries(
        entries, definition
    )
    tangent_u, tangent_v, normal = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    count = definition.count
    half_u = 0.5 * (count[0] - 1) * float(pitch[0])
    half_v = 0.5 * (count[1] - 1) * float(pitch[1])

    # Keep the membrane on the existing mounting plane (the collision box face)
    # and take only the in-plane pattern from the calibration.
    mount_plane = (
        np.asarray(geom.pos, dtype=np.float64)
        + mount_normal * (half_normal + FLEX_RADIUS + FLEX_CLEARANCE)
    )
    flex_position = centre + normal * float((mount_plane - centre) @ normal)

    spacing = [float(pitch[0]), float(pitch[1]), 2.1 * FLEX_RADIUS]
    flex_name = f"flex_{definition.geom_name}"
    patch_quat = _basis_quaternion(tangent_u, tangent_v, normal)
    flex = parent.make_flex(
        name=flex_name,
        type="grid",
        dim=2,
        count=list(count),
        spacing=spacing,
        radius=FLEX_RADIUS,
        pos=flex_position.tolist(),
        quat=patch_quat,
        mass=0.0005 * count[0] * count[1],
        equality=1,
        elastic2d=2,
    )
    flex.young = FLEX_YOUNG
    flex.poisson = 0.3
    flex.damping = 0.1
    flex.thickness = FLEX_THICKNESS
    # Bit 2 only: skins are touched by sensed objects (probe/objects carry
    # bit 2), never by hand geoms (bit 1) — hand contact snags and folds the
    # membrane during joint motion.
    flex.contype = 2
    flex.conaffinity = 2
    flex.condim = 3
    flex.friction = [0.8, 0.005, 0.0001]
    flex.rgba = [1.0, 0.2, 0.1, 0.8]
    flex.selfcollide = mj.mjtFlexSelf.mjFLEXSELF_NONE
    # Soft contacts that win the param mixing against the hand's rigid class
    # (solref 0.0001 would otherwise explode the light vertex bodies).
    flex.priority = 1
    flex.solref = list(FLEX_CONTACT_SOLREF)

    # Every vertex stays free in x,y,z; a connect equality anchors it to its
    # home position on the link (solver-stable, unlike explicit joint springs).
    # MuJoCo leaves vertex bodies aligned with the parent link, so the taxel
    # frame has to be put on them explicitly. This re-bases the slide DOFs onto
    # (shear_u, shear_v, normal) — equivalent dynamics (isotropic mass and
    # damping, anchor at the body origin), but it needs the raised solver
    # budget in `main()`: at the default one the pads are left under-converged
    # and a palm patch picks up a 1.5 mm oscillation during a curl.
    # `flex.vertbody` follows the grid vertex order that `taxel_ids` is built in.
    for index, body_name in enumerate(flex.vertbody):
        body = spec.body(body_name)
        body.quat = patch_quat
        add_taxel_site(body, int(taxel_ids[index]))
        for joint in body.joints:
            joint.damping = [FLEX_VERTEX_DAMPING, 0.0, 0.0]
        anchor = spec.add_equality()
        anchor.type = mj.mjtEq.mjEQ_CONNECT
        anchor.objtype = mj.mjtObj.mjOBJ_BODY
        anchor.name1 = body_name
        anchor.name2 = parent.name
        anchor.data[:3] = [0.0, 0.0, 0.0]
        anchor.solref = list(FLEX_ANCHOR_SOLREF)
        anchor.solimp = list(FLEX_ANCHOR_SOLIMP)

    # The generated grid must land on the calibrated taxels (in-plane).
    generated = (
        flex_position
        + np.stack([
            (i - (count[0] - 1) / 2.0) * pitch[0] * tangent_u
            + (j - (count[1] - 1) / 2.0) * pitch[1] * tangent_v
            for i in range(count[0])
            for j in range(count[1])
        ])
    )
    order = np.argsort(taxel_ids)
    reference = calibrated[np.argsort([e.taxel_id for e in entries])]
    planar_error = generated[order] - reference
    planar_error -= np.outer(planar_error @ normal, normal)
    worst = float(np.max(np.linalg.norm(planar_error, axis=1)))
    if worst > TAXEL_POSITION_TOLERANCE:
        raise ValueError(
            f"{flex_name} vertices miss the calibrated taxels by "
            f"{worst * 1000.0:.3f} mm in-plane"
        )

    print(
        f"Added {count[0]}x{count[1]}x{count[2]} {flex_name} over "
        f"{definition.geom_name}; pitch "
        f"{pitch[0] * 1000:.2f}x{pitch[1] * 1000:.2f} mm, taxel ids "
        f"{taxel_ids.min()}..{taxel_ids.max()}, anchored={len(flex.vertbody)}."
    )
    return FlexSkin(
        definition,
        flex_name,
        parent.name,
        tangent_u,
        tangent_v,
        normal,
        half_u,
        half_v,
        taxel_ids,
        np.tile(rotation, (count[0] * count[1], 1, 1)),
    )


def trim_pad_boxes(spec: mj.MjSpec) -> None:
    """Pull each pad box's outward face down onto the link's CAD surface.

    The `uspa` boxes are hand-placed proxies for the sensor modules, reused
    from one class default across links of differing geometry, so their outward
    face overhangs the real surface by up to 1.7 mm. The membrane is mounted on
    that face, which is what makes it float above the finger.
    """
    model = mj.MjModel.from_xml_path(
        (_MODEL_DIR / "leapXela_base_model.xml").as_posix()
    )
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    for definition in SENSOR_DEFINITIONS:
        geom = spec.geom(definition.geom_name)
        geom_id = mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_GEOM, definition.geom_name
        )
        body_id = int(model.geom_bodyid[geom_id])
        _, _, normal = _surface_basis(geom, definition)
        half_sizes = np.asarray(geom.size[:3], dtype=np.float64)
        half_normal = float(np.dot(np.abs(normal), half_sizes))
        rotation = data.xmat[body_id].reshape(3, 3)
        normal_world = rotation @ normal
        face = data.geom_xpos[geom_id] + normal_world * half_normal
        overhangs = []
        for candidate in range(model.ngeom):
            if int(model.geom_bodyid[candidate]) != body_id:
                continue
            if int(model.geom_group[candidate]) != 2:
                continue
            if model.geom_type[candidate] != mj.mjtGeom.mjGEOM_MESH:
                continue
            hit = mj.mj_rayMesh(model, data, candidate, face, -normal_world)
            if hit >= 0.0:
                overhangs.append(float(hit))
        if not overhangs:
            raise ValueError(
                f"No visual mesh under '{definition.geom_name}' to trim against"
            )
        overhang = min(overhangs)
        if overhang <= 0.0:
            continue
        axis = int(np.argmax(np.abs(normal)))
        shrink = 0.5 * overhang
        if half_sizes[axis] - shrink <= PAD_BOX_MIN_HALF_SIZE:
            raise ValueError(
                f"Trimming '{definition.geom_name}' by {overhang * 1000:.2f} mm "
                f"would leave a degenerate collision box"
            )
        size = list(np.asarray(geom.size, dtype=np.float64))
        size[axis] -= shrink
        geom.size = size
        # Keep the inner extent fixed: only the outward face moves.
        geom.pos = (
            np.asarray(geom.pos, dtype=np.float64)
            - normal * shrink
        ).tolist()
        print(
            f"Trimmed {definition.geom_name} outward face by "
            f"{overhang * 1000:.2f} mm onto the link surface."
        )


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    mj.mju_rotVecQuat(out, np.asarray(vector, dtype=np.float64),
                      np.asarray(quat, dtype=np.float64))
    return out


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    flat = np.zeros(9)
    mj.mju_quat2Mat(flat, np.asarray(quat, dtype=np.float64))
    return flat.reshape(3, 3)


def tip_surface_offsets(magnet_pose: dict) -> dict[str, np.ndarray]:
    """Outward offset (m) from each calibrated magnet pose to the skin surface.

    The JSON poses are magnet centers embedded under the fingertip shell; the
    sensing membrane must sit on the outer surface, so each taxel is pushed
    along its own +z until it clears the outermost shell mesh. The extra
    clearance is the silicone-skin standoff: without it the flat triangles
    chord across the convex dome and their interiors sink back into the shell.
    """
    model = mj.MjModel.from_xml_path(
        (_MODEL_DIR / "leapXela_base_model.xml").as_posix()
    )
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    offsets = {}
    for finger in TIP_FINGERS:
        base = magnet_pose[f"{finger}_pointcloud_base_frame"]
        base_pos = np.asarray(base["pos"], dtype=np.float64)
        base_quat = np.asarray(base["quat"], dtype=np.float64)
        base_rotation = _quat_to_matrix(base_quat)
        link_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{finger}_ds")
        link_rotation = data.xmat[link_id].reshape(3, 3)
        link_position = data.xpos[link_id]
        shell_ids = []
        for suffix in ("", "_2", "_3", "_4", "_5", "_6"):
            geom_id = mj.mj_name2id(
                model, mj.mjtObj.mjOBJ_GEOM, f"{finger}_ds_tip{suffix}"
            )
            if geom_id >= 0:
                shell_ids.append(geom_id)
        finger_offsets = np.zeros(TIP_TAXEL_COUNT)
        for k in range(1, TIP_TAXEL_COUNT + 1):
            relative = magnet_pose[str(k)]
            local = base_pos + _quat_rotate(base_quat, relative["pos"])
            direction_local = (
                base_rotation @ _quat_to_matrix(relative["quat"])
            )[:, 2]
            point = link_position + link_rotation @ local
            direction = link_rotation @ direction_local
            distance = 0.0
            for geom_id in shell_ids:
                hit = mj.mj_rayMesh(model, data, geom_id, point, direction)
                if hit > distance:
                    distance = float(hit)
            finger_offsets[k - 1] = (
                distance + FLEX_RADIUS + TIP_SURFACE_CLEARANCE
            )
        offsets[finger] = finger_offsets
    return offsets


TIP_MIN_FACING = 0.1


def _orient_tip_elements(
    elements: np.ndarray,
    positions: np.ndarray,
    vertex_normals: np.ndarray,
    finger: str,
) -> tuple[list[int], int]:
    """Wind every tip triangle so its normal faces away from the finger.

    MuJoCo shades flex elements from their winding, so a triangle wound
    against its neighbours renders inverted (a dark facet with a lit edge).
    """
    oriented = []
    reversed_count = 0
    for triangle in elements:
        corners = positions[triangle]
        face_normal = np.cross(
            corners[1] - corners[0], corners[2] - corners[0]
        )
        norm = float(np.linalg.norm(face_normal))
        outward = np.mean(vertex_normals[triangle], axis=0)
        outward = outward / np.linalg.norm(outward)
        facing = float(face_normal @ outward) / norm if norm > 0.0 else 0.0
        if abs(facing) < TIP_MIN_FACING:
            raise ValueError(
                f"{finger} tip triangle {triangle.tolist()} is degenerate or "
                f"tangential (facing {facing:+.3f}); its winding is ambiguous"
            )
        if facing < 0.0:
            triangle = [triangle[0], triangle[2], triangle[1]]
            reversed_count += 1
        oriented.extend(int(index) for index in triangle)
    return oriented, reversed_count


def add_tip_flex(
    spec: mj.MjSpec,
    finger: str,
    entries: list,
    surface_offsets: np.ndarray,
) -> TipSkin:
    """Curved fingertip skin: 30 vertex bodies at calibrated poses + mesh flex.

    The built-in continuum elasticity (elastic2d) destabilizes this irregular
    curved mesh, so membrane structure comes from the flex edge-equality
    constraint plus the per-vertex anchors instead.
    """
    parent = spec.body(f"{finger}_ds")
    if parent is None:
        raise ValueError(f"Body '{finger}_ds' does not exist")
    if len(entries) != TIP_TAXEL_COUNT:
        raise ValueError(
            f"{finger} tip has {len(entries)} calibrated taxels, "
            f"expected {TIP_TAXEL_COUNT}"
        )

    body_names = []
    positions = np.zeros((TIP_TAXEL_COUNT, 3))
    rotations = np.zeros((TIP_TAXEL_COUNT, 3, 3))
    taxel_ids = np.array([entry.taxel_id for entry in entries], dtype=np.int64)
    for k, entry in enumerate(entries, start=1):
        rotations[k - 1] = _quat_to_matrix(
            np.asarray(entry.quat, dtype=np.float64)
        )
        position = (
            np.asarray(entry.pos, dtype=np.float64)
            + surface_offsets[k - 1] * rotations[k - 1][:, 2]
        )
        positions[k - 1] = position
        name = f"flex_{finger}_tip_{k}"
        # Each vertex of a curved tip has its own normal, so the taxel frame is
        # per vertex here: the shared `{finger}_ds` link frame cannot be the
        # surface frame for more than one of the 30 taxels.
        body = parent.add_body(
            name=name,
            pos=position.tolist(),
            quat=np.asarray(entry.quat, dtype=np.float64).tolist(),
        )
        body.explicitinertial = True
        body.mass = 0.0005
        body.inertia = [1.0e-9, 1.0e-9, 1.0e-9]
        # A vertex is a point mass, so its centre of mass is the body origin.
        # Left unset this defaults to the body position, putting the centre of
        # mass ~64 mm away; inert while the frame matched the link, but with
        # the taxel rotation above that offset swings with the frame and
        # perturbs the finger's composite inertia.
        body.ipos = [0.0, 0.0, 0.0]
        add_taxel_site(body, int(entry.taxel_id))
        for axis_index, axis in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
            joint = body.add_joint(
                name=f"{name}_j{axis_index}",
                type=mj.mjtJoint.mjJNT_SLIDE,
                axis=axis,
            )
            joint.damping = [FLEX_VERTEX_DAMPING, 0.0, 0.0]
        anchor = spec.add_equality()
        anchor.type = mj.mjtEq.mjEQ_CONNECT
        anchor.objtype = mj.mjtObj.mjOBJ_BODY
        anchor.name1 = name
        anchor.name2 = parent.name
        anchor.data[:3] = [0.0, 0.0, 0.0]
        anchor.solref = list(FLEX_ANCHOR_SOLREF)
        anchor.solimp = list(FLEX_ANCHOR_SOLIMP)
        body_names.append(name)

    # A mesh edge far longer than the ~6.5 mm taxel pitch means the canvas
    # mapping no longer matches the calibration and triangles would slice
    # through the fingertip.
    elements = np.asarray(TIP_ELEMENTS).reshape(-1, 3)
    edges = np.concatenate([
        positions[elements[:, i]] - positions[elements[:, (i + 1) % 3]]
        for i in range(3)
    ])
    longest = float(np.max(np.linalg.norm(edges, axis=1)))
    if longest > TIP_MAX_EDGE_LENGTH:
        raise ValueError(
            f"{finger} tip mesh has a {longest * 1000.0:.1f} mm edge "
            f"(limit {TIP_MAX_EDGE_LENGTH * 1000.0:.1f} mm): the taxel canvas "
            f"does not match the calibrated layout"
        )

    oriented, reversed_count = _orient_tip_elements(
        elements, positions, rotations[:, :, 2], finger
    )

    flex = spec.add_flex()
    flex.name = f"flex_{finger}_tip"
    flex.dim = 2
    flex.vertbody = body_names
    flex.elem = oriented
    flex.radius = FLEX_RADIUS
    flex.thickness = FLEX_THICKNESS
    flex.contype = 2
    flex.conaffinity = 2
    flex.condim = 3
    flex.friction = [0.8, 0.005, 0.0001]
    flex.priority = 1
    flex.solref = list(FLEX_CONTACT_SOLREF)
    flex.rgba = [1.0, 0.2, 0.1, 0.9]
    flex.selfcollide = mj.mjtFlexSelf.mjFLEXSELF_NONE
    flex.internal = False
    edge_equality = spec.add_equality()
    edge_equality.type = mj.mjtEq.mjEQ_FLEX
    edge_equality.objtype = mj.mjtObj.mjOBJ_FLEX
    edge_equality.name1 = flex.name

    print(
        f"Added curved {flex.name}: {TIP_TAXEL_COUNT} taxels, "
        f"{len(TIP_ELEMENTS) // 3} triangles ({reversed_count} rewound "
        f"outward), taxel ids {taxel_ids.min()}..{taxel_ids.max()}, "
        f"anchored={len(body_names)}."
    )
    return TipSkin(
        finger,
        flex.name,
        parent.name,
        tuple(body_names),
        rotations,
        TIP_GRID_ROWCOL,
        taxel_ids,
    )


def pad_vertex_rotations(skin: FlexSkin) -> np.ndarray:
    return skin.vertex_rotations


def tip_canvas(tip: TipSkin, flat: np.ndarray) -> np.ndarray:
    canvas = np.full((6, 6, 3), np.nan)
    for index, (row, column) in enumerate(tip.grid_rowcol):
        canvas[row, column] = flat[index]
    return canvas


class FlexTaxelSensor:
    """Reads per-vertex taxel forces from external contacts on a flex skin.

    Every flex vertex is one taxel with its own frame (`vertex_rotations`,
    (n, 3, 3), columns = shear_x/shear_y/normal axes in the parent link
    frame). Contact forces from the listed geoms are splatted onto the
    vertices of the contacted flex element (barycentric, force-conserving)
    and expressed per taxel as (shear_x, shear_y, normal_z), compression
    positive.
    """

    def __init__(
        self,
        model: mj.MjModel,
        flex_name: str,
        parent_body_name: str,
        vertex_rotations: np.ndarray,
        contact_geom_names: list[str],
    ):
        self.model = model
        self.flex_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_FLEX, flex_name)
        if self.flex_id < 0:
            raise ValueError(f"Flex '{flex_name}' not found")
        self.link_id = mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_BODY, parent_body_name
        )
        if self.link_id < 0:
            raise ValueError(f"Body '{parent_body_name}' not found")
        self.contact_geom_ids = set()
        for name in contact_geom_names:
            geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"Contact geom '{name}' not found")
            self.contact_geom_ids.add(geom_id)
        self.vert_adr = int(model.flex_vertadr[self.flex_id])
        self.vert_num = int(model.flex_vertnum[self.flex_id])
        self.vertex_rotations = np.asarray(vertex_rotations, dtype=np.float64)
        if self.vertex_rotations.shape != (self.vert_num, 3, 3):
            raise ValueError(
                f"vertex_rotations must be ({self.vert_num}, 3, 3), got "
                f"{self.vertex_rotations.shape}"
            )
        element_count = int(model.flex_elemnum[self.flex_id])
        data_address = int(model.flex_elemdataadr[self.flex_id])
        self.elements = model.flex_elem[
            data_address : data_address + 3 * element_count
        ].reshape(element_count, 3)

    def read(self, data: mj.MjData) -> tuple[np.ndarray, int]:
        """Return ((n_taxels, 3) readings, contact count)."""
        vertex_forces = np.zeros((self.vert_num, 3))
        contact_count = 0
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            flex_sides = (int(contact.flex[0]), int(contact.flex[1]))
            if self.flex_id not in flex_sides:
                continue
            side = flex_sides.index(self.flex_id)
            if int(contact.geom[1 - side]) not in self.contact_geom_ids:
                continue
            wrench = np.zeros(6)
            mj.mj_contactForce(self.model, data, contact_id, wrench)
            # Contact-frame rows are world axes; the world force acts on the
            # second contact side, so negate when the flex is the first side.
            force_world = contact.frame.reshape(3, 3).T @ wrench[:3]
            if side == 0:
                force_world = -force_world
            for vertex_id, weight in self._splat_weights(data, contact, side):
                vertex_forces[vertex_id] += weight * force_world
            contact_count += 1
        link_rotation = data.xmat[self.link_id].reshape(3, 3)
        axes_world = link_rotation[None, :, :] @ self.vertex_rotations
        readings = np.einsum("ij,ijk->ik", vertex_forces, axes_world)
        readings[:, 2] = -readings[:, 2]
        return readings, contact_count

    def _splat_weights(
        self, data: mj.MjData, contact: mj.MjContact, side: int
    ) -> list[tuple[int, float]]:
        vertex = int(contact.vert[side])
        if 0 <= vertex < self.vert_num:
            return [(vertex, 1.0)]
        element = int(contact.elem[side])
        if element < 0 or element >= len(self.elements):
            vertices = data.flexvert_xpos[
                self.vert_adr : self.vert_adr + self.vert_num
            ]
            distances = np.linalg.norm(vertices - contact.pos[None, :], axis=1)
            return [(int(np.argmin(distances)), 1.0)]
        triangle = self.elements[element]
        points = data.flexvert_xpos[self.vert_adr + triangle]
        basis = np.column_stack((points[1] - points[0], points[2] - points[0]))
        offsets, *_ = np.linalg.lstsq(basis, contact.pos - points[0], rcond=None)
        weights = np.array(
            [1.0 - offsets[0] - offsets[1], offsets[0], offsets[1]]
        )
        weights = np.clip(weights, 0.0, None)
        total = float(weights.sum())
        if total <= 0.0:
            weights = np.full(3, 1.0 / 3.0)
        else:
            weights = weights / total
        return [(int(triangle[i]), float(weights[i])) for i in range(3)]


def write_xml(xml: str, path: Path) -> None:
    path.write_text(xml)
    print(f"Wrote {path}")


def write_scene_xml(model_filename: str) -> str:
    return f"""
<mujoco model="leap_flex_sensor_scene">
  <include file="{model_filename}"/>

  <statistic center="0.02 0.08 0.1" extent="0.3" meansize="0.01"/>
  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".2 .2 .2" specular="1 1 1"/>
    <rgba force="1 0 0 1" haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
    <map force="0.01" stiffness="500"/>
    <scale forcewidth="0.1" contactwidth="0.5" contactheight="0.2"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
      width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
      width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <camera name="side" pos="-0.183 0.396 0.296"
      xyaxes="-0.783 -0.622 0 0.332 -0.419 0.845"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.25"
      material="groundplane" contype="2" conaffinity="2"/>
    <body name="{PROBE_BODY_NAME}" mocap="true" pos="0 0 1">
      <geom name="{PROBE_GEOM_NAME}" type="sphere" size="{PROBE_RADIUS}"
        mass="0.01" rgba="0.1 0.3 1 1" friction="0.8 0.005 0.0001"
        contype="3" conaffinity="3" condim="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def _flex_vertices(
    model: mj.MjModel, data: mj.MjData, flex_name: str
) -> np.ndarray:
    flex_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_FLEX, flex_name)
    address = int(model.flex_vertadr[flex_id])
    count = int(model.flex_vertnum[flex_id])
    return data.flexvert_xpos[address : address + count]


def _surface_normal(
    vertices: np.ndarray, count: tuple[int, int, int]
) -> np.ndarray:
    tangent_u = vertices[(count[0] - 1) * count[1]] - vertices[0]
    tangent_v = vertices[count[1] - 1] - vertices[0]
    normal = np.cross(tangent_u, tangent_v)
    return normal / np.linalg.norm(normal)


def _interior_deformation(
    vertices: np.ndarray, count: tuple[int, int, int]
) -> float:
    normal = _surface_normal(vertices, count)
    interior_ids = [
        vertex_id
        for vertex_id in range(len(vertices))
        if not _is_perimeter_vertex(vertex_id, count)
    ]
    distances = np.abs(
        (vertices[interior_ids] - vertices[0][None, :]) @ normal
    )
    return float(np.max(distances))


def save_heatmap(
    grid: np.ndarray,
    skin: FlexSkin,
    output_path: Path,
    peak_total_force: float,
) -> None:
    extent = [
        -skin.half_u * 1000.0,
        skin.half_u * 1000.0,
        -skin.half_v * 1000.0,
        skin.half_v * 1000.0,
    ]
    fig, axis = plt.subplots(figsize=(5.8, 5.0))
    image = axis.imshow(
        grid[:, :, 2].T,
        cmap="hot",
        interpolation="nearest",
        origin="lower",
        extent=extent,
        aspect="equal",
        vmin=0.0,
    )
    fig.colorbar(image, ax=axis, label="Taxel normal force (N)")
    axis.set_xlabel("flex surface U (mm)")
    axis.set_ylabel("flex surface V (mm)")
    axis.set_title(
        f"{skin.definition.geom_name}: "
        f"{skin.definition.count[0]}x{skin.definition.count[1]} taxels\n"
        f"total normal={peak_total_force:.3f} N"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_sensor_test(
    model: mj.MjModel,
    skin: FlexSkin,
    heatmap_path: Path,
    duration: float,
) -> tuple[dict[str, float], np.ndarray]:
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    sensor = FlexTaxelSensor(
        model,
        skin.flex_name,
        skin.parent_body_name,
        pad_vertex_rotations(skin),
        [PROBE_GEOM_NAME],
    )

    settle_steps = int(np.ceil(REST_SETTLE_DURATION / model.opt.timestep))
    for _ in range(settle_steps):
        mj.mj_step(model, data)
    rest_vertices = _flex_vertices(model, data, skin.flex_name).copy()
    rest_lift = _interior_deformation(rest_vertices, skin.definition.count)
    if rest_lift > REST_FLATNESS_TOLERANCE:
        raise RuntimeError(
            f"Sensor test failed: {skin.flex_name} is not flat at rest "
            f"(interior lift {rest_lift * 1000.0:.3f} mm)"
        )
    link_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, skin.parent_body_name)
    normal = data.xmat[link_id].reshape(3, 3) @ skin.normal
    center = np.mean(rest_vertices, axis=0)
    probe_body_id = mj.mj_name2id(
        model, mj.mjtObj.mjOBJ_BODY, PROBE_BODY_NAME
    )
    probe_mocap_id = int(model.body_mocapid[probe_body_id])
    start_offset = PROBE_RADIUS + FLEX_RADIUS + PROBE_CLEARANCE
    end_offset = PROBE_RADIUS + FLEX_RADIUS - PROBE_PENETRATION

    count = skin.definition.count
    peak_grid = np.zeros((count[0], count[1], 3))
    peak_total_force = 0.0
    peak_taxel_force = 0.0
    peak_contact_count = 0
    peak_deformation = 0.0
    step_count = max(2, int(np.ceil(duration / model.opt.timestep)))
    for step in range(step_count):
        phase = (step + 1) / step_count
        smooth_phase = phase * phase * (3.0 - 2.0 * phase)
        offset = (
            (1.0 - smooth_phase) * start_offset
            + smooth_phase * end_offset
        )
        data.mocap_pos[probe_mocap_id] = center + normal * offset
        mj.mj_step(model, data)
        flat, contact_count = sensor.read(data)
        grid = flat.reshape(count[0], count[1], 3)
        total_force = float(np.sum(grid[:, :, 2]))
        if total_force > peak_total_force:
            peak_total_force = total_force
            peak_grid = grid.copy()
            peak_taxel_force = float(np.max(grid[:, :, 2]))
            peak_contact_count = contact_count
            peak_deformation = _interior_deformation(
                _flex_vertices(model, data, skin.flex_name),
                skin.definition.count,
            )

    if peak_total_force <= 1.0e-3 or peak_contact_count == 0:
        raise RuntimeError(
            f"Sensor test failed: probe did not contact {skin.flex_name}"
        )
    if peak_deformation <= 1.0e-7:
        raise RuntimeError(
            f"Sensor test failed: {skin.flex_name} did not deform"
        )

    save_heatmap(
        peak_grid, skin, heatmap_path, peak_total_force
    )
    stats = {
        "peak_total_force_n": peak_total_force,
        "peak_taxel_force_n": peak_taxel_force,
        "peak_contact_count": float(peak_contact_count),
        "interior_deformation_mm": peak_deformation * 1000.0,
    }
    return stats, peak_grid


def save_tip_heatmap(
    canvas: np.ndarray,
    tip: TipSkin,
    output_path: Path,
    peak_total_force: float,
) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 5.0))
    cmap = plt.get_cmap("hot").copy()
    cmap.set_bad("0.88")
    image = axis.imshow(
        np.ma.masked_invalid(canvas[:, :, 2]),
        cmap=cmap,
        interpolation="nearest",
        origin="upper",
        vmin=0.0,
    )
    fig.colorbar(image, ax=axis, label="Taxel normal force (N)")
    axis.set_xlabel("tip column")
    axis.set_ylabel("tip row (0 = apex)")
    axis.set_title(
        f"{tip.flex_name}: {TIP_TAXEL_COUNT} curved taxels\n"
        f"total normal={peak_total_force:.3f} N"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _vertex_displacements(
    model: mj.MjModel, data: mj.MjData, flex_id: int
) -> np.ndarray:
    address = int(model.flex_vertadr[flex_id])
    number = int(model.flex_vertnum[flex_id])
    displacements = np.zeros(number)
    for offset in range(number):
        body_id = int(model.flex_vertbodyid[address + offset])
        joint_address = int(model.body_jntadr[body_id])
        displacement = np.array([
            data.qpos[int(model.jnt_qposadr[joint_address + axis])]
            for axis in range(3)
        ])
        displacements[offset] = float(np.linalg.norm(displacement))
    return displacements


def tip_membrane_burial(
    model: mj.MjModel, data: mj.MjData, tip: TipSkin
) -> float:
    """Deepest point of the tip membrane that sits inside the finger shell.

    Samples each triangle's centroid and edge midpoints and casts outward
    along the averaged vertex normal: a hit means the sample is buried.
    """
    shell_ids = []
    for suffix in ("", "_2", "_3", "_4", "_5", "_6"):
        geom_id = mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_GEOM, f"{tip.finger}_ds_tip{suffix}"
        )
        if geom_id >= 0:
            shell_ids.append(geom_id)
    vertices = _flex_vertices(model, data, tip.flex_name)
    link_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, tip.parent_body_name)
    normals = (
        data.xmat[link_id].reshape(3, 3) @ tip.vertex_rotations[:, :, 2].T
    ).T
    deepest = 0.0
    for triangle in np.asarray(TIP_ELEMENTS).reshape(-1, 3):
        corners = vertices[triangle]
        direction = np.mean(normals[triangle], axis=0)
        direction = direction / np.linalg.norm(direction)
        samples = [np.mean(corners, axis=0)] + [
            0.5 * (corners[i] + corners[(i + 1) % 3]) for i in range(3)
        ]
        for sample in samples:
            for geom_id in shell_ids:
                hit = mj.mj_rayMesh(model, data, geom_id, sample, direction)
                deepest = max(deepest, float(hit))
    return deepest


def run_tip_sensor_test(
    model: mj.MjModel,
    tip: TipSkin,
    heatmap_path: Path,
    duration: float,
) -> tuple[dict[str, float], np.ndarray]:
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    sensor = FlexTaxelSensor(
        model,
        tip.flex_name,
        tip.parent_body_name,
        tip.vertex_rotations,
        [PROBE_GEOM_NAME],
    )

    settle_steps = int(np.ceil(REST_SETTLE_DURATION / model.opt.timestep))
    for _ in range(settle_steps):
        mj.mj_step(model, data)
    flex_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_FLEX, tip.flex_name)
    rest_drift = float(np.max(_vertex_displacements(model, data, flex_id)))
    if rest_drift > REST_FLATNESS_TOLERANCE: 
        raise RuntimeError(
            f"Tip test failed: {tip.flex_name} drifted "
            f"{rest_drift * 1000.0:.3f} mm from its calibrated poses at rest"
        )
    burial = tip_membrane_burial(model, data, tip)
    if burial > 0.0:
        raise RuntimeError(
            f"Tip test failed: {tip.flex_name} is buried "
            f"{burial * 1000.0:.2f} mm inside the finger shell"
        )
        print(f"{burial * 1000.0:.2f} mm inside the finger shell")
    link_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, tip.parent_body_name)
    link_rotation = data.xmat[link_id].reshape(3, 3)
    press_ids = list(TIP_PRESS_TAXELS)
    vertices = _flex_vertices(model, data, tip.flex_name).copy()
    center = np.mean(vertices[press_ids], axis=0)
    normal = np.mean(
        [link_rotation @ tip.vertex_rotations[i][:, 2] for i in press_ids],
        axis=0,
    )
    normal /= np.linalg.norm(normal)
    probe_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, PROBE_BODY_NAME)
    probe_mocap_id = int(model.body_mocapid[probe_body_id])
    start_offset = PROBE_RADIUS + FLEX_RADIUS + PROBE_CLEARANCE
    end_offset = PROBE_RADIUS + FLEX_RADIUS - PROBE_PENETRATION

    peak_flat = np.zeros((TIP_TAXEL_COUNT, 3))
    peak_total_force = 0.0
    peak_taxel_force = 0.0
    peak_contact_count = 0
    peak_displacement = 0.0
    step_count = max(2, int(np.ceil(duration / model.opt.timestep)))
    for step in range(step_count):
        phase = (step + 1) / step_count
        smooth_phase = phase * phase * (3.0 - 2.0 * phase)
        offset = (
            (1.0 - smooth_phase) * start_offset
            + smooth_phase * end_offset
        )
        data.mocap_pos[probe_mocap_id] = center + normal * offset
        mj.mj_step(model, data)
        flat, contact_count = sensor.read(data)
        total_force = float(np.sum(flat[:, 2]))
        if total_force > peak_total_force:
            peak_total_force = total_force
            peak_flat = flat.copy()
            peak_taxel_force = float(np.max(flat[:, 2]))
            peak_contact_count = contact_count
            peak_displacement = float(
                np.max(_vertex_displacements(model, data, flex_id))
            )

    if peak_total_force <= 1.0e-3 or peak_contact_count == 0:
        raise RuntimeError(
            f"Tip test failed: probe did not contact {tip.flex_name}"
        )
    if peak_displacement <= 1.0e-7:
        raise RuntimeError(
            f"Tip test failed: {tip.flex_name} did not deform"
        )

    canvas = tip_canvas(tip, peak_flat)
    save_tip_heatmap(canvas, tip, heatmap_path, peak_total_force)
    stats = {
        "peak_total_force_n": peak_total_force,
        "peak_taxel_force_n": peak_taxel_force,
        "peak_contact_count": float(peak_contact_count),
        "peak_displacement_mm": peak_displacement * 1000.0,
    }
    return stats, canvas


def _snap_to_axis(direction: np.ndarray, length: float) -> np.ndarray:
    """Nearest axis-aligned vector of `length`, keeping the sign."""
    if abs(direction[0]) >= abs(direction[1]):
        return np.array([np.sign(direction[0]) * length, 0.0])
    return np.array([0.0, np.sign(direction[1]) * length])


def build_hand_layout(
    model: mj.MjModel,
    data: mj.MjData,
    skins: list[FlexSkin],
    tips: list[TipSkin],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Project each skin's taxel-cell corners onto the palm plane (mm).

    Returns {skin_name: (X, Y)} with (nu+1, nv+1) pcolormesh corner arrays in
    palm-frame coordinates: x across the knuckles, z toward the fingertips.
    """
    settle_steps = int(np.ceil(REST_SETTLE_DURATION / model.opt.timestep))
    for _ in range(settle_steps):
        mj.mj_step(model, data)
    palm_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "palm")
    if palm_id < 0:
        raise ValueError("Body 'palm' not found")
    palm_rotation = data.xmat[palm_id].reshape(3, 3)
    palm_position = data.xpos[palm_id]

    def project(point_world: np.ndarray) -> np.ndarray:
        local = palm_rotation.T @ (point_world - palm_position)
        return np.array([local[0], local[2]])

    layout = {}
    for skin in skins:
        link_id = mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_BODY, skin.parent_body_name
        )
        link_rotation = data.xmat[link_id].reshape(3, 3)
        vertices = _flex_vertices(model, data, skin.flex_name)
        center_2d = project(np.mean(vertices, axis=0))
        origin_2d = project(palm_position)
        axis_u_2d = (
            project(palm_position + link_rotation @ skin.tangent_u) - origin_2d
        )
        axis_v_2d = (
            project(palm_position + link_rotation @ skin.tangent_v) - origin_2d
        )
        # Patches nearly edge-on to the palm plane would project to slivers;
        # draw those at full physical scale instead.
        determinant = (
            axis_u_2d[0] * axis_v_2d[1] - axis_u_2d[1] * axis_v_2d[0]
        )
        if abs(determinant) < 0.4:
            norm_u = float(np.linalg.norm(axis_u_2d))
            axis_u_2d = (
                axis_u_2d / norm_u if norm_u > 1.0e-6 else np.array([1.0, 0.0])
            )
            perpendicular = np.array([-axis_u_2d[1], axis_u_2d[0]])
            if float(perpendicular @ axis_v_2d) < 0.0:
                perpendicular = -perpendicular
            axis_v_2d = perpendicular

        count = skin.definition.count
        # Cells evenly tile the physical pad footprint so neighboring pads
        # (the palm trio) don't overlap in the figure.
        corners_u = np.linspace(-skin.half_u, skin.half_u, count[0] + 1)
        corners_v = np.linspace(-skin.half_v, skin.half_v, count[1] + 1)
        corners = (
            center_2d[None, None, :]
            + corners_u[:, None, None] * axis_u_2d[None, None, :]
            + corners_v[None, :, None] * axis_v_2d[None, None, :]
        )
        layout[skin.definition.geom_name] = (
            corners[:, :, 0] * 1000.0,
            corners[:, :, 1] * 1000.0,
        )
    # A tip canvas is schematic (fixed pitch), but its orientation is fitted to
    # the real projected taxels so columns are not drawn mirrored.
    canvas_columns = np.array([column for _, column in TIP_GRID_ROWCOL], float)
    canvas_rows = np.array([row for row, _ in TIP_GRID_ROWCOL], float)
    canvas_basis = np.column_stack(
        [np.ones(len(canvas_columns)), canvas_columns, canvas_rows]
    )
    pitch = TIP_HEATMAP_PITCH * 1000.0
    for tip in tips:
        vertices = _flex_vertices(model, data, tip.flex_name)
        projected = np.array([project(vertex) for vertex in vertices]) * 1000.0
        coefficients, *_ = np.linalg.lstsq(canvas_basis, projected, rcond=None)
        origin, column_step, row_step = coefficients
        # The thumb tip faces across the palm, so one of its canvas axes nearly
        # collapses in this projection: keep the well-determined direction and
        # take the other perpendicular to it. Both are then snapped to the
        # nearest panel axis, which leaves the finger canvases untouched and
        # renders the thumb's genuinely sideways canvas cleanly instead of
        # tilted by a few degrees.
        if np.linalg.norm(column_step) >= np.linalg.norm(row_step):
            primary, secondary, primary_is_column = column_step, row_step, True
        else:
            primary, secondary, primary_is_column = row_step, column_step, False
        primary = primary / np.linalg.norm(primary)
        perpendicular = np.array([-primary[1], primary[0]])
        if perpendicular @ secondary < 0.0:
            perpendicular = -perpendicular
        if primary_is_column:
            column_step, row_step = primary, perpendicular
        else:
            row_step, column_step = primary, perpendicular
        column_step = _snap_to_axis(column_step, pitch)
        row_step = _snap_to_axis(row_step, pitch)

        grid_columns, grid_rows = np.meshgrid(
            np.arange(7) - 0.5, np.arange(7) - 0.5, indexing="xy"
        )
        corners = (
            origin[None, None, :]
            + grid_columns[:, :, None] * column_step[None, None, :]
            + grid_rows[:, :, None] * row_step[None, None, :]
        )
        layout[f"{tip.finger}_tip"] = (corners[:, :, 0], corners[:, :, 1])
    return layout


HAND_HEATMAP_CHANNELS = (
    ("shear-x", 0, "RdBu_r"),
    ("shear-y", 1, "RdBu_r"),
    ("normal-z", 2, "Greens"),
)


def save_hand_heatmap(
    layout: dict[str, tuple[np.ndarray, np.ndarray]],
    grids: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    shear_limit = max(
        float(np.nanmax(np.abs(grid[:, :, :2]))) for grid in grids.values()
    )
    shear_limit = max(shear_limit, 1.0e-4)
    normal_limit = max(
        float(np.nanmax(grid[:, :, 2])) for grid in grids.values()
    )
    normal_limit = max(normal_limit, 1.0e-4)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.5))
    for axis, (label, channel, cmap) in zip(axes, HAND_HEATMAP_CHANNELS):
        if channel < 2:
            vmin, vmax = -shear_limit, shear_limit
        else:
            vmin, vmax = 0.0, normal_limit
        mesh = None
        for name, (corners_x, corners_y) in layout.items():
            mesh = axis.pcolormesh(
                corners_x,
                corners_y,
                np.ma.masked_invalid(grids[name][:, :, channel]),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgecolors="0.85",
                linewidth=0.3,
            )
            short = name.replace("_uspa44", "").replace("uspa46_", "palm ")
            axis.text(
                float(np.mean(corners_x)),
                float(np.min(corners_y)) - 2.5,
                short,
                ha="center",
                va="top",
                fontsize=6.5,
                color="0.35",
            )
        all_x = np.concatenate(
            [corners[0].ravel() for corners in layout.values()]
        )
        all_y = np.concatenate(
            [corners[1].ravel() for corners in layout.values()]
        )
        axis.set_xlim(float(all_x.min()) - 6.0, float(all_x.max()) + 6.0)
        axis.set_ylim(float(all_y.min()) - 12.0, float(all_y.max()) + 6.0)
        axis.set_aspect("equal")
        axis.set_title(label)
        axis.set_xlabel("palm x (mm)")
        for spine in axis.spines.values():
            spine.set_color("0.8")
        axis.tick_params(colors="0.5", labelsize=7)
        fig.colorbar(mesh, ax=axis, shrink=0.75, label="force (N)")
    axes[0].set_ylabel("palm z (mm)")
    fig.suptitle("Flex taxel readings — palm view, fingers up")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_curl_test(model: mj.MjModel) -> dict[str, float]:
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    max_ratio = np.ones(model.nflex)
    peak_error = np.zeros(model.nflex)
    qpos_index = np.array([
        [
            int(model.jnt_qposadr[
                int(model.body_jntadr[int(model.flex_vertbodyid[vertex])])
                + axis
            ])
            for axis in range(3)
        ]
        for vertex in range(model.nflexvert)
    ])

    def home_errors() -> np.ndarray:
        norms = np.linalg.norm(data.qpos[qpos_index], axis=1)
        return np.array([
            float(np.max(
                norms[
                    int(model.flex_vertadr[flex_id]) :
                    int(model.flex_vertadr[flex_id])
                    + int(model.flex_vertnum[flex_id])
                ]
            ))
            for flex_id in range(model.nflex)
        ])

    def drive(target: np.ndarray) -> None:
        start = data.ctrl.copy()
        total = CURL_RAMP_DURATION + CURL_HOLD_DURATION + CURL_RETURN_DURATION
        step_count = int(np.ceil(total / model.opt.timestep))
        for step in range(step_count):
            t = (step + 1) * model.opt.timestep
            if t < CURL_RAMP_DURATION:
                phase = t / CURL_RAMP_DURATION
            elif t < CURL_RAMP_DURATION + CURL_HOLD_DURATION:
                phase = 1.0
            else:
                phase = max(
                    0.0,
                    1.0
                    - (t - CURL_RAMP_DURATION - CURL_HOLD_DURATION)
                    / CURL_RETURN_DURATION,
                )
            smooth = phase * phase * (3.0 - 2.0 * phase)
            data.ctrl[:] = start + smooth * (target - start)
            mj.mj_step(model, data)
            ratios = data.flexedge_length / model.flexedge_length0
            for flex_id in range(model.nflex):
                address = int(model.flex_edgeadr[flex_id])
                number = int(model.flex_edgenum[flex_id])
                max_ratio[flex_id] = max(
                    max_ratio[flex_id],
                    float(np.max(ratios[address : address + number])),
                )
            np.maximum(peak_error, home_errors(), out=peak_error)

    thumb_target = data.ctrl.copy()
    for actuator_id in range(model.nu):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name.startswith("th_"):
            thumb_target[actuator_id] = low[actuator_id] + CURL_FRACTION * (
                high[actuator_id] - low[actuator_id]
            )
    drive(thumb_target)
    drive(low + CURL_FRACTION * (high - low))
    for _ in range(int(np.ceil(REST_SETTLE_DURATION / model.opt.timestep))):
        mj.mj_step(model, data)
    residual = home_errors()

    divergences = int(data.warning[mj.mjtWarning.mjWARN_BADQACC].number)
    if divergences > 0:
        raise RuntimeError(
            f"Curl test failed: {divergences} solver divergences during motion"
        )
    worst_id = int(np.argmax(max_ratio))
    worst_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_FLEX, worst_id)
    worst = float(max_ratio[worst_id])
    if worst > CURL_MAX_STRETCH_RATIO:
        raise RuntimeError(
            f"Curl test failed: {worst_name} stretched {worst:.2f}x "
            f"(limit {CURL_MAX_STRETCH_RATIO:.1f}x)"
        )
    lag_id = int(np.argmax(peak_error))
    lag_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_FLEX, lag_id)
    lag = float(peak_error[lag_id])
    if lag > CURL_PEAK_HOME_ERROR:
        raise RuntimeError(
            f"Curl test failed: {lag_name} lagged {lag * 1000.0:.1f} mm from "
            f"home (limit {CURL_PEAK_HOME_ERROR * 1000.0:.1f} mm)"
        )
    residual_id = int(np.argmax(residual))
    residual_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_FLEX, residual_id)
    residual_worst = float(residual[residual_id])
    if residual_worst > CURL_RESIDUAL_HOME_ERROR:
        raise RuntimeError(
            f"Curl test failed: {residual_name} kept a residual displacement "
            f"of {residual_worst * 1000.0:.2f} mm after returning "
            f"(limit {CURL_RESIDUAL_HOME_ERROR * 1000.0:.2f} mm)"
        )
    print(
        f"PASS curl: max edge stretch {worst:.2f}x on {worst_name}, "
        f"peak lag {lag * 1000.0:.1f} mm on {lag_name}, "
        f"residual {residual_worst * 1000.0:.2f} mm on {residual_name}, "
        f"no divergences."
    )
    return {
        "max_stretch_ratio": worst,
        "peak_home_error_m": lag,
        "residual_home_error_m": residual_worst,
        "divergences": float(divergences),
    }


def run_all_sensor_tests(
    scene_path: Path,
    skins: list[FlexSkin],
    tips: list[TipSkin],
    output_dir: Path,
    duration: float,
) -> dict[str, dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = mj.MjModel.from_xml_path(scene_path.as_posix())
    results = {}
    grids = {}
    for skin in skins:
        heatmap_path = output_dir / f"{skin.definition.geom_name}.png"
        stats, peak_grid = run_sensor_test(model, skin, heatmap_path, duration)
        results[skin.definition.geom_name] = stats
        grids[skin.definition.geom_name] = peak_grid
        print(
            f"PASS {skin.definition.geom_name}: "
            f"force={stats['peak_total_force_n']:.3f} N, "
            f"peak taxel={stats['peak_taxel_force_n']:.3f} N, "
            f"deformation={stats['interior_deformation_mm']:.3f} mm"
        )
    for tip in tips:
        heatmap_path = output_dir / f"{tip.finger}_tip.png"
        stats, canvas = run_tip_sensor_test(model, tip, heatmap_path, duration)
        results[f"{tip.finger}_tip"] = stats
        grids[f"{tip.finger}_tip"] = canvas
        print(
            f"PASS {tip.finger}_tip: "
            f"force={stats['peak_total_force_n']:.3f} N, "
            f"peak taxel={stats['peak_taxel_force_n']:.3f} N, "
            f"displacement={stats['peak_displacement_mm']:.3f} mm"
        )
    layout = build_hand_layout(model, mj.MjData(model), skins, tips)
    hand_heatmap_path = output_dir / "hand_taxel_heatmap.png"
    save_hand_heatmap(layout, grids, hand_heatmap_path)
    print(f"Wrote combined hand heatmap: {hand_heatmap_path}")
    results["curl"] = run_curl_test(model)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and test flex skins over LEAP XELA box sensors."
    )
    parser.add_argument("--mode", choices=FINGER_TIP_TYPES, default="Box")
    parser.add_argument("--test-duration", type=float, default=0.35)
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries_by_patch = defaultdict(list)
    for entry in build_layout().entries:
        entries_by_patch[entry.patch].append(entry)
    print(f"Loaded canonical taxel layout from {FINGERTIP_POSE_JSON.parent}")

    spec = load_base_model(args.mode)
    spec.option.iterations = SOLVER_ITERATIONS
    spec.option.tolerance = SOLVER_TOLERANCE
    trim_pad_boxes(spec)
    skins = [
        add_flex_sensor(
            spec,
            definition,
            entries_by_patch[GEOM_TO_PATCH[definition.geom_name]],
        )
        for definition in SENSOR_DEFINITIONS
    ]
    magnet_pose = json.loads(FINGERTIP_POSE_JSON.read_text())
    surface_offsets = tip_surface_offsets(magnet_pose)
    tips = [
        add_tip_flex(
            spec,
            finger,
            entries_by_patch[f"tip_{finger}"],
            surface_offsets[finger],
        )
        for finger in TIP_FINGERS
    ]
    assigned = np.concatenate(
        [skin.taxel_ids for skin in skins] + [tip.taxel_ids for tip in tips]
    )
    if sorted(assigned.tolist()) != list(range(TAXEL_COUNT)):
        raise ValueError(
            f"Taxel ids do not cover 0..{TAXEL_COUNT - 1} exactly "
            f"({len(assigned)} assigned, {len(set(assigned.tolist()))} unique)"
        )
    print(f"Taxel ids cover 0..{TAXEL_COUNT - 1} with no duplicates.")

    generated_model_path = _MODEL_DIR / "leapXela_generated_flex_sensor.xml"
    scene_path = _MODEL_DIR / f"scene_flex_sensor_{args.mode}.xml"
    heatmap_dir = _MODEL_DIR / "flex_sensor_heatmaps"
    write_xml(spec.to_xml(), generated_model_path)
    write_xml(write_scene_xml(generated_model_path.name), scene_path)

    if not args.skip_test:
        run_all_sensor_tests(
            scene_path, skins, tips, heatmap_dir, args.test_duration
        )
        print(
            f"All {len(skins) + len(tips)} virtual flex sensors passed. "
            f"Heatmaps: {heatmap_dir}"
        )


if __name__ == "__main__":
    main()
