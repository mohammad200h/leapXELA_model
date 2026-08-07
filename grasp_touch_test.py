"""Grasp test for the flex taxel sensors.

Places an object on the palm-up hand, closes the fingers, then pulse-squeezes
while recording all 368 taxels in hardware taxel-ID order. Saves an npz sensor
log, 3-channel hand-heatmap snapshots, and an orbiting MP4 render with the live
heatmap drawn on the hand and as a palm-view panel.
"""

import argparse
import itertools
import zlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mediapy
import mujoco as mj
import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from generatehand_flexcom_sensor import (
    FINGERTIP_POSE_JSON,
    GEOM_TO_PATCH,
    HAND_WORKSPACE,
    REST_SETTLE_DURATION,
    SENSOR_DEFINITIONS,
    TAXEL_COUNT,
    TIP_FINGERS,
    TIP_GRID_ROWCOL,
    FlexTaxelSensor,
    add_flex_sensor,
    add_tip_flex,
    build_hand_layout,
    load_base_model,
    pad_vertex_rotations,
    save_hand_heatmap,
    tip_canvas,
    tip_surface_offsets,
    trim_pad_boxes,
    _flex_vertices,
)
from leapxela.taxel_layout import build_layout
from taxel_visualizer import (
    CHANNELS,
    CLOUD_MARKER_RADIUS,
    MARKER_RADIUS,
    TaxelPanel,
    bare_scene_option,
    channel_values,
    dim_flex_skins,
    draw_taxel_arrows,
    draw_taxel_markers,
    marker_colours,
    panel_taxel_ids,
    paste_panel,
)


OBJECT_BODY_NAME = "grasp_object"
OBJECT_GEOM_NAME = "grasp_object_geom"
# Multi-object placement. Objects must start low: a 30 mm per-object stagger
# dropped N=5 from ~120 mm and lost half of them.
PLACEMENT_JITTER = 0.012
PLACEMENT_HEIGHT = 0.008
SETTLE_DURATION = 0.5
PLACEMENT_ATTEMPTS = 40
MULTI_SHAPES = ("sphere", "box", "cylinder")
MULTI_SCALE_RANGE = (0.45, 0.80)
# Distinct colours so objects can be told apart in the video.
OBJECT_COLOURS = (
    [0.15, 0.35, 0.90, 1.0], [0.90, 0.45, 0.10, 1.0],
    [0.20, 0.75, 0.35, 1.0], [0.80, 0.20, 0.60, 1.0],
    [0.95, 0.85, 0.20, 1.0],
)
OBJECT_MASS = 0.8
# (slide, spin, roll) on the object geoms. Shared with the object-object contact
# pairs below: a <pair> ignores the geoms' friction and falls back to its own
# default of slide 1.0, so writing it in one place is what keeps adhesion the
# only thing those pairs change.
OBJECT_FRICTION = (0.8, 0.005, 0.0001)
# Default reach of object-object adhesion. This has to span the gap objects are
# PLACED at, not just the contact tolerance: at 2 mm the objects never touch each
# other and adhesion does literally nothing (3 spheres measured identical to
# adhesion off, 1/3 held and 0/3 sensed). At 10 mm the same case goes to 3/3 held
# and 3/3 sensed. 25 mm over-clumps -- 5 boxes drop from 4/5 sensed to 3/5 -- so
# the useful range is bounded on both sides.
OBJECT_ADHESION_GAP = 0.010
# pair/adhesion landed in MuJoCo 3.11.0 (2026-07-27). 3.10 silently lacks the
# attribute -- spec.add_pair() has no `adhesion` field at all -- so without this
# check the flag would appear to work and do nothing.
ADHESION_MIN_MUJOCO = (3, 11)
# Share of post-close frames an object must be touching a membrane to count
# toward objects_sensed.
SENSED_FRACTION = 0.5
OBJECT_SIZES = {
    "sphere": [0.028, 0.0, 0.0],
    "box": [0.022, 0.022, 0.022],
    "cylinder": [0.02, 0.045, 0.0],
}
OBJECT_GEOM_TYPES = {
    "sphere": mj.mjtGeom.mjGEOM_SPHERE,
    "box": mj.mjtGeom.mjGEOM_BOX,
    "cylinder": mj.mjtGeom.mjGEOM_CYLINDER,
}
# The sensor pads are separated islands on an open palm, so a free-standing
# object (especially the sphere) is in unstable equilibrium and rolls off
# within ~0.9 s. The object is therefore placed at its resting height and the
# fingers start cradling it immediately, which is also how you would load a
# real hand.
CLOSE_START = 0.0
CLOSE_END = 1.5
SPAWN_GAP = 0.001
PALM_GEOM_NAMES = ("uspa46_1", "uspa46_2", "uspa46_3")
# A ball on the flat, level palm sits in unstable equilibrium between the
# separated pads and escapes before the fingers close. Three measures keep it
# in: spawn over the fingerward pads, pitch the palm so any roll runs into the
# fingers, and start with the fingers already slightly curled.
SPAWN_PAD_NAMES = ("uspa46_1", "uspa46_2")
# Pitching the palm fingers-down (as leapxela/scene_builder.py does) retains the
# object but rolls it past the pads into the unsensed pocket between the palm
# and the finger bases: measured, even -2 deg drops the 0.5x sphere from 24
# sensed contacts to 0. The fingerward spawn and pre-curl below achieve the
# retention on their own, so the default is level; --palm-pitch still tilts.
PALM_PITCH_DEG = 0.0
# Measured: at 0.15 the palm is too flat to hold a cluster (1-2 of 3 objects
# retained); the deeper cup formed at 0.40 retains 3 of 3. Tightening the
# cluster instead made retention worse, so the bowl depth is the mechanism.
PREGRIP_FRACTION = 0.40
# At a common grip fraction the commanded pose is self-colliding: the thumb is
# driven into the index and stalls ~57 deg short of its target even with no
# object present. Giving the thumb a smaller closure target removes the
# collision entirely; delaying its ramp additionally frees the index.
THUMB_GRIP_FRACTION = 0.35
THUMB_DELAY = 0.5
GRASP_PATTERNS = ("hold", "pulse", "squeeze", "regrasp", "tap", "shear")
# regrasp and tap deliberately break contact, so they cannot be held to the
# same contact-fraction floor as the patterns that stay closed.
PATTERN_MIN_CONTACT = {
    "hold": 0.5, "pulse": 0.5, "squeeze": 0.5, "shear": 0.5,
    "regrasp": 0.3, "tap": 0.2,
}
SQUEEZE_STEP_SECONDS = 0.9
REGRASP_HZ = 0.35
TAP_HZ = 1.5
TAP_DEPTH = 0.5
SHEAR_HZ = 0.7
VIDEO_FPS_TARGET = 30.0
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
# The camera circles the hand so no contact stays hidden behind the fingers
# for the whole clip. Start on the familiar side view used elsewhere.
ORBIT_DISTANCE = 0.30
ORBIT_ELEVATION = -25.0
ORBIT_START_AZIMUTH = 120.0
ORBIT_LOOKAT_LIFT = 0.02
PANEL_SIZE = 320
PANEL_MARGIN_PX = 16
# Bare taxel cloud rendered beside the hand, same viewpoint, nothing occluding.
# The camera is framed on the cloud itself: the thumb sits far lateral of the
# fingers, so the cloud spans ~235 mm and needs its own distance, not the hand's.
CLOUD_WIDTH = 640
# The cloud shrinks a lot as the hand closes (181 mm radius at pre-grip vs
# 78 mm once gripping), so the camera tracks the live centroid and extent
# instead of being framed once on the rest pose.
CLOUD_TRACK_ALPHA = 0.1
CLOUD_MARGIN = 1.12
CLOUD_GRIDS = ("surface", "magnet")


def cloud_camera_distance(model: mj.MjModel, radius: float) -> float:
    """Distance at which a bounding sphere of `radius` fills the cloud panel.

    Derived from the camera's field of view rather than a tuned constant: the
    panel is narrower than it is tall, so the horizontal fov is the limiting
    one.
    """
    fovy = np.radians(float(model.vis.global_.fovy))
    aspect = CLOUD_WIDTH / VIDEO_HEIGHT
    fovx = 2.0 * np.arctan(np.tan(0.5 * fovy) * aspect)
    return CLOUD_MARGIN * radius / np.sin(0.5 * min(fovy, fovx))


# Acceptance needs every spawned object, so an episode that has already lost one
# can never be kept. Measured over 10 s episodes, the held count reaches its
# final value by t~0.5-1.0 s and never recovers, so simulating the remaining 9 s
# only delays a rejection that is already decided. Checked after CLOSE_END so the
# grip has fully closed. Rejected samples are discarded either way -- nothing
# that reaches the dataset changes, only the time spent not reaching it.
ABORT_CHECK_TIME = 2.0


MIN_PEAK_TOTAL_NORMAL = 0.1
# Multi-object floor: only rejects episodes with no sensed contact at all, well
# below the 0.04 N faint grasps that are deliberately kept.
MULTI_MIN_PEAK_NORMAL = 0.01
MAX_FINAL_DISTANCE = 0.15


def layout_entries_by_patch() -> dict:
    entries_by_patch = {}
    for entry in build_layout(HAND_WORKSPACE).entries:
        entries_by_patch.setdefault(entry.patch, []).append(entry)
    return entries_by_patch


@dataclass(frozen=True)
class ObjectSpec:
    shape: str
    scale: float
    mass: float


def object_body_name(index: int) -> str:
    return OBJECT_BODY_NAME if index == 0 else f"{OBJECT_BODY_NAME}_{index}"


def object_geom_name(index: int) -> str:
    return OBJECT_GEOM_NAME if index == 0 else f"{OBJECT_GEOM_NAME}_{index}"


def add_object_adhesion(spec: mj.MjSpec, count: int, adhesion: float, gap: float) -> int:
    """Stick the objects to each other -- and to nothing else. Returns pair count.

    Scoped through explicit <pair> elements rather than geom/adhesion because a
    contact takes the SUM of its two geoms' adhesion: a value on the object
    geoms would leak into every object-hand contact as `a + 0`, and those are
    exactly the contacts the taxels read. A pair overrides that sum for the
    named pair alone, so flex-vs-object contacts stay at adhesion 0 and the
    recorded signal is untouched.

    `gap` is not optional. Adhesion only acts while a contact exists, and with
    gap=0 a contact exists only while the geoms interpenetrate -- so the moment
    objects begin to separate the contact disappears and takes the pull with it.
    Measured on two spheres: at gap=0 even 0.5 N (2.4x the object's weight) only
    slowed the fall, while 1 mm of gap held it.
    """
    if adhesion <= 0.0:
        return 0
    version = tuple(int(part) for part in mj.__version__.split(".")[:2])
    if version < ADHESION_MIN_MUJOCO:
        raise RuntimeError(
            f"--object-adhesion needs MuJoCo >= "
            f"{'.'.join(str(v) for v in ADHESION_MIN_MUJOCO)}, found {mj.__version__}. "
            "pair/adhesion does not exist in this build and would be silently ignored."
        )
    if gap <= 0.0:
        raise ValueError(
            f"object adhesion {adhesion} N needs a non-zero gap to do anything; "
            "with gap=0 the contact vanishes as soon as the objects separate"
        )
    slide, spin, roll = OBJECT_FRICTION
    pairs = 0
    for first, second in itertools.combinations(range(count), 2):
        pair = spec.add_pair()
        pair.geomname1 = object_geom_name(first)
        pair.geomname2 = object_geom_name(second)
        pair.condim = 3
        pair.friction = [slide, slide, spin, roll, roll]
        pair.adhesion = adhesion
        pair.gap = gap
        pairs += 1
    return pairs


def sensed_objects(data: mj.MjData, object_geom_ids: np.ndarray) -> np.ndarray:
    """Boolean per object: is it touching a taxel-bearing membrane right now?

    `count_held` is kinematic -- distance from the palm and height -- so an
    object resting in the unsensed pocket between the palm pads and the finger
    bases counts as held while contributing nothing to the tactile signal. This
    is the count the taxels actually support, and the two differ exactly in the
    case this adhesion work exists to fix.
    """
    sensed = np.zeros(len(object_geom_ids), dtype=bool)
    count = data.ncon
    if count == 0:
        return sensed
    flex = np.asarray(data.contact.flex)[:count]
    geom = np.asarray(data.contact.geom)[:count]
    for side in (0, 1):
        touching = geom[(flex[:, side] >= 0), 1 - side]
        sensed |= np.isin(object_geom_ids, touching)
    return sensed


def assert_adhesion_scoped(data: mj.MjData, object_geom_ids: np.ndarray) -> None:
    """Every adhesive contact must be object-object, and never touch a flex.

    The entire case for object-only adhesion is that the contacts the taxels
    read are left alone, so it is checked rather than trusted: a stray geom-level
    adhesion, or a MuJoCo change to how pair adhesion is resolved, would
    otherwise quietly bias every force in the dataset.
    """
    count = data.ncon
    if count == 0:
        return
    adhesive = np.asarray(data.contact.adhesion)[:count] != 0.0
    if not adhesive.any():
        return
    flex = np.asarray(data.contact.flex)[:count][adhesive]
    geoms = np.asarray(data.contact.geom)[:count][adhesive]
    if (flex >= 0).any():
        raise AssertionError(
            "adhesion reached a flex contact: the taxel readings are being "
            f"biased (flex ids {np.unique(flex[flex >= 0]).tolist()})"
        )
    if not np.isin(geoms, object_geom_ids).all():
        stray = np.unique(geoms[~np.isin(geoms, object_geom_ids)]).tolist()
        raise AssertionError(
            f"adhesion reached non-object geoms {stray}; it must apply between "
            "objects only"
        )


def build_model(
    specs: list, palm_pitch_deg: float, object_adhesion: float, object_adhesion_gap: float
):
    entries_by_patch = layout_entries_by_patch()
    spec = load_base_model("Box")
    trim_pad_boxes(spec)
    tilt_palm(spec, palm_pitch_deg)
    skins = [
        add_flex_sensor(
            spec, definition, entries_by_patch[GEOM_TO_PATCH[definition.geom_name]]
        )
        for definition in SENSOR_DEFINITIONS
    ]
    magnet_pose = json.loads(FINGERTIP_POSE_JSON.read_text())
    surface_offsets = tip_surface_offsets(magnet_pose)
    tips = [
        add_tip_flex(
            spec, finger, entries_by_patch[f"tip_{finger}"], surface_offsets[finger]
        )
        for finger in TIP_FINGERS
    ]
    camera_x = np.array([-0.783, -0.622, 0.0])
    camera_y = np.array([0.332, -0.419, 0.845])
    camera_x /= np.linalg.norm(camera_x)
    camera_y /= np.linalg.norm(camera_y)
    camera_z = np.cross(camera_x, camera_y)
    camera_quat = np.zeros(4)
    mj.mju_mat2Quat(
        camera_quat, np.column_stack((camera_x, camera_y, camera_z)).ravel()
    )
    spec.worldbody.add_camera(
        name="side", pos=[-0.183, 0.396, 0.296], quat=camera_quat.tolist()
    )
    spec.worldbody.add_light(
        name="top", pos=[0.0, 0.3, 1.0], dir=[0.0, -0.2, -1.0], type=mj.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    for index, item in enumerate(specs):
        # Parked apart until the episode places them; the real pose is written
        # into qpos after the hand has been pre-curled.
        body = spec.worldbody.add_body(
            name=object_body_name(index), pos=[0.0, 0.0, 0.5 + 0.1 * index]
        )
        body.add_freejoint()
        body.add_geom(
            name=object_geom_name(index),
            type=OBJECT_GEOM_TYPES[item.shape],
            size=(
                np.asarray(OBJECT_SIZES[item.shape], dtype=np.float64)
                * item.scale
            ).tolist(),
            mass=item.mass,
            rgba=OBJECT_COLOURS[index % len(OBJECT_COLOURS)],
            contype=3,
            conaffinity=3,
            condim=3,
            friction=list(OBJECT_FRICTION),
        )
    add_object_adhesion(spec, len(specs), object_adhesion, object_adhesion_gap)
    model = spec.compile()
    model.vis.global_.offwidth = max(VIDEO_WIDTH, CLOUD_WIDTH)
    model.vis.global_.offheight = VIDEO_HEIGHT
    model.vis.headlight.ambient = [0.3, 0.3, 0.3]
    model.vis.headlight.diffuse = [0.7, 0.7, 0.7]
    model.vis.headlight.specular = [0.8, 0.8, 0.8]
    return model, skins, tips


def tilt_palm(spec: mj.MjSpec, pitch_deg: float) -> None:
    """Pitch the welded palm fingers-down about the world x axis."""
    if pitch_deg == 0.0:
        return
    palm = spec.body("palm")
    rotation = np.zeros(4)
    mj.mju_axisAngle2Quat(
        rotation, np.array([1.0, 0.0, 0.0]), np.deg2rad(pitch_deg)
    )
    tilted = np.zeros(4)
    mj.mju_mulQuat(tilted, rotation, np.asarray(palm.quat, dtype=np.float64))
    palm.quat = tilted.tolist()


def pregrip_targets(model: mj.MjModel, pregrip_fraction: float) -> np.ndarray:
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    return low + pregrip_fraction * (high - low)


def apply_pregrip(
    model: mj.MjModel, data: mj.MjData, pregrip_fraction: float
) -> None:
    """Start the hand already curled, so it cannot flick the object at t=0."""
    targets = pregrip_targets(model, pregrip_fraction)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        data.qpos[int(model.jnt_qposadr[joint_id])] = targets[actuator_id]
    data.ctrl[:] = targets
    mj.mj_forward(model, data)


def orbit_camera(lookat: np.ndarray, azimuth: float) -> mj.MjvCamera:
    camera = mj.MjvCamera()
    camera.lookat[:] = lookat
    camera.azimuth = azimuth
    camera.elevation = ORBIT_ELEVATION
    camera.distance = ORBIT_DISTANCE
    return camera



def palm_pad_centre(model: mj.MjModel, data: mj.MjData) -> np.ndarray:
    pad_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        for name in PALM_GEOM_NAMES
    ]
    return np.mean([data.geom_xpos[g] for g in pad_ids], axis=0)


def count_held(
    model: mj.MjModel, data: mj.MjData, body_ids: list, palm_center: np.ndarray
) -> int:
    """Objects still in the hand -- the count the tactile signal reflects."""
    return sum(
        1 for body_id in body_ids
        if np.linalg.norm(data.xpos[body_id] - palm_center) <= MAX_FINAL_DISTANCE
        and data.xpos[body_id][2] >= palm_center[2] - 0.05
    )


def support_extent(
    model: mj.MjModel, geom_id: int, rotation: np.ndarray
) -> float:
    """Half-extent of a rotated geom along world up.

    Needed so a randomly oriented object rests on the palm instead of
    interpenetrating it: a tumbled box reaches further down than its half-size.
    """
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    up = rotation[2, :]
    kind = int(model.geom_type[geom_id])
    if kind == mj.mjtGeom.mjGEOM_SPHERE:
        return float(size[0])
    if kind == mj.mjtGeom.mjGEOM_BOX:
        return float(np.abs(up) @ size[:3])
    # Cylinder: radius sweeps the axes perpendicular to its own, half-length
    # projects along it.
    radial = float(size[0] * np.sqrt(max(0.0, 1.0 - up[2] ** 2)))
    return float(abs(size[1] * up[2]) + radial)


def placement_region(model: mj.MjModel, data: mj.MjData) -> tuple:
    """Palm-frame region objects may be placed in: (origin, across, fingerward,
    support height). Derived from the palm pads, not hard-coded."""
    palm_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "palm")
    rotation = data.xmat[palm_id].reshape(3, 3)
    spawn_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        for name in SPAWN_PAD_NAMES
    ]
    centre = np.mean([data.geom_xpos[g] for g in spawn_ids], axis=0)
    support = max(
        float(np.max(_flex_vertices(model, data, f"flex_uspa46_{index}")[:, 2]))
        for index in (1, 2, 3)
    )
    return centre, rotation[:, 0], rotation[:, 2], support


def bounding_radius(model: mj.MjModel, geom_id: int) -> float:
    """Radius of the smallest sphere enclosing the geom, for any orientation.

    `max(geom_size)` is not it: a box's half-edge is sqrt(3) smaller than its
    corner distance, so using it lets two boxes spawn inside one another.
    """
    size = model.geom_size[geom_id]
    kind = int(model.geom_type[geom_id])
    if kind == int(mj.mjtGeom.mjGEOM_SPHERE):
        return float(size[0])
    if kind == int(mj.mjtGeom.mjGEOM_BOX):
        return float(np.linalg.norm(size[:3]))
    if kind in (int(mj.mjtGeom.mjGEOM_CYLINDER),
                int(mj.mjtGeom.mjGEOM_CAPSULE)):
        return float(np.hypot(size[0], size[1]))
    return float(np.linalg.norm(size[:3]))


def place_objects(
    model: mj.MjModel, data: mj.MjData, specs: list, jitter: float,
    height_step: float, randomize_orientation: bool, rng: np.random.Generator,
) -> None:
    """Scatter the objects over the palm region without overlapping them."""
    centre, across, fingerward, support = placement_region(model, data)
    placed = []
    for index in range(len(specs)):
        geom_id = mj.mj_name2id(
            model, mj.mjtObj.mjOBJ_GEOM, object_geom_name(index)
        )
        radius = bounding_radius(model, geom_id)
        if randomize_orientation:
            quat = rng.normal(size=4)
            quat /= np.linalg.norm(quat)
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        rotation = np.zeros(9)
        mj.mju_quat2Mat(rotation, quat)
        lift = support_extent(model, geom_id, rotation.reshape(3, 3))
        # Rejection-sample a slot on the palm that clears everything already
        # placed. Bounding spheres are conservative, so a "clear" slot really
        # is free.
        position = None
        for _ in range(PLACEMENT_ATTEMPTS):
            candidate = (
                centre
                + rng.uniform(-jitter, jitter) * across
                + rng.uniform(-jitter, jitter) * fingerward
            )
            candidate[2] = support + lift + SPAWN_GAP + index * height_step
            if all(
                np.linalg.norm(candidate - other) > radius + other_radius
                for other, other_radius in placed
            ):
                position = candidate
                break
        if position is None:
            # The palm cannot fit another object side by side, so stack this
            # one directly on top of the cluster. Guarantees no overlap (the
            # old code fell through and spawned objects 30 mm inside each
            # other, which launched them at ~1 m/s) and is how a real handful
            # of objects piles up anyway.
            top = max(
                other[2] + other_radius for other, other_radius in placed
            )
            position = centre.copy()
            position[2] = top + lift + SPAWN_GAP
        placed.append((position, radius))
        address = int(model.jnt_qposadr[int(model.body_jntadr[
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, object_body_name(index))
        ])])
        data.qpos[address : address + 7] = np.concatenate([position, quat])
    mj.mj_forward(model, data)



@dataclass(frozen=True)
class GraspProfile:
    pattern: str
    grip_fraction: float
    thumb_grip_fraction: float
    pregrip_fraction: float
    thumb_delay: float
    pulse_hz: float
    pulse_amplitude: float
    shear_amplitude: float
    squeeze_steps: int


def profile_from_args(pattern: str, args: argparse.Namespace) -> GraspProfile:
    return GraspProfile(
        pattern=pattern,
        grip_fraction=args.grip_fraction,
        thumb_grip_fraction=args.thumb_grip_fraction,
        pregrip_fraction=args.pregrip_fraction,
        thumb_delay=args.thumb_delay,
        pulse_hz=args.pulse_hz,
        pulse_amplitude=args.pulse_amplitude,
        shear_amplitude=args.shear_amplitude,
        squeeze_steps=args.squeeze_steps,
    )


def thumb_mask(model: mj.MjModel) -> np.ndarray:
    return np.array([
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i).startswith("th_")
        for i in range(model.nu)
    ])


def lateral_mask(model: mj.MjModel) -> np.ndarray:
    """Abduction / axial-roll actuators, which slide the object sideways."""
    names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    return np.array(["_rot_" in name or "_axl_" in name for name in names])


def control_targets(
    model: mj.MjModel, t: float, profile: GraspProfile
) -> np.ndarray:
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    span = high - low
    is_thumb = thumb_mask(model)
    # The thumb closes less far than the fingers, otherwise the commanded pose
    # drives it into the index and both stall.
    grip = low + np.where(
        is_thumb, profile.thumb_grip_fraction, profile.grip_fraction
    ) * span
    pregrip = pregrip_targets(model, profile.pregrip_fraction)

    close_span = CLOSE_END - CLOSE_START
    finger_phase = np.clip((t - CLOSE_START) / close_span, 0.0, 1.0)
    thumb_phase = np.clip(
        (t - CLOSE_START - profile.thumb_delay) / close_span, 0.0, 1.0
    )
    phase = np.where(is_thumb, thumb_phase, finger_phase)
    smooth = phase * phase * (3.0 - 2.0 * phase)
    targets = pregrip + smooth * (grip - pregrip)
    if t < CLOSE_END + profile.thumb_delay:
        return np.clip(targets, low, high)

    elapsed = t - CLOSE_END - profile.thumb_delay
    if profile.pattern == "hold":
        modulation = 0.0
    elif profile.pattern == "pulse":
        modulation = 0.5 * profile.pulse_amplitude * (
            1.0 - np.cos(2.0 * np.pi * profile.pulse_hz * elapsed)
        )
    elif profile.pattern == "squeeze":
        # Staircase to progressively firmer grips, then back to the start.
        cycle = profile.squeeze_steps * 2
        index = int(elapsed / SQUEEZE_STEP_SECONDS) % cycle
        level = index if index < profile.squeeze_steps else cycle - index - 1
        modulation = profile.pulse_amplitude * level / max(
            profile.squeeze_steps - 1, 1
        )
    elif profile.pattern == "regrasp":
        # Release most of the way back to the pre-grip pose, then re-close.
        release = 0.5 * (1.0 - np.cos(2.0 * np.pi * REGRASP_HZ * elapsed))
        return np.clip(
            targets + release * (pregrip - targets), low, high
        )
    elif profile.pattern == "tap":
        release = 0.5 * (1.0 - np.cos(2.0 * np.pi * TAP_HZ * elapsed))
        return np.clip(
            targets + TAP_DEPTH * release * (pregrip - targets), low, high
        )
    elif profile.pattern == "shear":
        lateral = profile.shear_amplitude * np.sin(
            2.0 * np.pi * SHEAR_HZ * elapsed
        )
        return np.clip(
            targets + np.where(lateral_mask(model), lateral * span, 0.0),
            low, high,
        )
    else:
        raise ValueError(f"Unknown grasp pattern '{profile.pattern}'")
    # Squeeze with the fingers only. Modulating the thumb too would push it
    # past the closure ceiling that keeps it clear of the index, undoing the
    # anti-jam fix exactly when the grip is firmest.
    return np.clip(
        targets + np.where(is_thumb, 0.0, modulation * span), low, high
    )


class TaxelFrames:
    """Fixed local grid of all 368 taxels, plus rigid-body FK onto it.

    Mirrors the sparsh convention (`tactile_ssl/data/xela_tactile.py`):
    positions are forward kinematics of the parent link composed with a grid
    that is constant in that link's frame. Two grids are carried:
    `local_surface` is where contact happens (the membrane rest vertices) and
    is the primary one; `local_magnet` is the raw calibrated magnet centres,
    which sit ~8-10 mm inside the surface but match the real hand's layout.
    """

    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        skins: list,
        tips: list,
        entries_by_patch: dict,
    ):
        self.body_ids = np.zeros(TAXEL_COUNT, dtype=np.int64)
        self.body_names = np.empty(TAXEL_COUNT, dtype=object)
        self.local_surface = np.zeros((TAXEL_COUNT, 3))
        self.local_magnet = np.zeros((TAXEL_COUNT, 3))
        self.local_normal = np.zeros((TAXEL_COUNT, 3))
        # Per-taxel frame in its link: columns are the shear-x, shear-y and
        # outward-normal axes the readings are expressed in.
        self.local_rotation = np.zeros((TAXEL_COUNT, 3, 3))
        self.palm_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "palm")

        patches = [
            (skin.parent_body_name, skin.flex_name, skin.taxel_ids,
             GEOM_TO_PATCH[skin.definition.geom_name],
             np.tile(
                 np.column_stack(
                     (skin.tangent_u, skin.tangent_v, skin.normal)
                 ),
                 (len(skin.taxel_ids), 1, 1),
             ))
            for skin in skins
        ] + [
            (tip.parent_body_name, tip.flex_name, tip.taxel_ids,
             f"tip_{tip.finger}", tip.vertex_rotations)
            for tip in tips
        ]
        for body_name, flex_name, taxel_ids, patch, rotations in patches:
            body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
            rotation = data.xmat[body_id].reshape(3, 3)
            origin = data.xpos[body_id]
            # Membrane rest vertices pulled back into the link frame: the
            # anchors hold every vertex here, so this is a fixed local grid.
            vertices = _flex_vertices(model, data, flex_name)
            self.local_surface[taxel_ids] = (vertices - origin) @ rotation
            self.local_rotation[taxel_ids] = rotations
            self.local_normal[taxel_ids] = rotations[:, :, 2]
            self.body_ids[taxel_ids] = body_id
            self.body_names[taxel_ids] = body_name
            for entry in entries_by_patch[patch]:
                self.local_magnet[entry.taxel_id] = entry.pos

    def world(self, data: mj.MjData, local: np.ndarray) -> np.ndarray:
        rotations = data.xmat[self.body_ids].reshape(-1, 3, 3)
        return data.xpos[self.body_ids] + np.einsum(
            "nij,nj->ni", rotations, local
        )

    def palm(self, data: mj.MjData, local: np.ndarray) -> np.ndarray:
        palm_rotation = data.xmat[self.palm_id].reshape(3, 3)
        return (self.world(data, local) - data.xpos[self.palm_id]) @ palm_rotation

    def world_vectors(self, data: mj.MjData, readings: np.ndarray) -> np.ndarray:
        """Per-taxel (shear_x, shear_y, normal_z) rotated into world.

        The normal channel is compression-positive, so it maps onto the
        outward normal: a pressed taxel yields an arrow standing off the
        surface, tilted by the shear.
        """
        rotations = data.xmat[self.body_ids].reshape(-1, 3, 3)
        axes = np.einsum("nij,njk->nik", rotations, self.local_rotation)
        return np.einsum("nik,nk->ni", axes, readings)

    def palm_directions(self, data: mj.MjData, local: np.ndarray) -> np.ndarray:
        palm_rotation = data.xmat[self.palm_id].reshape(3, 3)
        rotations = data.xmat[self.body_ids].reshape(-1, 3, 3)
        return np.einsum("nij,nj->ni", rotations, local) @ palm_rotation


def sample_specs(
    shape: str, scale: float, count: int, args: argparse.Namespace,
    rng: np.random.Generator,
) -> list:
    """Object specs for one episode.

    A single object honours the --objects/--object-scale sweep exactly.
    Multi-object episodes draw from the independent --multi-shapes and
    --multi-scale-range pools, so shape and size can each be fixed (one shape /
    equal range bounds) or varied without touching the other.
    """
    if count == 1:
        return [ObjectSpec(shape, scale, object_mass(scale, args.object_mass))]
    low, high = args.multi_scale_range
    specs = []
    for _ in range(count):
        item_shape = str(rng.choice(args.multi_shapes))
        item_scale = float(rng.uniform(low, high))
        specs.append(ObjectSpec(
            item_shape, item_scale, object_mass(item_scale, args.object_mass)
        ))
    return specs


def object_mass(scale: float, override: float) -> float:
    """Density-preserving mass unless an absolute mass is pinned."""
    if override > 0.0:
        return override
    return OBJECT_MASS * scale ** 3


def run_episode(
    specs: list, pattern: str, args: argparse.Namespace, output_dir: Path,
    rng: np.random.Generator,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = profile_from_args(pattern, args)
    object_count = len(specs)
    geom_names = [object_geom_name(i) for i in range(object_count)]
    model, skins, tips = build_model(
        specs, args.palm_pitch, args.object_adhesion, args.object_adhesion_gap
    )
    object_geom_ids = np.array(
        [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name) for name in geom_names]
    )
    timestep = model.opt.timestep
    sensor_entries = []
    for skin in skins:
        sensor_entries.append((
            skin.definition.geom_name,
            skin.taxel_ids,
            FlexTaxelSensor(
                model,
                skin.flex_name,
                skin.parent_body_name,
                pad_vertex_rotations(skin),
                geom_names,
            ),
        ))
    for tip in tips:
        sensor_entries.append((
            f"{tip.finger}_tip",
            tip.taxel_ids,
            FlexTaxelSensor(
                model,
                tip.flex_name,
                tip.parent_body_name,
                tip.vertex_rotations,
                geom_names,
            ),
        ))
    layout = build_hand_layout(model, mj.MjData(model), skins, tips)

    data = mj.MjData(model)
    # Curl first, then place: the pads move as the fingers curl, and a hand that
    # snapped into the pre-grip after placement would flick the object away.
    apply_pregrip(model, data, args.pregrip_fraction)
    # Capture the fixed local grid while the hand is at rest and untouched.
    frames_ref = TaxelFrames(
        model, data, skins, tips, layout_entries_by_patch()
    )
    palm_center = palm_pad_centre(model, data)
    place_objects(
        model, data, specs, args.placement_jitter, args.placement_height,
        args.orientation_random, rng,
    )
    # Let the cluster settle and stack before the grip starts.
    for _ in range(int(np.ceil(args.settle_duration / timestep))):
        data.ctrl[:] = control_targets(model, 0.0, profile)
        mj.mj_step(model, data)
    object_addresses = [
        int(model.jnt_qposadr[int(model.body_jntadr[
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, object_body_name(i))
        ])]) for i in range(object_count)
    ]
    object_body_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, object_body_name(i))
        for i in range(object_count)
    ]
    settled_held = count_held(model, data, object_body_ids, palm_center)

    hand_joint_ids = model.actuator_trnid[:, 0]
    hand_qpos_index = np.array(
        [int(model.jnt_qposadr[j]) for j in hand_joint_ids]
    )
    taxel_counts = [len(ids) for _, ids, _ in sensor_entries]
    total_taxels = TAXEL_COUNT

    def display_grids(flats: dict) -> dict:
        grids = {}
        for skin in skins:
            count_u, count_v = skin.definition.count[:2]
            grids[skin.definition.geom_name] = flats[
                skin.definition.geom_name
            ].reshape(count_u, count_v, 3)
        for tip in tips:
            grids[f"{tip.finger}_tip"] = tip_canvas(
                tip, flats[f"{tip.finger}_tip"]
            )
        return grids
    sample_count = int(args.duration * args.sample_hz)
    sample_every = max(1, round(1.0 / (args.sample_hz * timestep)))
    frame_every = max(1, round(1.0 / (VIDEO_FPS_TARGET * timestep)))

    tactile = np.zeros((sample_count, total_taxels, 3))
    taxel_positions_log = np.zeros((sample_count, total_taxels, 3), np.float32)
    taxel_magnet_log = np.zeros((sample_count, total_taxels, 3), np.float32)
    taxel_normal_log = np.zeros((sample_count, total_taxels, 3), np.float32)
    tactile_times = np.zeros(sample_count)
    hand_qpos = np.zeros((sample_count, model.nu))
    ctrl_log = np.zeros((sample_count, model.nu))
    object_poses = np.zeros((sample_count, object_count, 7))
    object_vels = np.zeros((sample_count, object_count, 6))
    contact_counts = np.zeros(sample_count, dtype=np.int64)
    total_normal = np.zeros(sample_count)

    settled_index = max(0, int(CLOSE_START * args.sample_hz) - 1)
    closed_index = int(CLOSE_END * args.sample_hz) - 1
    snapshot_grids = {"settled": None, "closed": None, "peak": None}
    peak_normal = -1.0
    latest_flat = np.zeros((TAXEL_COUNT, 3))

    # Rendering dominates episode cost (~24 s for a 5 s episode). Dataset
    # collection retries rejected episodes, so it runs with --no-video and pays
    # the render cost only for the samples it keeps.
    renderer = (
        mj.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
        if args.video else None
    )
    scene_option = mj.MjvOption()
    scene_option.geomgroup[:] = 1
    cloud_renderer = (
        mj.Renderer(model, height=VIDEO_HEIGHT, width=CLOUD_WIDTH)
        if args.video and args.cloud_view else None
    )
    cloud_option = bare_scene_option()
    # One grid drives the framing, the spheres and the arrows, so they cannot
    # disagree. Both grids are logged to the npz regardless of this choice.
    cloud_local = (
        frames_ref.local_surface if args.cloud_grid == "surface"
        else frames_ref.local_magnet
    )
    rest_cloud = frames_ref.world(data, cloud_local)
    cloud_lookat = rest_cloud.mean(axis=0)
    cloud_distance = cloud_camera_distance(
        model,
        float(np.max(np.linalg.norm(rest_cloud - cloud_lookat, axis=1))),
    )
    orbit_lookat = palm_center + np.array([0.0, 0.0, ORBIT_LOOKAT_LIFT])
    panel = TaxelPanel(
        layout, panel_taxel_ids(skins, tips), PANEL_SIZE, PANEL_SIZE,
        TAXEL_COUNT,
    )
    marker_sensors = (
        [(skin.flex_name, skin.taxel_ids) for skin in skins]
        + [(tip.flex_name, tip.taxel_ids) for tip in tips]
    )
    dim_flex_skins(model)
    force_scale = args.force_scale
    frames = []

    total_steps = int(np.ceil(args.duration / timestep))
    sample_index = 0
    sensed_log = np.zeros((sample_count, object_count), dtype=bool)
    abort_reason = None
    t = 0.0
    for step in range(total_steps):
        t = (step + 1) * timestep
        data.ctrl[:] = control_targets(model, t, profile)
        mj.mj_step(model, data)
        # Bail out of an episode that can no longer be accepted (see
        # ABORT_CHECK_TIME). Both tests are necessary conditions for acceptance:
        # a divergence fails outright, and an object outside the hand can be
        # neither held nor sensed.
        if args.abort_check_time > 0.0 and step % sample_every == 0:
            if data.warning[mj.mjtWarning.mjWARN_BADQACC].number > 0:
                abort_reason = "divergence"
            elif t >= args.abort_check_time and count_held(
                model, data, object_body_ids, palm_center
            ) < object_count:
                abort_reason = "object_lost"
            if abort_reason is not None:
                break
        if step % sample_every == 0 and sample_index < sample_count:
            if args.object_adhesion > 0.0:
                assert_adhesion_scoped(data, object_geom_ids)
            sensed_log[sample_index] = sensed_objects(data, object_geom_ids)
            flats = {}
            contact_total = 0
            flat = np.zeros((TAXEL_COUNT, 3))
            for name, ids, sensor in sensor_entries:
                flat_read, contacts = sensor.read(data)
                flats[name] = flat_read
                # Scatter into hardware taxel-ID order (0..367), so the array
                # is index-compatible with real XELA streams.
                flat[ids] = flat_read
                contact_total += contacts
            tactile[sample_index] = flat
            taxel_positions_log[sample_index] = frames_ref.palm(
                data, frames_ref.local_surface
            )
            taxel_magnet_log[sample_index] = frames_ref.palm(
                data, frames_ref.local_magnet
            )
            taxel_normal_log[sample_index] = frames_ref.palm_directions(
                data, frames_ref.local_normal
            )
            tactile_times[sample_index] = t
            hand_qpos[sample_index] = data.qpos[hand_qpos_index]
            ctrl_log[sample_index] = data.ctrl
            for oi, addr in enumerate(object_addresses):
                object_poses[sample_index, oi] = data.qpos[addr : addr + 7]
                dof = int(model.jnt_dofadr[int(model.body_jntadr[
                    object_body_ids[oi]
                ])])
                object_vels[sample_index, oi] = data.qvel[dof : dof + 6]
            contact_counts[sample_index] = contact_total
            total_normal[sample_index] = float(np.sum(flat[:, 2]))
            latest_flat = flat
            if sample_index == settled_index:
                snapshot_grids["settled"] = display_grids(flats)
            if sample_index == closed_index:
                snapshot_grids["closed"] = display_grids(flats)
            if t >= CLOSE_END and total_normal[sample_index] > peak_normal:
                peak_normal = total_normal[sample_index]
                snapshot_grids["peak"] = display_grids(flats)
            sample_index += 1
        if renderer is not None and step % frame_every == 0:
            azimuth = ORBIT_START_AZIMUTH + 360.0 * args.orbit_revolutions * (
                t / args.duration
            )
            renderer.update_scene(
                data,
                camera=orbit_camera(orbit_lookat, azimuth),
                scene_option=scene_option,
            )
            values = channel_values(latest_flat, args.heatmap_channel)
            force_scale = max(force_scale, float(np.max(np.abs(values))))
            # The rendered cloud is exactly the one logged to the dataset: FK
            # of the fixed local grid, in world coordinates for the renderer.
            draw_taxel_markers(
                renderer.scene,
                frames_ref.world(data, frames_ref.local_surface),
                marker_colours(values, args.heatmap_channel, force_scale),
                MARKER_RADIUS,
            )
            frame = paste_panel(
                renderer.render(),
                panel.render(values, args.heatmap_channel, force_scale),
                PANEL_MARGIN_PX,
            )
            if cloud_renderer is not None:
                cloud_positions = frames_ref.world(data, cloud_local)
                # Track the live cloud so it keeps filling the panel as the
                # hand closes; smoothed so the view does not jitter.
                centre = cloud_positions.mean(axis=0)
                radius = float(np.max(
                    np.linalg.norm(cloud_positions - centre, axis=1)
                ))
                cloud_lookat += CLOUD_TRACK_ALPHA * (centre - cloud_lookat)
                cloud_distance += CLOUD_TRACK_ALPHA * (
                    cloud_camera_distance(model, radius) - cloud_distance
                )
                cloud_camera = orbit_camera(cloud_lookat, azimuth)
                cloud_camera.distance = cloud_distance
                cloud_renderer.update_scene(
                    data, camera=cloud_camera, scene_option=cloud_option
                )
                cloud_colours = marker_colours(
                    values, args.heatmap_channel, force_scale
                )
                draw_taxel_markers(
                    cloud_renderer.scene, cloud_positions, cloud_colours,
                    CLOUD_MARKER_RADIUS,
                )
                draw_taxel_arrows(
                    cloud_renderer.scene,
                    cloud_positions,
                    frames_ref.world_vectors(data, latest_flat),
                    cloud_colours,
                    args.arrow_scale * force_scale,
                )
                frame = np.concatenate(
                    [frame, cloud_renderer.render()], axis=1
                )
            frames.append(frame)
    if renderer is not None:
        renderer.close()
    if cloud_renderer is not None:
        cloud_renderer.close()

    if abort_reason is not None:
        # No npz, metadata or heatmaps: the episode is rejected, and writing
        # them would only cost I/O for something the caller throws away.
        objects_held = count_held(model, data, object_body_ids, palm_center)
        print(
            f"ABORT {output_dir.name}: {abort_reason} at t={t:.2f}s of "
            f"{args.duration:g}s, held={objects_held}/{object_count}"
        )
        return {
            "passed": False,
            "aborted": True,
            "abort_reason": abort_reason,
            "abort_time_s": float(t),
            "divergences": int(
                data.warning[mj.mjtWarning.mjWARN_BADQACC].number
            ),
            "object_held": bool(objects_held > 0),
            "object_count": object_count,
            "objects_held": objects_held,
            "objects_sensed": 0,
            "peak_total_normal_n": float(max(peak_normal, 0.0)),
            "contact_fraction": 0.0,
            "mujoco_version": mj.__version__,
        }

    for tag, grids in snapshot_grids.items():
        if grids is not None:
            save_hand_heatmap(
                layout, grids, output_dir / f"hand_heatmap_{tag}.png"
            )
    if renderer is not None:
        mediapy.write_video(
            output_dir / "grasp.mp4", frames, fps=1.0 / (frame_every * timestep)
        )

    pulse_mask = tactile_times >= CLOSE_END
    contact_fraction = float(
        np.mean(contact_counts[pulse_mask] > 0)
    )
    final_distance = float(
        np.min(np.linalg.norm(object_poses[-1, :, :3] - palm_center, axis=1))
    )
    divergences = int(data.warning[mj.mjtWarning.mjWARN_BADQACC].number)
    objects_held = count_held(model, data, object_body_ids, palm_center)
    # Scored over the same post-close window as contact_fraction, and by
    # majority rather than "ever": contacts flicker, and an object that brushed
    # a pad early and then fell out is not something the taxels can be asked to
    # count. Measured on the 3-sphere baseline, "ever" gave 2 while only 1
    # object was still in the hand at the end.
    object_sensed_fraction = sensed_log[pulse_mask].mean(axis=0)
    objects_sensed = int(np.sum(object_sensed_fraction >= SENSED_FRACTION))
    held = objects_held > 0
    # Small clusters read weakly on purpose (0.04-0.3 N for 1-5 small objects
    # is expected and accepted), so multi-object episodes are not failed for a
    # faint signal -- dropping them would bias the dataset toward heavy grasps.
    # But an episode with *no* sensed contact is not weak, it is empty: it
    # carries a count label with zero tactile evidence for it, which is
    # actively wrong training data for both the shape and quantity tasks.
    signal_ok = (
        peak_normal >= MIN_PEAK_TOTAL_NORMAL
        and contact_fraction >= PATTERN_MIN_CONTACT[pattern]
    ) if object_count == 1 else (
        peak_normal >= MULTI_MIN_PEAK_NORMAL and contact_fraction > 0.0
    )
    passed = divergences == 0 and held and signal_ok

    palm_quat = np.zeros(4)
    mj.mju_mat2Quat(palm_quat, data.xmat[frames_ref.palm_id])
    np.savez_compressed(
        output_dir / "sensor_log.npz",
        tactile=tactile,
        # Rigid-body FK of the fixed local grid, palm frame, taxel-id order.
        taxel_positions=taxel_positions_log,
        taxel_positions_magnet=taxel_magnet_log,
        taxel_normals=taxel_normal_log,
        taxel_local_surface=frames_ref.local_surface.astype(np.float32),
        taxel_local_magnet=frames_ref.local_magnet.astype(np.float32),
        taxel_local_normal=frames_ref.local_normal.astype(np.float32),
        taxel_body_names=frames_ref.body_names.astype(str),
        palm_world_pos=data.xpos[frames_ref.palm_id].astype(np.float32),
        palm_world_quat=palm_quat.astype(np.float32),
        feature_layout=np.array(
            "sparsh 6-channel input = concatenate([tactile, taxel_positions], "
            "axis=-1) -> (T, 368, 6) as [shear_x, shear_y, normal_z, x, y, z]"
        ),
        position_frame=np.array("palm"),
        shape_label=np.array(specs[0].shape),
        grasp_pattern=np.array(pattern),
        shape_index=np.array(sorted(OBJECT_SIZES).index(specs[0].shape)),
        # Quantity-task labels, weakest to strongest evidence:
        #   object_count   - spawned; includes objects that fell out entirely
        #   objects_held   - still near the palm, but KINEMATIC only: an object
        #                    lying in the unsensed pocket counts here while
        #                    contributing nothing to the tactile signal
        #   objects_sensed - touched a taxel membrane at some point in the
        #                    episode; the only one the tactile data supports
        # Train quantity on objects_sensed. The others overcount by exactly the
        # objects the taxels never felt.
        object_count=np.array(object_count),
        objects_held=np.array(objects_held),
        objects_sensed=np.array(objects_sensed),
        object_sensed_fraction=object_sensed_fraction.astype(np.float32),
        object_sensed_log=sensed_log,
        mujoco_version=np.asarray(mj.__version__),
        objects_held_after_settle=np.array(settled_held),
        object_shapes=np.array([o.shape for o in specs]),
        object_shape_indices=np.array(
            [sorted(OBJECT_SIZES).index(o.shape) for o in specs]
        ),
        object_scales=np.array([o.scale for o in specs], dtype=np.float32),
        object_sizes=np.array(
            [np.asarray(OBJECT_SIZES[o.shape]) * o.scale for o in specs],
            dtype=np.float32,
        ),
        object_masses=np.array([o.mass for o in specs], dtype=np.float32),
        object_scale=np.array(specs[0].scale, dtype=np.float32),
        object_size=(
            np.asarray(OBJECT_SIZES[specs[0].shape], dtype=np.float32)
            * specs[0].scale
        ),
        object_mass_kg=np.array(specs[0].mass, dtype=np.float32),
        tactile_times=tactile_times,
        skin_names=np.array([name for name, _, _ in sensor_entries]),
        skin_taxel_ids=np.concatenate([ids for _, ids, _ in sensor_entries]),
        skin_taxel_counts=np.array(taxel_counts),
        tip_grid_rowcol=np.array(TIP_GRID_ROWCOL),
        id_grid=build_layout(HAND_WORKSPACE).id_grid,
        hand_qpos=hand_qpos,
        ctrl=ctrl_log,
        object_poses=object_poses,
        object_vels=object_vels,
        object_pose=object_poses[:, 0],
        object_vel=object_vels[:, 0],
        contact_count=contact_counts,
        total_normal=total_normal,
    )
    results = {
        "passed": bool(passed),
        "aborted": False,
        "divergences": divergences,
        "object_held": bool(held),
        "object_count": object_count,
        "objects_held": objects_held,
        "objects_sensed": objects_sensed,
        "object_sensed_fraction": object_sensed_fraction.round(3).tolist(),
        "objects_held_after_settle": settled_held,
        "object_final_distance_m": final_distance,
        "peak_total_normal_n": float(peak_normal),
        "contact_fraction": contact_fraction,
        "object_adhesion_n": float(args.object_adhesion),
        "object_adhesion_gap_m": float(args.object_adhesion_gap),
        # MuJoCo 3.11 moved flex elasticity into the CG solver, so tactile
        # forces are not comparable across versions -- measured, the same
        # seeded episode shifts by a few percent. Recorded so a dataset can
        # never be silently mixed across a version boundary.
        "mujoco_version": mj.__version__,
    }
    metadata = {
        "shape": specs[0].shape,
        "object_scale": specs[0].scale,
        "object_count": object_count,
        "objects": [
            {"shape": o.shape, "scale": o.scale, "mass_kg": o.mass,
             "size": (np.asarray(OBJECT_SIZES[o.shape]) * o.scale).tolist()}
            for o in specs
        ],
        "placement_jitter": args.placement_jitter,
        "placement_height": args.placement_height,
        "settle_duration": args.settle_duration,
        "orientation_random": args.orientation_random,
        "seed": args.seed,
        "object_size": (
            np.asarray(OBJECT_SIZES[specs[0].shape], dtype=np.float64)
            * specs[0].scale
        ).tolist(),
        "object_mass_kg": specs[0].mass,
        "phases": {
            "close_start_s": CLOSE_START,
            "close_end_s": CLOSE_END,
            "duration_s": args.duration,
        },
        "grasp_pattern": pattern,
        "grip_fraction": args.grip_fraction,
        "thumb_grip_fraction": args.thumb_grip_fraction,
        "thumb_delay": args.thumb_delay,
        "pregrip_fraction": args.pregrip_fraction,
        "palm_pitch_deg": args.palm_pitch,
        "pulse_hz": args.pulse_hz,
        "pulse_amplitude": args.pulse_amplitude,
        "shear_amplitude": args.shear_amplitude,
        "squeeze_steps": args.squeeze_steps,
        "orbit_revolutions": args.orbit_revolutions,
        "heatmap_channel": args.heatmap_channel,
        "force_scale_n": force_scale,
        "cloud_view": args.cloud_view,
        "cloud_grid": args.cloud_grid,
        "arrow_scale": args.arrow_scale,
        "sample_hz": args.sample_hz,
        "tactile_shape": list(tactile.shape),
        "taxel_positions_shape": list(taxel_positions_log.shape),
        "position_frame": "palm",
        "position_convention": (
            "rigid-body FK of a fixed local grid; taxel_positions are the "
            "sensing surface (contact happens here), taxel_positions_magnet "
            "are the calibrated magnet centres ~8-10 mm inside it"
        ),
        "results": results,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    status = "PASS" if passed else "FAIL"
    label = output_dir.name
    print(
        f"{status} {label}: held={objects_held}/{object_count}, "
        f"sensed={objects_sensed}/{object_count}, "
        f"peak normal={peak_normal:.3f} N, "
        f"contact fraction={contact_fraction:.2f}, "
        f"divergences={divergences}"
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grasp an object and record the flex taxel sensors."
    )
    parser.add_argument(
        "--objects", nargs="+", choices=sorted(OBJECT_SIZES),
        default=["sphere", "box", "cylinder"],
    )
    parser.add_argument(
        "--object-scale", nargs="+", type=float, default=[1.0],
        help="object size multipliers, e.g. 0.5 1.0 1.5 (mass follows scale^3)",
    )
    parser.add_argument(
        "--object-mass", type=float, default=0.0,
        help="pin an absolute object mass in kg (0 = density-preserving)",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sample-hz", type=float, default=100.0)
    parser.add_argument("--grip-fraction", type=float, default=0.55)
    parser.add_argument(
        "--thumb-grip-fraction", type=float, default=THUMB_GRIP_FRACTION,
        help="thumb closure target; lower than the fingers so the commanded "
             "pose does not drive the thumb into the index",
    )
    parser.add_argument(
        "--thumb-delay", type=float, default=THUMB_DELAY,
        help="seconds the thumb ramp lags the fingers",
    )
    parser.add_argument(
        "--grasp-pattern", nargs="+", choices=GRASP_PATTERNS, default=["pulse"],
        help="motion pattern(s) after closing; runs the cross-product with "
             "--objects and --object-scale",
    )
    parser.add_argument(
        "--multi-count", nargs="+", type=int, default=[1],
        help="number(s) of objects per episode; cross-producted with the "
             "other sweeps. Counts above 1 use the --multi-* pools below",
    )
    parser.add_argument(
        "--multi-shapes", nargs="+", choices=sorted(OBJECT_SIZES),
        default=list(MULTI_SHAPES),
        help="shape pool for multi-object episodes; pass one to fix the shape",
    )
    parser.add_argument(
        "--multi-scale-range", nargs=2, type=float, default=list(MULTI_SCALE_RANGE),
        metavar=("MIN", "MAX"),
        help="scale range for multi-object episodes; equal values fix the size",
    )
    parser.add_argument(
        "--object-adhesion", type=float, default=0.0,
        help="adhesive force in N between OBJECTS ONLY, to hold a multi-object "
             "cluster together (0 = off). Never applied to object-hand or "
             "object-skin contacts, so the taxel signal is unaffected. "
             "Needs MuJoCo >= 3.11",
    )
    parser.add_argument(
        "--object-adhesion-gap", type=float, default=OBJECT_ADHESION_GAP,
        help="distance in metres over which object adhesion keeps acting. "
             "Must be > 0: at gap=0 the contact dies the instant the objects "
             "separate and adhesion does nothing at any strength",
    )
    parser.add_argument(
        "--placement-jitter", type=float, default=PLACEMENT_JITTER,
        help="half-width of the palm-plane placement region in metres",
    )
    parser.add_argument(
        "--placement-height", type=float, default=PLACEMENT_HEIGHT,
        help="vertical stagger per object; objects settle into a cluster",
    )
    parser.add_argument(
        "--settle-duration", type=float, default=SETTLE_DURATION,
        help="seconds the cluster settles before the grip closes",
    )
    parser.add_argument(
        "--orientation-random", action=argparse.BooleanOptionalAction,
        default=True, help="place objects at uniform random orientation",
    )
    parser.add_argument(
        "--abort-check-time", type=float, default=ABORT_CHECK_TIME,
        help="seconds after which an episode that has already lost an object "
             "is abandoned instead of simulated to the end; 0 disables",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shear-amplitude", type=float, default=0.15)
    parser.add_argument("--squeeze-steps", type=int, default=4)
    parser.add_argument(
        "--pregrip-fraction", type=float, default=PREGRIP_FRACTION,
        help="how far the fingers are curled before the object is placed",
    )
    parser.add_argument(
        "--palm-pitch", type=float, default=PALM_PITCH_DEG,
        help="palm pitch in degrees, negative tips the fingers down",
    )
    parser.add_argument("--pulse-hz", type=float, default=1.0)
    parser.add_argument("--pulse-amplitude", type=float, default=0.15)
    parser.add_argument(
        "--orbit-revolutions", type=float, default=1.0,
        help="camera revolutions around the hand per episode (0 = fixed view)",
    )
    parser.add_argument(
        "--heatmap-channel", choices=CHANNELS, default="normal",
        help="taxel channel shown by the live heatmap in the video",
    )
    parser.add_argument(
        "--force-scale", type=float, default=0.5,
        help="heatmap full-scale force in N (auto-expands if exceeded)",
    )
    parser.add_argument(
        "--video", action=argparse.BooleanOptionalAction, default=True,
        help="render grasp.mp4; --no-video keeps the npz, heatmaps and "
             "metadata but skips rendering, which dominates episode cost",
    )
    parser.add_argument(
        "--cloud-view", action=argparse.BooleanOptionalAction, default=True,
        help="render the bare taxel point cloud beside the hand view",
    )
    parser.add_argument(
        "--cloud-grid", choices=CLOUD_GRIDS, default="surface",
        help="which fixed local grid the point cloud draws: the sensing "
             "surface, or the calibrated magnet centres ~8 mm inside it",
    )
    parser.add_argument(
        "--arrow-scale", type=float, default=1.0,
        help="force at which a cloud arrow reaches full length, "
             "as a multiple of the heatmap force scale",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=_MODEL_DIR.parent / "data" / "leapxela_flex_grasp_test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = []
    runs = 0
    for shape in args.objects:
        for scale in args.object_scale:
            for count in args.multi_count:
                for pattern in args.grasp_pattern:
                    # Each episode gets its own stream so a run is reproducible
                    # from --seed regardless of the sweep order.
                    # zlib.crc32, not hash(): Python salts string hashes per
                    # process, so hash() would make --seed non-reproducible.
                    rng = np.random.default_rng(
                        [args.seed, zlib.crc32(shape.encode()),
                         int(scale * 1000), count,
                         GRASP_PATTERNS.index(pattern)]
                    )
                    specs = sample_specs(shape, scale, count, args, rng)
                    # Scale 1, count 1 and the default pattern keep the
                    # original directory name so existing paths stay valid.
                    name = shape if scale == 1.0 else f"{shape}_{scale:g}x"
                    if count != 1:
                        name = f"{name}_x{count}"
                    if pattern != "pulse":
                        name = f"{name}_{pattern}"
                    results = run_episode(
                        specs, pattern, args, args.output_root / name, rng
                    )
                    runs += 1
                    if not results["passed"]:
                        failures.append(name)
    if failures:
        raise SystemExit(f"Grasp test failed for: {', '.join(failures)}")
    print(f"All {runs} grasp runs passed. Logs: {args.output_root}")


if __name__ == "__main__":
    main()
