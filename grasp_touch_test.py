"""Grasp test for the flex taxel sensors.

Places an object on the palm-up hand, closes the fingers, then pulse-squeezes
while recording all 368 taxels in hardware taxel-ID order. Saves an npz sensor
log, 3-channel hand-heatmap snapshots, and an orbiting MP4 render with the live
heatmap drawn on the hand and as a palm-view panel.
"""

import argparse
import json
import sys
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
    MARKER_RADIUS,
    TaxelPanel,
    channel_values,
    dim_flex_skins,
    draw_taxel_markers,
    marker_colours,
    panel_taxel_ids,
    paste_panel,
    taxel_positions,
)


OBJECT_BODY_NAME = "grasp_object"
OBJECT_GEOM_NAME = "grasp_object_geom"
OBJECT_MASS = 0.08
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
CYLINDER_FINGERWARD_OFFSET = 0.025
PALM_GEOM_NAMES = ("uspa46_1", "uspa46_2", "uspa46_3")
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
MIN_PEAK_TOTAL_NORMAL = 0.1
MIN_CONTACT_FRACTION = 0.5
MAX_FINAL_DISTANCE = 0.15


def build_model(shape: str, scale: float, mass: float):
    entries_by_patch = {}
    for entry in build_layout(HAND_WORKSPACE).entries:
        entries_by_patch.setdefault(entry.patch, []).append(entry)
    spec = load_base_model("Box")
    trim_pad_boxes(spec)
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
    body = spec.worldbody.add_body(name=OBJECT_BODY_NAME, pos=[0.0, 0.0, 0.5])
    body.add_freejoint()
    body.add_geom(
        name=OBJECT_GEOM_NAME,
        type=OBJECT_GEOM_TYPES[shape],
        size=(np.asarray(OBJECT_SIZES[shape], dtype=np.float64) * scale).tolist(),
        mass=mass,
        rgba=[0.15, 0.35, 0.9, 1.0],
        contype=3,
        conaffinity=3,
        condim=3,
        friction=[0.8, 0.005, 0.0001],
    )
    model = spec.compile()
    model.vis.global_.offwidth = VIDEO_WIDTH
    model.vis.global_.offheight = VIDEO_HEIGHT
    model.vis.headlight.ambient = [0.3, 0.3, 0.3]
    model.vis.headlight.diffuse = [0.7, 0.7, 0.7]
    model.vis.headlight.specular = [0.8, 0.8, 0.8]
    return model, skins, tips


def orbit_camera(lookat: np.ndarray, azimuth: float) -> mj.MjvCamera:
    camera = mj.MjvCamera()
    camera.lookat[:] = lookat
    camera.azimuth = azimuth
    camera.elevation = ORBIT_ELEVATION
    camera.distance = ORBIT_DISTANCE
    return camera


def object_qpos_address(model: mj.MjModel) -> int:
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
    joint_id = int(model.body_jntadr[body_id])
    return int(model.jnt_qposadr[joint_id])


def spawn_pose(model: mj.MjModel, shape: str) -> np.ndarray:
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    pad_ids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
        for name in PALM_GEOM_NAMES
    ]
    palm_center = np.mean([data.geom_xpos[g] for g in pad_ids], axis=0)
    # Rest the object on the palm membranes rather than dropping it: a drop of
    # even 1 mm bounces the sphere off a pad edge and out of the hand.
    support_height = max(
        float(np.max(_flex_vertices(model, data, f"flex_uspa46_{index}")[:, 2]))
        for index in (1, 2, 3)
    )
    # Read the half-extent off the compiled geom rather than OBJECT_SIZES, so a
    # scaled object can never disagree with its resting height. Index 0 is the
    # vertical half-extent at rest for all three shapes: sphere radius, box
    # half-size, and cylinder radius (it lies on its side).
    object_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, OBJECT_GEOM_NAME)
    half_height = float(model.geom_size[object_geom_id][0])
    position = np.array([
        palm_center[0],
        palm_center[1],
        support_height + half_height + SPAWN_GAP,
    ])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    if shape == "cylinder":
        palm_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "palm")
        # Spawn toward the fingertips: the cylinder rolls wrist-ward off the
        # pads otherwise.
        fingerward = data.xmat[palm_id].reshape(3, 3)[:, 2]
        position = position + CYLINDER_FINGERWARD_OFFSET * fingerward
        axis = data.xmat[palm_id].reshape(3, 3)[:, 0]
        z_new = axis / np.linalg.norm(axis)
        x_new = np.cross([0.0, 0.0, 1.0], z_new)
        x_new /= np.linalg.norm(x_new)
        y_new = np.cross(z_new, x_new)
        mj.mju_mat2Quat(quat, np.column_stack((x_new, y_new, z_new)).ravel())
    return np.concatenate([position, quat]), palm_center


def control_targets(
    model: mj.MjModel, t: float, grip_fraction: float,
    pulse_hz: float, pulse_amplitude: float,
) -> np.ndarray:
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    grip = low + grip_fraction * (high - low)
    if t < CLOSE_START:
        return np.zeros(model.nu)
    if t < CLOSE_END:
        phase = (t - CLOSE_START) / (CLOSE_END - CLOSE_START)
        smooth = phase * phase * (3.0 - 2.0 * phase)
        return smooth * grip
    pulse = 0.5 * pulse_amplitude * (
        1.0 - np.cos(2.0 * np.pi * pulse_hz * (t - CLOSE_END))
    )
    return np.clip(grip + pulse * (high - low), low, high)


def object_mass(scale: float, override: float) -> float:
    """Density-preserving mass unless an absolute mass is pinned."""
    if override > 0.0:
        return override
    return OBJECT_MASS * scale ** 3


def run_episode(
    shape: str, scale: float, args: argparse.Namespace, output_dir: Path
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mass = object_mass(scale, args.object_mass)
    model, skins, tips = build_model(shape, scale, mass)
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
                [OBJECT_GEOM_NAME],
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
                [OBJECT_GEOM_NAME],
            ),
        ))
    layout = build_hand_layout(model, mj.MjData(model), skins, tips)
    pose, palm_center = spawn_pose(model, shape)

    data = mj.MjData(model)
    mj.mj_forward(model, data)
    qpos_address = object_qpos_address(model)
    data.qpos[qpos_address : qpos_address + 7] = pose
    mj.mj_forward(model, data)

    hand_joint_ids = model.actuator_trnid[:, 0]
    hand_qpos_index = np.array(
        [int(model.jnt_qposadr[j]) for j in hand_joint_ids]
    )
    object_dof = int(model.jnt_dofadr[int(model.body_jntadr[
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
    ])])

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
    tactile_times = np.zeros(sample_count)
    hand_qpos = np.zeros((sample_count, model.nu))
    ctrl_log = np.zeros((sample_count, model.nu))
    object_pose = np.zeros((sample_count, 7))
    object_vel = np.zeros((sample_count, 6))
    contact_counts = np.zeros(sample_count, dtype=np.int64)
    total_normal = np.zeros(sample_count)

    settled_index = max(0, int(CLOSE_START * args.sample_hz) - 1)
    closed_index = int(CLOSE_END * args.sample_hz) - 1
    snapshot_grids = {"settled": None, "closed": None, "peak": None}
    peak_normal = -1.0
    latest_flat = np.zeros((TAXEL_COUNT, 3))

    renderer = mj.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
    scene_option = mj.MjvOption()
    scene_option.geomgroup[:] = 1
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
    for step in range(total_steps):
        t = (step + 1) * timestep
        data.ctrl[:] = control_targets(
            model, t, args.grip_fraction, args.pulse_hz, args.pulse_amplitude
        )
        mj.mj_step(model, data)
        if step % sample_every == 0 and sample_index < sample_count:
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
            tactile_times[sample_index] = t
            hand_qpos[sample_index] = data.qpos[hand_qpos_index]
            ctrl_log[sample_index] = data.ctrl
            object_pose[sample_index] = data.qpos[qpos_address : qpos_address + 7]
            object_vel[sample_index] = data.qvel[object_dof : object_dof + 6]
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
        if step % frame_every == 0:
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
            positions, marker_ids = taxel_positions(model, data, marker_sensors)
            draw_taxel_markers(
                renderer.scene,
                positions,
                marker_colours(
                    values[marker_ids], args.heatmap_channel, force_scale
                ),
                MARKER_RADIUS,
            )
            frames.append(paste_panel(
                renderer.render(),
                panel.render(values, args.heatmap_channel, force_scale),
                PANEL_MARGIN_PX,
            ))
    renderer.close()

    for tag, grids in snapshot_grids.items():
        if grids is not None:
            save_hand_heatmap(
                layout, grids, output_dir / f"hand_heatmap_{tag}.png"
            )
    mediapy.write_video(
        output_dir / "grasp.mp4", frames, fps=1.0 / (frame_every * timestep)
    )

    pulse_mask = tactile_times >= CLOSE_END
    contact_fraction = float(
        np.mean(contact_counts[pulse_mask] > 0)
    )
    final_distance = float(
        np.linalg.norm(object_pose[-1, :3] - palm_center)
    )
    divergences = int(data.warning[mj.mjtWarning.mjWARN_BADQACC].number)
    held = (
        final_distance <= MAX_FINAL_DISTANCE
        and object_pose[-1, 2] >= palm_center[2] - 0.05
    )
    passed = (
        divergences == 0
        and held
        and peak_normal >= MIN_PEAK_TOTAL_NORMAL
        and contact_fraction >= MIN_CONTACT_FRACTION
    )

    np.savez_compressed(
        output_dir / "sensor_log.npz",
        tactile=tactile,
        tactile_times=tactile_times,
        skin_names=np.array([name for name, _, _ in sensor_entries]),
        skin_taxel_ids=np.concatenate([ids for _, ids, _ in sensor_entries]),
        skin_taxel_counts=np.array(taxel_counts),
        tip_grid_rowcol=np.array(TIP_GRID_ROWCOL),
        id_grid=build_layout(HAND_WORKSPACE).id_grid,
        hand_qpos=hand_qpos,
        ctrl=ctrl_log,
        object_pose=object_pose,
        object_vel=object_vel,
        contact_count=contact_counts,
        total_normal=total_normal,
    )
    results = {
        "passed": bool(passed),
        "divergences": divergences,
        "object_held": bool(held),
        "object_final_distance_m": final_distance,
        "peak_total_normal_n": float(peak_normal),
        "contact_fraction": contact_fraction,
    }
    metadata = {
        "shape": shape,
        "object_scale": scale,
        "object_size": (
            np.asarray(OBJECT_SIZES[shape], dtype=np.float64) * scale
        ).tolist(),
        "object_mass_kg": mass,
        "phases": {
            "close_start_s": CLOSE_START,
            "close_end_s": CLOSE_END,
            "duration_s": args.duration,
        },
        "grip_fraction": args.grip_fraction,
        "pulse_hz": args.pulse_hz,
        "pulse_amplitude": args.pulse_amplitude,
        "orbit_revolutions": args.orbit_revolutions,
        "heatmap_channel": args.heatmap_channel,
        "force_scale_n": force_scale,
        "sample_hz": args.sample_hz,
        "tactile_shape": list(tactile.shape),
        "results": results,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    status = "PASS" if passed else "FAIL"
    label = shape if scale == 1.0 else f"{shape} {scale:g}x"
    print(
        f"{status} {label}: peak normal={peak_normal:.3f} N, "
        f"contact fraction={contact_fraction:.2f}, "
        f"final distance={final_distance * 1000.0:.0f} mm, "
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
            # Scale 1 keeps the original directory name so existing paths and
            # comparisons stay valid.
            name = shape if scale == 1.0 else f"{shape}_{scale:g}x"
            results = run_episode(
                shape, scale, args, args.output_root / name
            )
            runs += 1
            if not results["passed"]:
                failures.append(name)
    if failures:
        raise SystemExit(f"Grasp test failed for: {', '.join(failures)}")
    print(f"All {runs} grasp runs passed. Logs: {args.output_root}")


if __name__ == "__main__":
    main()
