"""Interactive grasp with a live taxel heatmap.

Runs the same grasp schedule as grasp_touch_test.py in the passive MuJoCo
viewer, painting per-taxel forces onto the hand as 3D markers and drawing the
palm-view panel inside the viewer window.

macOS needs mjpython:
    conda run -n leapxela mjpython live_taxel_viewer.py --object box

Keys: SPACE pause/resume, C cycle channel, R restart the grasp.
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer
import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from generatehand_flexcom_sensor import (
    TAXEL_COUNT,
    FlexTaxelSensor,
    build_hand_layout,
    pad_vertex_rotations,
)
from grasp_touch_test import (
    OBJECT_GEOM_NAME,
    OBJECT_SIZES,
    build_model,
    control_targets,
    object_mass,
    object_qpos_address,
    spawn_pose,
)
from taxel_visualizer import (
    CHANNELS,
    MARKER_RADIUS,
    TaxelPanel,
    channel_values,
    dim_flex_skins,
    draw_taxel_markers,
    marker_colours,
    panel_taxel_ids,
    taxel_positions,
)


PANEL_SIZE = 300
PANEL_MARGIN = 12
DISPLAY_HZ = 60.0
KEY_SPACE = 32
KEY_C = ord("C")
KEY_R = ord("R")


class LiveState:
    def __init__(self, channel: str, force_scale: float):
        self.channel = channel
        self.force_scale = force_scale
        self.paused = False
        self.restart = False

    def on_key(self, keycode: int) -> None:
        if keycode == KEY_SPACE:
            self.paused = not self.paused
        elif keycode == KEY_C:
            self.channel = CHANNELS[
                (CHANNELS.index(self.channel) + 1) % len(CHANNELS)
            ]
        elif keycode == KEY_R:
            self.restart = True


def reset_episode(model: mj.MjModel, data: mj.MjData, shape: str) -> None:
    mj.mj_resetData(model, data)
    pose, _ = spawn_pose(model, shape)
    address = object_qpos_address(model)
    data.qpos[address : address + 7] = pose
    mj.mj_forward(model, data)


def main() -> None:
    args = parse_args()
    model, skins, tips = build_model(
        args.object, args.object_scale, object_mass(args.object_scale, 0.0)
    )
    dim_flex_skins(model)
    data = mj.MjData(model)

    sensors = []
    for skin in skins:
        sensors.append((
            skin.taxel_ids,
            FlexTaxelSensor(
                model, skin.flex_name, skin.parent_body_name,
                pad_vertex_rotations(skin), [OBJECT_GEOM_NAME],
            ),
        ))
    for tip in tips:
        sensors.append((
            tip.taxel_ids,
            FlexTaxelSensor(
                model, tip.flex_name, tip.parent_body_name,
                tip.vertex_rotations, [OBJECT_GEOM_NAME],
            ),
        ))
    marker_sensors = (
        [(skin.flex_name, skin.taxel_ids) for skin in skins]
        + [(tip.flex_name, tip.taxel_ids) for tip in tips]
    )
    panel = TaxelPanel(
        build_hand_layout(model, mj.MjData(model), skins, tips),
        panel_taxel_ids(skins, tips), PANEL_SIZE, PANEL_SIZE, TAXEL_COUNT,
    )
    viewport = mj.MjrRect(PANEL_MARGIN, PANEL_MARGIN, PANEL_SIZE, PANEL_SIZE)

    state = LiveState(args.channel, args.force_scale)
    reset_episode(model, data, args.object)
    # Physics costs ~37 ms/step once the grip loads the flex skins, so the
    # display, not real time, sets the pace: few substeps per frame keeps the
    # viewer responsive. Raise --steps-per-frame to advance the grasp faster.
    steps_per_frame = args.steps_per_frame
    readings = np.zeros((TAXEL_COUNT, 3))
    episode_start = 0.0

    print(f"Live taxel heatmap — {args.object}. "
          f"SPACE pause, C cycle channel, R restart.")
    with mujoco.viewer.launch_passive(
        model, data, key_callback=state.on_key
    ) as viewer:
        while viewer.is_running():
            frame_start = time.time()
            if state.restart:
                reset_episode(model, data, args.object)
                episode_start = data.time
                state.restart = False
            if not state.paused:
                for _ in range(steps_per_frame):
                    elapsed = data.time - episode_start
                    if elapsed >= args.duration:
                        reset_episode(model, data, args.object)
                        episode_start = data.time
                        elapsed = 0.0
                    data.ctrl[:] = control_targets(
                        model, elapsed, args.grip_fraction, args.pulse_hz,
                        args.pulse_amplitude,
                    )
                    mj.mj_step(model, data)

            contact_total = 0
            readings[:] = 0.0
            for taxel_ids, sensor in sensors:
                values, contacts = sensor.read(data)
                readings[taxel_ids] = values
                contact_total += contacts
            scalars = channel_values(readings, state.channel)
            state.force_scale = max(
                state.force_scale, float(np.max(np.abs(scalars)))
            )

            positions, marker_ids = taxel_positions(model, data, marker_sensors)
            viewer.user_scn.ngeom = 0
            draw_taxel_markers(
                viewer.user_scn, positions,
                marker_colours(
                    scalars[marker_ids], state.channel, state.force_scale
                ),
                MARKER_RADIUS,
            )
            viewer.set_images(
                (viewport, panel.render(
                    scalars, state.channel, state.force_scale
                ))
            )
            viewer.set_texts((
                None, None,
                "channel\nscale\ntotal normal\npeak taxel\ncontacts\nstate",
                f"{state.channel}\n{state.force_scale:.2f} N\n"
                f"{np.sum(readings[:, 2]):.2f} N\n"
                f"{np.max(readings[:, 2]):.3f} N\n{contact_total}\n"
                f"{'paused' if state.paused else 'running'}",
            ))
            viewer.sync()

            remaining = steps_per_frame * model.opt.timestep - (
                time.time() - frame_start
            )
            if remaining > 0:
                time.sleep(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive grasp with a live taxel heatmap."
    )
    parser.add_argument(
        "--object", choices=sorted(OBJECT_SIZES), default="box"
    )
    parser.add_argument(
        "--object-scale", type=float, default=1.0,
        help="object size multiplier (mass follows scale^3)",
    )
    parser.add_argument("--channel", choices=CHANNELS, default="normal")
    parser.add_argument("--force-scale", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--grip-fraction", type=float, default=0.55)
    parser.add_argument("--pulse-hz", type=float, default=1.0)
    parser.add_argument("--pulse-amplitude", type=float, default=0.15)
    parser.add_argument(
        "--steps-per-frame", type=int, default=1,
        help="physics steps per rendered frame (higher = faster grasp, "
             "choppier display)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
