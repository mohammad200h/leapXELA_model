"""
Visual verification of the LeapXELA taxel layout and virtual touch sensor.

layout mode: renders the palm-up hand with per-patch colored taxel sites and
prints per-patch counts and rest-pose world normals (palmar patches should
face world +z with the palm-up base).

press mode: drops a sphere onto the flat open hand, logs sensor totals vs the
object weight, and saves taxel heatmaps plus scene renders.

Example:
    conda run -n manta-ray mjpython leapxela/visualize_taxel_layout.py --mode layout
    conda run -n manta-ray mjpython leapxela/visualize_taxel_layout.py --mode press
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import defaultdict

import cv2
import mujoco as mj
import numpy as np

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from leapxela.scene_builder import build_scene_xml, palm_center_world  # noqa: E402
from leapxela.taxel_images import save_taxel_images  # noqa: E402
from leapxela.taxel_layout import ROBOT_XML, build_layout, pack_frames, patch_indices  # noqa: E402
from leapxela.touch_sensor import VirtualTaxelSensor  # noqa: E402

from train_mantaray_shape_classifier import SHAPES  # noqa: E402


def save_render(renderer, data, camera, path):
    scene_option = mj.MjvOption()
    scene_option.sitegroup[4] = 1
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    frame = renderer.render()
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"saved {path}")


def closeup_camera(azimuth, elevation, distance):
    camera = mj.MjvCamera()
    camera.lookat[:] = palm_center_world()
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    return camera


def run_layout_mode(args, layout, output_dir):
    spawn = np.array([0.4, 0.4, 0.5])
    xml = build_scene_xml(
        robot_xml=ROBOT_XML,
        layout=layout,
        shape=SHAPES["sphere"],
        spawn_pos=spawn,
        spawn_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        timestep=args.timestep,
        actuator_kp_scale=1.0,
        actuator_kv=args.actuator_kv,
        site_rgba_by_patch=True,
    )
    try:
        model = mj.MjModel.from_xml_path(str(xml))
        data = mj.MjData(model)
        mj.mj_forward(model, data)

        indices = patch_indices(layout)
        print(f"total taxels: {len(layout.entries)}")
        print(f"{'patch':8s} {'count':>5s}  mean world normal at rest")
        sensor = VirtualTaxelSensor(model, layout, ["obj"], args.kernel_sigma, args.kernel_cutoff)
        frames = sensor.taxel_frames(data)
        for patch in sorted(indices):
            normal = frames[indices[patch]][:, :, 2].mean(axis=0)
            print(
                f"{patch:8s} {len(indices[patch]):5d}  "
                f"[{normal[0]:+.2f} {normal[1]:+.2f} {normal[2]:+.2f}]"
            )

        renderer = mj.Renderer(model, args.render_height, args.render_width)
        try:
            save_render(renderer, data, "fixed", output_dir / "layout_fixed.png")
            save_render(renderer, data, "top", output_dir / "layout_top.png")
            save_render(renderer, data, closeup_camera(120, -25, 0.22), output_dir / "layout_closeup_fingers.png")
            save_render(renderer, data, closeup_camera(-135, -30, 0.22), output_dir / "layout_closeup_thumb.png")
        finally:
            renderer.close()
    finally:
        xml.unlink()


def run_press_mode(args, layout, output_dir):
    spawn = palm_center_world() + np.array([0.0, 0.0, 0.04])
    xml = build_scene_xml(
        robot_xml=ROBOT_XML,
        layout=layout,
        shape=SHAPES["sphere"],
        spawn_pos=spawn,
        spawn_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        timestep=args.timestep,
        actuator_kp_scale=1.0,
        actuator_kv=args.actuator_kv,
        site_rgba_by_patch=True,
    )
    try:
        model = mj.MjModel.from_xml_path(str(xml))
        data = mj.MjData(model)
        sensor = VirtualTaxelSensor(model, layout, ["obj"], args.kernel_sigma, args.kernel_cutoff)
        object_body = model.body("object")
        weight = float(object_body.mass[0]) * np.linalg.norm(model.opt.gravity)
        print(f"object mass {float(object_body.mass[0]):.4f} kg, weight {weight:.3f} N")

        tactile_frames = []
        sample_dt = 0.05
        next_sample = 0.0
        while data.time < args.press_duration:
            mj.mj_step(model, data)
            if data.time >= next_sample:
                grid = sensor.update(data)
                tactile_frames.append(grid)
                total_normal = float(grid[:, 2].sum())
                print(
                    f"t={data.time:5.2f}s  sum(normal_z)={total_normal:+.3f} N  "
                    f"|last_total_force|={np.linalg.norm(sensor.last_total_force):.3f} N  "
                    f"world_fz={sensor.last_total_force[2]:+.3f} N  "
                    f"obj_z={float(data.joint('free_object').qpos[2]):.3f}"
                )
                next_sample += sample_dt

        flat = np.stack(tactile_frames, axis=0)
        indices = patch_indices(layout)
        print("\nper-patch max |normal_z| over the press:")
        for patch in sorted(indices):
            peak = float(np.abs(flat[:, indices[patch], 2]).max())
            if peak > 0:
                print(f"  {patch:8s} {peak:.3f} N")

        packed = pack_frames(flat, layout)
        save_taxel_images(packed, layout.id_grid, output_dir)
        print(f"saved heatmaps under {output_dir / 'taxel_images'}")

        renderer = mj.Renderer(model, args.render_height, args.render_width)
        try:
            save_render(renderer, data, "fixed", output_dir / "press_fixed.png")
            save_render(renderer, data, "top", output_dir / "press_top.png")
        finally:
            renderer.close()
    finally:
        xml.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("layout", "press"), default="layout")
    parser.add_argument("--output-dir", default="data/leapxela_debug")
    parser.add_argument("--kernel-sigma", type=float, default=0.0035)
    parser.add_argument("--kernel-cutoff", type=float, default=0.01)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--actuator-kv", type=float, default=1.0)
    parser.add_argument("--press-duration", type=float, default=2.5)
    parser.add_argument("--render-width", type=int, default=1440)
    parser.add_argument("--render-height", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    layout = build_layout()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve() / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "layout":
        run_layout_mode(args, layout, output_dir)
    else:
        run_press_mode(args, layout, output_dir)


if __name__ == "__main__":
    main()
