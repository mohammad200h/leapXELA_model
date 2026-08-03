"""
Build a temporary MuJoCo scene XML for LeapXELA in-hand data collection.

Transforms leapXela_with_4_4_markers/robot.xml with ElementTree (MuJoCo 3.1.1
has no MjSpec): fixes the palm base palm-up, injects one site per taxel,
and adds floor, lights, cameras, and the manipulated object. The temp XML is
written beside robot.xml so meshdir="assets" resolves; callers unlink it.
"""

from __future__ import annotations

import pathlib
import tempfile
import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np

from leapxela.taxel_layout import TaxelLayout, quat_rotate

BASE_POS = (0.0, 0.0, 0.15)
# R_y(5 deg) * R_x(-100 deg): palm-up tilted 10 deg toward the fingers (so
# the object settles into the finger crease) and 5 deg toward the thumb side
# (so it does not escape past the ring finger, where no pinky blocks it).
BASE_QUAT = (0.64217645, -0.76531577, 0.02803736, 0.03341485)
PALM_CENTER_BODY = (0.02, -0.027, 0.10)

PALM_BODY_NAME = "leap_hand_xela_back_cover"
DEFAULT_CLASS_NAME = "leapXela_with_markers"

PATCH_SITE_COLORS = (
    "0.9 0.1 0.1 1", "0.1 0.9 0.1 1", "0.1 0.1 0.9 1", "0.9 0.9 0.1 1",
    "0.9 0.1 0.9 1", "0.1 0.9 0.9 1", "0.9 0.5 0.1 1", "0.5 0.1 0.9 1",
    "0.1 0.5 0.5 1", "0.5 0.5 0.1 1", "0.7 0.3 0.3 1", "0.3 0.7 0.3 1",
    "0.3 0.3 0.7 1", "0.8 0.8 0.8 1", "0.6 0.2 0.5 1", "0.2 0.6 0.5 1",
    "0.5 0.6 0.2 1", "0.9 0.7 0.4 1",
)


def palm_center_world() -> np.ndarray:
    return np.asarray(BASE_POS) + quat_rotate(np.asarray(BASE_QUAT), np.asarray(PALM_CENTER_BODY))


def _fmt(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _camera_xyaxes(pos: np.ndarray, target: np.ndarray) -> str:
    forward = np.asarray(target, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(forward, up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(x_axis, forward)
    return _fmt(np.concatenate([x_axis, y_axis]))


def build_scene_xml(
    robot_xml: pathlib.Path,
    layout: TaxelLayout,
    shape,
    spawn_pos: np.ndarray,
    spawn_quat: np.ndarray,
    timestep: float,
    actuator_kp_scale: float,
    actuator_kv: float,
    site_rgba_by_patch: bool,
) -> pathlib.Path:
    tree = ET.parse(robot_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    palm = worldbody.find(f"body[@name='{PALM_BODY_NAME}']")
    if palm is None:
        raise ValueError(f"Body '{PALM_BODY_NAME}' not found in {robot_xml}")
    freejoint = palm.find("freejoint")
    if freejoint is not None:
        palm.remove(freejoint)
    palm.set("pos", _fmt(BASE_POS))
    palm.set("quat", _fmt(BASE_QUAT))

    option = ET.SubElement(root, "option")
    option.set("timestep", f"{timestep:.8g}")
    option.set("integrator", "implicitfast")

    # robot.xml targets a newer MuJoCo; rewrite attributes MuJoCo 3.1.1 rejects:
    # position dampratio -> explicit kv, inheritrange -> explicit ctrlrange.
    default = root.find(f"default/default[@class='{DEFAULT_CLASS_NAME}']")
    position = default.find("position")
    if "dampratio" in position.attrib:
        del position.attrib["dampratio"]
    position.set("kv", f"{actuator_kv:.8g}")
    position.set("kp", f"{float(position.get('kp')) * actuator_kp_scale:.8g}")
    joint_ranges = {
        joint.get("name"): joint.get("range")
        for joint in worldbody.iter("joint")
        if joint.get("range") is not None
    }
    actuator_block = root.find("actuator")
    for actuator in actuator_block:
        if "inheritrange" in actuator.attrib:
            del actuator.attrib["inheritrange"]
            actuator.set("ctrlrange", joint_ranges[actuator.get("joint")])

    bodies = {body.get("name"): body for body in worldbody.iter("body")}
    patch_names = sorted({entry.patch for entry in layout.entries})
    for entry in layout.entries:
        site = ET.SubElement(bodies[entry.body], "site")
        site.set("name", entry.site_name)
        site.set("pos", _fmt(entry.pos))
        site.set("quat", _fmt(entry.quat))
        site.set("size", "0.0012")
        site.set("group", "4")
        if site_rgba_by_patch:
            site.set("rgba", PATCH_SITE_COLORS[patch_names.index(entry.patch)])
        else:
            site.set("rgba", "1 0 0 1")

    visual = ET.SubElement(root, "visual")
    offscreen = ET.SubElement(visual, "global")
    offscreen.set("offwidth", "1920")
    offscreen.set("offheight", "1440")
    headlight = ET.SubElement(visual, "headlight")
    headlight.set("diffuse", "0.6 0.6 0.6")
    headlight.set("ambient", "0.3 0.3 0.3")
    headlight.set("specular", "0 0 0")

    asset = root.find("asset")
    sky = ET.SubElement(asset, "texture")
    sky.set("type", "skybox")
    sky.set("builtin", "gradient")
    sky.set("rgb1", "0.3 0.5 0.7")
    sky.set("rgb2", "0 0 0")
    sky.set("width", "512")
    sky.set("height", "3072")
    ground_tex = ET.SubElement(asset, "texture")
    ground_tex.set("type", "2d")
    ground_tex.set("name", "groundplane")
    ground_tex.set("builtin", "checker")
    ground_tex.set("mark", "edge")
    ground_tex.set("rgb1", "0.2 0.3 0.4")
    ground_tex.set("rgb2", "0.1 0.2 0.3")
    ground_tex.set("markrgb", "0.8 0.8 0.8")
    ground_tex.set("width", "300")
    ground_tex.set("height", "300")
    ground_mat = ET.SubElement(asset, "material")
    ground_mat.set("name", "groundplane")
    ground_mat.set("texture", "groundplane")
    ground_mat.set("texuniform", "true")
    ground_mat.set("texrepeat", "5 5")
    ground_mat.set("reflectance", "0.2")

    light = ET.SubElement(worldbody, "light")
    light.set("pos", "0 0 3.5")
    light.set("dir", "0 0 -1")
    light.set("directional", "true")
    floor = ET.SubElement(worldbody, "geom")
    floor.set("name", "floor")
    floor.set("type", "plane")
    floor.set("size", "0 0 0.05")
    floor.set("material", "groundplane")

    center = palm_center_world()
    fixed_cam = ET.SubElement(worldbody, "camera")
    fixed_cam.set("name", "fixed")
    fixed_pos = center + np.array([0.28, -0.22, 0.22])
    fixed_cam.set("pos", _fmt(fixed_pos))
    fixed_cam.set("xyaxes", _camera_xyaxes(fixed_pos, center))
    top_cam = ET.SubElement(worldbody, "camera")
    top_cam.set("name", "top")
    top_pos = center + np.array([0.0, 0.0, 0.45])
    top_cam.set("pos", _fmt(top_pos))
    top_cam.set("xyaxes", "1 0 0 0 1 0")

    obj_body = ET.SubElement(worldbody, "body")
    obj_body.set("name", "object")
    obj_body.set("pos", _fmt(spawn_pos))
    obj_body.set("quat", _fmt(spawn_quat))
    obj_free = ET.SubElement(obj_body, "freejoint")
    obj_free.set("name", "free_object")
    obj_geom = ET.SubElement(obj_body, "geom")
    for key, value in shape.geom_attrs.items():
        obj_geom.set(key, value)

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=".leapxela_scene_",
        suffix=f"_{shape.name}.xml",
        dir=robot_xml.parent,
        delete=False,
    )
    with handle:
        tree.write(handle, encoding="utf-8", xml_declaration=True)
    return pathlib.Path(handle.name)
