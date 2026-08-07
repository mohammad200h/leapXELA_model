"""Side-by-side 3D taxel-ID maps for SparshSkin and LeapXELA.

Both stacks carry 368 taxels but order them differently. This plots each hand as
a 3D scatter with every taxel annotated with its taxel ID, so you can read off
where a given ID physically sits.

The two hands are DIFFERENT ROBOTS -- SparshSkin is an Allegro hand, LeapXELA is
a LEAP hand. They share the taxel composition (4 fingertips x 30 + 11 x 4x4 +
3 x 4x6 = 368) but not their kinematics, so the maps are read side by side, not
overlaid.

    conda run --no-capture-output -n leapxela python visualize_taxel_id_maps.py
    conda run --no-capture-output -n leapxela python visualize_taxel_id_maps.py \
        --leap-patch tip_if --sparsh-patch 0aftc --font-size 9
    conda run --no-capture-output -n leapxela python visualize_taxel_id_maps.py \
        --leap-patch tip_if --interactive

--interactive has to force a GUI backend explicitly: importing
generatehand_flexcom_sensor calls matplotlib.use("Agg") at module scope (correct
for the headless dataset renderer, wrong for a window), so plt.show() would
otherwise sit on a non-interactive canvas and warn instead of opening anything.
See select_interactive_backend().
"""

import argparse
import ast
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import mujoco as mj
import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODEL_DIR.parent
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from generatehand_flexcom_sensor import (
    FINGERTIP_POSE_JSON,
    GEOM_TO_PATCH,
    HAND_WORKSPACE,
    SENSOR_DEFINITIONS,
    TAXEL_COUNT,
    TIP_FINGERS,
    add_flex_sensor,
    add_tip_flex,
    load_base_model,
    tip_surface_offsets,
    trim_pad_boxes,
    _flex_vertices,
)
from leapxela.taxel_layout import build_layout

import hand_map_svg

SPARSH_ROOT = _REPO_ROOT / "sparsh-multisensory-touch-main"
SPARSH_UTILS = "tactile_ssl/data/xela/utils.py"
SPARSH_URDF = "assets/xela/urdf/ahrcpcpn.urdf"
SPARSH_MESH_DIR = "assets/xela/mesh"
SPARSH_ROOT_LINK = "world"
# Patch geometry from tactile_ssl/data/xela_tactile.py:220-252 ("numbers taken
# from mesh boundingbox"). Reproduced rather than imported: importing
# tactile_ssl pulls in torch/cv2/einops, which the leapxela env does not have.
SPARSH_AFTC_HWD = (0.031, 0.039, 0.029)
SPARSH_4X4_HWD = (0.026, 0.024, 0.0044)
SPARSH_4X6_HW = (0.052, 0.032)


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw, i.e. Rz @ Ry @ Rx."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def joint_transform(joint: ET.Element) -> np.ndarray:
    """Homogeneous origin transform of a URDF joint at zero joint angle.

    A revolute joint at angle 0 contributes only its origin, which is what the
    rest-pose map wants.
    """
    origin = joint.find("origin")
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if origin is not None:
        if origin.get("xyz") is not None:
            xyz = np.array([float(v) for v in origin.get("xyz").split()])
        if origin.get("rpy") is not None:
            rpy = np.array([float(v) for v in origin.get("rpy").split()])
    transform = np.eye(4)
    transform[:3, :3] = rpy_matrix(*rpy)
    transform[:3, 3] = xyz
    return transform


def sparsh_flatten_order(sparsh_root: Path) -> dict:
    """XELA_FLATTEN_ORDER read out of the sparsh source.

    Parsed rather than imported so it stays in sync with the file without
    dragging in the tactile_ssl dependency stack.
    """
    source = (sparsh_root / SPARSH_UTILS).read_text()
    match = re.search(r"XELA_FLATTEN_ORDER\s*=\s*(\{.*?\})", source, re.S)
    if match is None:
        raise ValueError(f"XELA_FLATTEN_ORDER not found in {SPARSH_UTILS}")
    return ast.literal_eval(match.group(1))


def sparsh_local_grid(patch: str) -> tuple:
    """(points, cells, shape) for one patch, in its link frame.

    Mirrors tactile_ssl/data/xela_tactile.py:220-252 exactly, including the
    fingertip taper: a 6x6 meshgrid keeping rows 0-3 whole, row -2 without its
    end columns, and row -1 without its two end columns on each side -> 30.

    `cells` is the (row, column) each point occupies on that meshgrid, in the
    same order as `points`. The flatten order already implies it; returning it
    makes the 2D map's cell -> id assignment come from this construction rather
    than from re-deriving a grid out of the projected positions.
    """
    if "aftc" in patch:
        h, w, d = SPARSH_AFTC_HWD
        h_res, w_res = 6, 6
        x = np.linspace(
            0.5 - h_res / 2, h_res / 2 + 0.5, h_res, endpoint=False
        ) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx, yy = np.meshgrid(x, y)
        flat_x = np.concatenate(
            [xx[:4, :].flatten(), xx[-2, 1:-1], xx[-1, 2:-2]], axis=0
        )
        flat_y = np.concatenate(
            [yy[:4, :].flatten(), yy[-2, 1:-1], yy[-1, 2:-2]], axis=0
        )
        cells = (
            [(r, c) for r in range(4) for c in range(6)]
            + [(4, c) for c in range(1, 5)]
            + [(5, c) for c in range(2, 4)]
        )
        shape = (w_res, h_res)
        depth = d
    elif "4x4" in patch:
        h, w, d = SPARSH_4X4_HWD
        h_res, w_res = 4, 4
        x = np.linspace(0.5, h_res + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx, yy = np.meshgrid(x, y)
        flat_x, flat_y = xx.flatten(), yy.flatten()
        cells = [(r, c) for r in range(w_res) for c in range(h_res)]
        shape = (w_res, h_res)
        depth = d
    elif "4x6" in patch:
        h, w = SPARSH_4X6_HW
        h_res, w_res = 6, 4
        x = np.linspace(0.5, h_res + 0.5, h_res, endpoint=False) * h / h_res
        y = np.linspace(0.5, w_res + 0.5, w_res, endpoint=False) * w / w_res
        xx, yy = np.meshgrid(x, y)
        flat_x, flat_y = xx.flatten(), yy.flatten()
        cells = [(r, c) for r in range(w_res) for c in range(h_res)]
        shape = (w_res, h_res)
        # The 4x6 branch never assigns a depth, so these sit at z = 0.
        depth = 0.0
    else:
        raise ValueError(f"unknown sparsh patch type: {patch}")
    points = np.zeros((flat_x.size, 4))
    points[:, 0] = flat_x
    points[:, 1] = flat_y
    points[:, 2] = depth
    points[:, 3] = 1.0
    return points, cells, shape


def sparsh_joint_index(root: ET.Element) -> dict:
    return {
        joint.find("child").get("link"): joint
        for joint in root.findall("joint")
    }


def sparsh_link_transform(joint_by_child: dict, link: str) -> np.ndarray:
    """Rest-pose transform of any URDF link into the `world` frame.

    Module level rather than a closure so the silhouette meshes and the taxel
    positions are placed by the same code; a second implementation could drift
    and put the pads off the hand.
    """
    transform = np.eye(4)
    chain = []
    while link in joint_by_child:
        joint = joint_by_child[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    if link != SPARSH_ROOT_LINK:
        raise ValueError(f"chain did not reach {SPARSH_ROOT_LINK!r}")
    for joint in reversed(chain):
        transform = transform @ joint_transform(joint)
    return transform


def sparsh_taxel_positions(sparsh_root: Path) -> tuple:
    """(368, 3) rest-pose taxel positions and their patch names, in ID order."""
    order = sparsh_flatten_order(sparsh_root)
    root = ET.parse(sparsh_root / SPARSH_URDF).getroot()
    joint_by_child = sparsh_joint_index(root)

    def link_transform(link: str) -> np.ndarray:
        return sparsh_link_transform(joint_by_child, link)

    positions = []
    patches = []
    for patch, count in order.items():
        local, _, _ = sparsh_local_grid(patch)
        if local.shape[0] != count:
            raise ValueError(
                f"{patch}: grid produced {local.shape[0]} taxels, "
                f"XELA_FLATTEN_ORDER declares {count}"
            )
        world = (link_transform(patch) @ local.T).T[:, :3]
        positions.append(world)
        patches.extend([patch] * count)
    return np.concatenate(positions, axis=0), patches


def leapxela_skins():
    """The sensorised LeapXELA model, with authoritative per-skin taxel ids."""
    entries_by_patch = {}
    for entry in build_layout(HAND_WORKSPACE).entries:
        entries_by_patch.setdefault(entry.patch, []).append(entry)
    spec = load_base_model("Box")
    trim_pad_boxes(spec)
    skins = [
        add_flex_sensor(
            spec, definition,
            entries_by_patch[GEOM_TO_PATCH[definition.geom_name]],
        )
        for definition in SENSOR_DEFINITIONS
    ]
    surface_offsets = tip_surface_offsets(
        json.loads(FINGERTIP_POSE_JSON.read_text())
    )
    tips = [
        add_tip_flex(
            spec, finger, entries_by_patch[f"tip_{finger}"],
            surface_offsets[finger],
        )
        for finger in TIP_FINGERS
    ]
    model = spec.compile()
    return model, skins, tips


def leapxela_taxel_positions(model, skins, tips, data) -> tuple:
    """(368, 3) rest-pose taxel positions and patch names, in taxel-ID order.

    Positions come from the flex vertices of the sensorised model -- the same
    membranes the dataset reads -- scattered into hardware taxel-ID order using
    each skin's own `taxel_ids`, so the map cannot disagree with the data.
    """
    positions = np.full((TAXEL_COUNT, 3), np.nan)
    patches = [None] * TAXEL_COUNT
    for skin in skins:
        vertices = _flex_vertices(model, data, skin.flex_name)
        positions[skin.taxel_ids] = vertices
        for taxel_id in skin.taxel_ids:
            patches[int(taxel_id)] = skin.definition.geom_name
    for tip in tips:
        vertices = _flex_vertices(model, data, tip.flex_name)
        positions[tip.taxel_ids] = vertices
        for taxel_id in tip.taxel_ids:
            patches[int(taxel_id)] = f"tip_{tip.finger}"
    if np.isnan(positions).any():
        missing = int(np.isnan(positions[:, 0]).sum())
        raise ValueError(f"{missing} taxel ids received no position")
    return positions, patches


def sparsh_patch_grids(sparsh_root: Path) -> dict:
    """patch name -> (rows, cols) taxel-id grid, -1 where the canvas is empty.

    Ids are the running index into XELA_FLATTEN_ORDER, which is what
    tactile_ssl means by a taxel id, and the cell of each one comes from the
    same construction that produced its position.
    """
    grids = {}
    first = 0
    for patch, count in sparsh_flatten_order(sparsh_root).items():
        _, cells, shape = sparsh_local_grid(patch)
        grid = np.full(shape, -1, dtype=np.int64)
        for offset, (row, column) in enumerate(cells):
            grid[row, column] = first + offset
        grids[patch] = grid
        first += count
    return grids


def leapxela_patch_grids(skins, tips) -> dict:
    """patch name -> (rows, cols) taxel-id grid, -1 where the canvas is empty.

    Both sources are authoritative. `pad_grid_from_entries` resolves each pad's
    cell from the calibrated positions and stores taxel_ids row-major, and the
    tips reuse `grid_rowcol`, the same canvas mapping `tip_canvas` uses to
    place live taxel readings.
    """
    grids = {}
    for skin in skins:
        rows, cols = skin.definition.count[0], skin.definition.count[1]
        grids[skin.definition.geom_name] = np.asarray(
            skin.taxel_ids, dtype=np.int64
        ).reshape(rows, cols)
    for tip in tips:
        grid = np.full((6, 6), -1, dtype=np.int64)
        for index, (row, column) in enumerate(tip.grid_rowcol):
            grid[row, column] = int(tip.taxel_ids[index])
        grids[f"tip_{tip.finger}"] = grid
    return grids


def build_hand(key, title, subtitle, grids, positions, hull_vertices) -> hand_map_svg.Hand:
    """Assemble a Hand: patches in canonical order, then the view they share."""
    canonical = hand_map_svg.canonical_names(1 if key == "sparsh" else 2)
    missing = set(canonical) ^ set(grids)
    if missing:
        raise ValueError(f"{key}: patch names not in PATCH_CORRESPONDENCE: {sorted(missing)}")
    patches = tuple(
        hand_map_svg.build_patch(name, canonical[name], grid, positions)
        for name, grid in grids.items()
    )
    horizontal, vertical, h_sign, v_sign = hand_map_svg.choose_view(patches, positions)
    hulls = hand_map_svg.hulls_from_groups(
        hull_vertices, (horizontal, vertical), (h_sign, v_sign)
    )
    return hand_map_svg.Hand(
        key=key, title=title, subtitle=subtitle, patches=patches, hulls=hulls,
        positions=positions, horizontal=horizontal, vertical=vertical,
        h_sign=h_sign, v_sign=v_sign,
    )


def draw_hand(axes, positions, patches, title) -> None:
    names = sorted(set(patches))
    colours = matplotlib.colormaps["tab20"](
        np.linspace(0.0, 1.0, max(len(names), 2))
    )
    colour_by_patch = dict(zip(names, colours))
    point_colours = np.array([colour_by_patch[p] for p in patches])
    axes.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=point_colours, s=18, depthshade=False,
    )
    # Tight per-axis limits with a true equal-scale box: a cube box aspect
    # wastes most of the canvas here, because both hands are essentially flat
    # slabs (spans ~250 x 240 x 40 mm), and small points make the ids unreadable.
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    span = np.maximum(upper - lower, 1.0e-4)
    margin = 0.04 * span.max()
    axes.set_xlim(lower[0] - margin, upper[0] + margin)
    axes.set_ylim(lower[1] - margin, upper[1] + margin)
    axes.set_zlim(lower[2] - margin, upper[2] + margin)
    axes.set_box_aspect(span + 2.0 * margin)
    axes.set_title(title, fontsize=11, pad=14)
    axes.set_xlabel("x (m)", fontsize=8)
    axes.set_ylabel("y (m)", fontsize=8)
    axes.set_zlabel("z (m)", fontsize=8)
    axes.tick_params(labelsize=6)
    # Both skins are near-planar, so the thinnest axis gets a very short screen
    # extent and its default ticks pile into an unreadable blob.
    axes.locator_params(nbins=5)


def face_on_view(positions: np.ndarray) -> tuple:
    """Elevation/azimuth looking down a hand's thinnest axis.

    Both skins are near-planar but in different planes (the Allegro links run
    along x, the LEAP palm lies in xy), so one shared viewpoint shows one hand
    face-on and the other edge-on. Viewing along the axis of least extent puts
    every patch in the picture plane, which is what makes the ids readable.
    """
    span = positions.max(axis=0) - positions.min(axis=0)
    thin = int(np.argmin(span))
    return ((0.0, 0.0), (0.0, -90.0), (90.0, -90.0))[thin]


def select_interactive_backend(requested: str):
    """Switch matplotlib to a GUI backend, or return None if none is available.

    `force=True` is essential, not defensive: importing
    generatehand_flexcom_sensor runs `matplotlib.use("Agg")` at module scope,
    so by the time main() runs the backend is already Agg and pyplot is already
    imported. A plain `matplotlib.use(...)` will not dislodge that, which is why
    --interactive previously only produced a "FigureCanvasAgg is non-interactive"
    warning and no window.
    """
    if requested:
        candidates = [requested]
    elif sys.platform == "darwin":
        candidates = ["macosx", "tkagg"]
    else:
        candidates = ["tkagg", "qtagg"]
    for name in candidates:
        # The one try/except in this file: matplotlib raises when a toolkit is
        # missing, and probing for it any other way means reimplementing its
        # backend-name-to-module mapping. Narrow, and the fallback is the point.
        try:
            matplotlib.use(name, force=True)
        except Exception as error:
            print(f"  backend {name!r} unavailable: {error}")
            continue
        chosen = matplotlib.get_backend()
        if chosen.lower() == "agg":
            # e.g. --backend agg: selectable, but it can never show a window.
            print(f"  backend {name!r} is non-interactive, so no window can open")
            continue
        return chosen
    print(
        f"no interactive backend available (tried {', '.join(candidates)}); "
        "the PNG and PDF were still written -- open the PDF and zoom instead"
    )
    return None


def filter_patch(positions, patches, pattern):
    keep = [i for i, p in enumerate(patches) if pattern in p]
    if not keep:
        raise ValueError(
            f"no patch matching {pattern!r}; available: {sorted(set(patches))}"
        )
    return positions[keep], [patches[i] for i in keep], keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Side-by-side 3D taxel-ID maps for SparshSkin and LeapXELA.",
    )
    parser.add_argument("--sparsh-root", type=Path, default=SPARSH_ROOT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "data" / "taxel_id_maps",
    )
    parser.add_argument(
        "--label-step", type=int, default=1,
        help="label every Nth taxel id (0 disables labels)",
    )
    # Separate filters per hand: the two naming schemes have nothing in common
    # ("tip_if"/"if_md_uspa44" vs "0aftc_palm_link"/"link_2_4x4_palm_link"), so
    # a single shared substring would almost never match both.
    parser.add_argument(
        "--sparsh-patch", default="",
        help="substring filter for the sparsh hand, e.g. 'aftc' or '4x4'",
    )
    parser.add_argument(
        "--leap-patch", default="",
        help="substring filter for the leapxela hand, e.g. 'tip_if' or 'uspa44'",
    )
    parser.add_argument(
        "--list-patches", action="store_true",
        help="print both hands' patch names with their id ranges and exit",
    )
    parser.add_argument(
        "--elev", type=float, default=float("nan"),
        help="elevation in degrees; default views each hand face-on",
    )
    parser.add_argument(
        "--azim", type=float, default=float("nan"),
        help="azimuth in degrees; default views each hand face-on",
    )
    parser.add_argument("--font-size", type=float, default=5.0)
    parser.add_argument(
        "--view", choices=("3d", "hand", "both"), default="both",
        help="'3d' the scatter PNG/PDF, 'hand' the interactive flat hand map, "
             "'both' (default) writes each",
    )
    parser.add_argument(
        "--html-out", type=Path, default=None,
        help="path for the hand map; defaults to taxel_id_map.html in --output-dir",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="also open a rotatable window for the 3D view (files are written "
             "either way); the hand map is interactive in a browser regardless",
    )
    parser.add_argument(
        "--backend", default="",
        help="matplotlib GUI backend for --interactive; default tries macosx "
             "then tkagg on macOS. Use 'webagg' to rotate the plot in a "
             "browser, which does not need an attached window session",
    )
    return parser.parse_args()


def write_3d_figure(args, plt, sparsh, leap) -> None:
    sparsh_positions, sparsh_patches = sparsh
    leap_positions, leap_patches = leap
    sparsh_ids = list(range(len(sparsh_positions)))
    leap_ids = list(range(len(leap_positions)))
    if args.sparsh_patch:
        sparsh_positions, sparsh_patches, sparsh_ids = filter_patch(
            sparsh_positions, sparsh_patches, args.sparsh_patch
        )
        print(f"sparsh filtered to {args.sparsh_patch!r}: "
              f"{len(sparsh_ids)} taxels")
    if args.leap_patch:
        leap_positions, leap_patches, leap_ids = filter_patch(
            leap_positions, leap_patches, args.leap_patch
        )
        print(f"leapxela filtered to {args.leap_patch!r}: "
              f"{len(leap_ids)} taxels")

    figure = plt.figure(figsize=(20, 10))
    for index, (positions, patches, ids, title) in enumerate((
        (sparsh_positions, sparsh_patches, sparsh_ids,
         "SparshSkin (Allegro hand)\nid = index in XELA_FLATTEN_ORDER"),
        (leap_positions, leap_patches, leap_ids,
         "LeapXELA (LEAP hand)\nid = hardware LEAP_XELA_ID"),
    )):
        axes = figure.add_subplot(1, 2, index + 1, projection="3d")
        elev, azim = face_on_view(positions)
        if not np.isnan(args.elev):
            elev = args.elev
        if not np.isnan(args.azim):
            azim = args.azim
        axes.view_init(elev=elev, azim=azim)
        draw_hand(axes, positions, patches, title)
        for point, taxel_id in zip(positions, ids):
            if args.label_step > 0 and taxel_id % args.label_step == 0:
                axes.text(
                    point[0], point[1], point[2], str(taxel_id),
                    fontsize=args.font_size, ha="center", va="center",
                )
    figure.suptitle(
        "Taxel ID maps -- 368 taxels each. Different robots (Allegro vs LEAP): "
        "comparable in composition, not in physical position.",
        fontsize=13, y=0.98,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tags = [t for t in (args.sparsh_patch, args.leap_patch) if t]
    stem = "taxel_id_map" + ("_" + "_".join(tags) if tags else "")
    for suffix in ("png", "pdf"):
        path = args.output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=220 if suffix == "png" else None)
        print(f"wrote {path}")


def write_hand_map(args, sparsh_positions, leap_positions,
                   leap_model, leap_data, leap_skins, leap_tips) -> None:
    urdf_root = ET.parse(args.sparsh_root / SPARSH_URDF).getroot()
    sparsh = build_hand(
        "sparsh", "SparshSkin — Allegro hand",
        "id = running index into XELA_FLATTEN_ORDER (contiguous per patch)",
        sparsh_patch_grids(args.sparsh_root), sparsh_positions,
        hand_map_svg.sparsh_link_vertices(
            urdf_root, args.sparsh_root / SPARSH_MESH_DIR,
            lambda link: sparsh_link_transform(sparsh_joint_index(urdf_root), link),
            joint_transform,
        ),
    )
    leap = build_hand(
        "leapxela", "LeapXELA — LEAP hand",
        "id = hardware LEAP_XELA_ID (interleaved across patches)",
        leapxela_patch_grids(leap_skins, leap_tips), leap_positions,
        hand_map_svg.leapxela_link_vertices(leap_model, leap_data),
    )
    path = args.html_out if args.html_out is not None else \
        args.output_dir / "taxel_id_map.html"
    hand_map_svg.write_hand_map_html(path, (sparsh, leap))
    for hand in (sparsh, leap):
        unrolled = [p.canonical for p in hand.patches
                    if hand_map_svg.panel_axes(p, hand)[2]]
        axes = "xyz"
        print(f"{hand.key:9s} view: right = {'+-'[hand.h_sign < 0]}{axes[hand.horizontal]}, "
              f"up = {'+-'[hand.v_sign < 0]}{axes[hand.vertical]}, "
              f"{len(hand.hulls)} link hulls, "
              f"unrolled patches: {', '.join(unrolled) if unrolled else 'none'}")
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    interactive_backend = None
    if args.interactive:
        interactive_backend = select_interactive_backend(args.backend)
    else:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    sparsh_positions, sparsh_patches = sparsh_taxel_positions(args.sparsh_root)
    leap_model, leap_skins, leap_tips = leapxela_skins()
    leap_data = mj.MjData(leap_model)
    mj.mj_forward(leap_model, leap_data)
    leap_positions, leap_patches = leapxela_taxel_positions(
        leap_model, leap_skins, leap_tips, leap_data
    )
    print(f"sparsh (Allegro): {len(sparsh_positions)} taxels, "
          f"{len(set(sparsh_patches))} patches")
    print(f"leapxela (LEAP) : {len(leap_positions)} taxels, "
          f"{len(set(leap_patches))} patches")

    if args.list_patches:
        for tag, patches in (("sparsh", sparsh_patches),
                             ("leapxela", leap_patches)):
            print(f"\n{tag} patches:")
            for name in dict.fromkeys(patches):
                ids = [i for i, p in enumerate(patches) if p == name]
                print(f"  {name:26s} {len(ids):>3d} taxels  "
                      f"ids {min(ids)}..{max(ids)}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.view in ("3d", "both"):
        write_3d_figure(args, plt, (sparsh_positions, sparsh_patches),
                        (leap_positions, leap_patches))
    if args.view in ("hand", "both"):
        # The hand map always draws all 368: --sparsh-patch/--leap-patch narrow
        # a static figure, but this one zooms and searches instead.
        if args.sparsh_patch or args.leap_patch:
            print("note: patch filters apply to the 3D view only; "
                  "use the hand map's search box and zoom")
        write_hand_map(args, sparsh_positions, leap_positions,
                       leap_model, leap_data, leap_skins, leap_tips)
    if args.interactive and interactive_backend is not None:
        print(f"opening interactive window ({interactive_backend} backend); "
              "close it to exit")
        plt.show()


if __name__ == "__main__":
    main()
