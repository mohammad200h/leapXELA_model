"""Flat hand-outline taxel-ID maps, emitted as one self-contained HTML page.

The 3D scatter in visualize_taxel_id_maps.py shows where taxels are but not
which id is which -- labels overlap and the curved fingertips self-occlude.
This module draws each hand flat instead: the real link silhouette underneath,
and every sensor patch as an undistorted cell grid sitting at its true position
on that silhouette, one taxel id per cell.

Two deliberate choices, because they are the difference between a map and a
picture:

* Panels are drawn UNDISTORTED, not projected. Several patches (both thumbs)
  face sideways in their hand's view plane and project to a line, so a faithful
  projection would hide their ids completely. Each such panel is unrolled into
  the picture plane and flagged, rather than silently squashed.
* Cell -> taxel-id assignment is taken from the model's own tables (FlexSkin /
  TipSkin / the sparsh flatten order), never re-derived from geometry. Only the
  panel's on-screen angle and mirroring come from the projection, so a bad
  projection can rotate the map but can never mislabel it.

No matplotlib: this module is import-safe regardless of the Agg lock described
in visualize_taxel_id_maps.py, and the page it writes needs no backend at all.
"""

from __future__ import annotations

import html
import json
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial import ConvexHull

MM = 1000.0
# Below this projected length a patch axis is treated as edge-on: the panel is
# unrolled into the picture plane and captioned as such.
EDGE_ON_FRACTION = 0.35
CELL_GAP = 0.12          # fraction of pitch left as gutter between cells
HULL_TOLERANCE_MM = 3.0  # how far off its link a pad may sit; see check_hand
FLATNESS_TOLERANCE_MM = 0.5  # out-of-plane spread below which a patch counts as flat
PANEL_FIDELITY_MM = 0.1      # how far a flat face-on cell may draw from its taxel
# Three monospace digits occupy about 1.8 em, so this keeps the widest id
# inside the narrower cell dimension.
LABEL_DIGIT_FRACTION = 0.52
CAPTION_FRACTION = 0.014  # patch caption size as a fraction of the hand's extent

# Canonical patch names, shared by both hands so a patch can be highlighted on
# one and located on the other. Fingers and tips are verified geometrically:
# the sparsh fingertip centroids sit at y = +56 / 0 / -56 mm (index / middle /
# ring) and each chain's patches order monotonically outward from the palm,
# matching the leap bs -> px -> md -> tip order.
#
# The palm triple is INFERRED FROM ARRANGEMENT, not from any documented
# correspondence: in both hands two 4x6 pads sit side by side with a third
# directly below the non-thumb one, and ahr_palm_2 / uspa46_2 are the pads on
# the thumb side. Treat it as a strong guess, not as ground truth.
PATCH_CORRESPONDENCE = (
    ("thumb.tip",        "3aftc_palm_link",          "tip_th"),
    ("thumb.distal",     "link_15_4x4_palm_link",    "th_ds_uspa44"),
    ("thumb.proximal",   "link_14_4x4_palm_link",    "th_px_uspa44"),
    ("index.tip",        "0aftc_palm_link",          "tip_if"),
    ("index.distal",     "link_2_4x4_palm_link",     "if_md_uspa44"),
    ("index.proximal",   "link_1A_4x4_palm_link",    "if_px_uspa44"),
    ("index.base",       "link_1B_4x4_palm_link",    "if_bs_uspa44"),
    ("middle.tip",       "1aftc_palm_link",          "tip_mf"),
    ("middle.distal",    "link_6_4x4_palm_link",     "mf_md_uspa44"),
    ("middle.proximal",  "link_5A_4x4_palm_link",    "mf_px_uspa44"),
    ("middle.base",      "link_5B_4x4_palm_link",    "mf_bs_uspa44"),
    ("ring.tip",         "2aftc_palm_link",          "tip_rf"),
    ("ring.distal",      "link_10_4x4_palm_link",    "rf_md_uspa44"),
    ("ring.proximal",    "link_9A_4x4_palm_link",    "rf_px_uspa44"),
    ("ring.base",        "link_9B_4x4_palm_link",    "rf_bs_uspa44"),
    ("palm.thumbside",   "ahr_palm_2_4x6_palm_link", "uspa46_2"),
    ("palm.centre",      "ahr_palm_1_4x6_palm_link", "uspa46_1"),
    ("palm.lower",       "ahr_palm_3_4x6_palm_link", "uspa46_3"),
)

# Hue per finger group, lightness per segment: corresponding patches get the
# same colour on both hands, which is what makes the pairing readable at a
# glance without hovering anything.
GROUP_HUE = {"thumb": 18, "index": 205, "middle": 145, "ring": 275, "palm": 45}
SEGMENT_LIGHT = {"tip": 62, "distal": 52, "proximal": 42, "base": 34,
                 "thumbside": 55, "centre": 45, "lower": 36}


@dataclass(frozen=True)
class Patch:
    """One sensor patch: its id grid plus the frame needed to draw it flat."""
    name: str
    canonical: str
    id_grid: np.ndarray       # (rows, cols), -1 where the canvas has no taxel
    centre: np.ndarray        # (3,) patch centroid, hand frame
    axis_row: np.ndarray      # (3,) unit, direction of increasing row index
    axis_col: np.ndarray      # (3,) unit, direction of increasing column index
    pitch_row: float
    pitch_col: float


@dataclass(frozen=True)
class Hand:
    key: str
    title: str
    subtitle: str
    patches: tuple
    hulls: tuple              # each (N, 2) in view coords, metres
    positions: np.ndarray     # (368, 3) hand frame, for the tooltip
    horizontal: int           # hand-frame axis drawn left-to-right
    vertical: int             # hand-frame axis drawn bottom-to-top
    h_sign: float
    v_sign: float


# --------------------------------------------------------------------------
# meshes -> silhouette
# --------------------------------------------------------------------------

def binary_stl_vertices(path: Path) -> np.ndarray:
    """(3F, 3) triangle vertices of a binary STL.

    trimesh is installed in the leapxela env but is not worth a dependency for
    a 50-byte-per-triangle record: 80-byte header, uint32 count, then per
    triangle a normal, three vertices and a 2-byte attribute.
    """
    raw = path.read_bytes()
    count = struct.unpack("<I", raw[80:84])[0]
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"{path.name} is not a binary STL ({len(raw)} bytes)")
    records = np.frombuffer(raw, dtype=np.uint8, count=50 * count, offset=84)
    records = records.reshape(count, 50)
    return records[:, 12:48].copy().view(np.float32).reshape(-1, 3).astype(np.float64)


def hulls_from_groups(vertex_groups: dict, axes: tuple, signs: tuple) -> tuple:
    """Per-group 2D convex hulls in view coordinates.

    Hulled per link and drawn as an overlapping union rather than hulled once
    over the whole hand: a single hull spans the gaps between fingers and comes
    out as a mitten.
    """
    hulls = []
    for vertices in vertex_groups.values():
        points = np.concatenate(vertices)[:, list(axes)] * np.asarray(signs)
        hull = ConvexHull(points)
        hulls.append(points[hull.vertices])
    return tuple(hulls)


def leapxela_link_vertices(model: mj.MjModel, data: mj.MjData) -> dict:
    """body name -> world-frame mesh vertices, for every mesh geom on the hand."""
    groups: dict = {}
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mj.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        local = model.mesh_vert[start:start + model.mesh_vertnum[mesh]].astype(np.float64)
        world = data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T
        body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
        groups.setdefault(body, []).append(world)
    return groups


def sparsh_link_vertices(urdf_root: ET.Element, mesh_dir: Path,
                         link_transform, origin_transform) -> dict:
    """link name -> world-frame mesh vertices, from the URDF visual meshes.

    link_transform / origin_transform are passed in rather than reimplemented:
    visualize_taxel_id_maps.py already walks the joint chain to `world` and
    already parses a URDF <origin>, and the silhouette has to agree with the
    taxel positions those same functions produce.
    """
    groups: dict = {}
    for link in urdf_root.findall("link"):
        visual = link.find("visual")
        if visual is None:
            continue
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            raise ValueError(f"link {link.get('name')} has a non-mesh visual")
        relative = mesh.get("filename").replace("package://xela_models/mesh/", "")
        local = binary_stl_vertices(mesh_dir / relative)
        transform = link_transform(link.get("name")) @ origin_transform(visual)
        world = local @ transform[:3, :3].T + transform[:3, 3]
        groups.setdefault(link.get("name"), []).append(world)
    return groups


# --------------------------------------------------------------------------
# patches
# --------------------------------------------------------------------------

def patch_frame(id_grid: np.ndarray, positions: np.ndarray) -> tuple:
    """(centre, axis_row, axis_col, pitch_row, pitch_col) from the id grid.

    Derived from the taxel positions themselves given the known cell
    assignment, so it needs no per-hand metadata and cannot drift from the
    layout tables. Steps are averaged over every adjacent pair present in the
    canvas, which tolerates the holes in the 6x6 fingertip grids.
    """
    rows, cols = id_grid.shape
    present = id_grid >= 0

    def step(along_rows: bool) -> np.ndarray:
        deltas = []
        limit = rows - 1 if along_rows else cols - 1
        for index in range(limit):
            near = present[index] & present[index + 1] if along_rows else \
                present[:, index] & present[:, index + 1]
            if not near.any():
                continue
            first = id_grid[index][near] if along_rows else id_grid[:, index][near]
            second = id_grid[index + 1][near] if along_rows else id_grid[:, index + 1][near]
            deltas.append(positions[second] - positions[first])
        return np.concatenate(deltas).mean(axis=0)

    row_step, col_step = step(True), step(False)
    pitch_row = float(np.linalg.norm(row_step))
    pitch_col = float(np.linalg.norm(col_step))

    # Centre of the FULL grid, not the mean of the taxels present. The 6x6
    # fingertip canvases are missing 6 cells from their last two rows, so the
    # mean of what is there sits off-centre, and every cell then drew 2.8 mm
    # (sparsh) to 6.4 mm (leapxela) from its real position.
    where_row, where_col = np.nonzero(present)
    offsets = (
        (where_row - (rows - 1) / 2.0)[:, None] * row_step
        + (where_col - (cols - 1) / 2.0)[:, None] * col_step
    )
    centre = (positions[id_grid[present]] - offsets).mean(axis=0)
    return (centre, row_step / pitch_row, col_step / pitch_col, pitch_row, pitch_col)


def build_patch(name: str, canonical: str, id_grid: np.ndarray,
                positions: np.ndarray) -> Patch:
    centre, axis_row, axis_col, pitch_row, pitch_col = patch_frame(id_grid, positions)
    return Patch(
        name=name, canonical=canonical, id_grid=id_grid, centre=centre,
        axis_row=axis_row, axis_col=axis_col,
        pitch_row=pitch_row, pitch_col=pitch_col,
    )


def canonical_names(column: int) -> dict:
    return {row[column]: row[0] for row in PATCH_CORRESPONDENCE}


# --------------------------------------------------------------------------
# view selection
# --------------------------------------------------------------------------

def choose_view(patches: tuple, positions: np.ndarray) -> tuple:
    """(horizontal, vertical, h_sign, v_sign) putting fingers up, thumb right.

    The plane is the hand's own: both skins are near-planar slabs, so the axis
    of least extent is the one to look down. Which in-plane axis is "up" and
    which way each points is then fixed from the anatomy -- tips above the
    palm, thumb right of the ring finger -- so the two hands can be read
    against each other instead of arriving in whatever orientation their model
    frames happen to use.
    """
    span = positions.max(axis=0) - positions.min(axis=0)
    plane = [axis for axis in range(3) if axis != int(np.argmin(span))]

    by_canonical = {patch.canonical: patch.centre for patch in patches}
    tips = np.array([by_canonical[f"{f}.tip"]
                     for f in ("thumb", "index", "middle", "ring")]).mean(axis=0)
    palm = np.array([by_canonical[f"palm.{p}"]
                     for p in ("thumbside", "centre", "lower")]).mean(axis=0)

    separation = np.abs(tips - palm)[plane]
    vertical = plane[int(np.argmax(separation))]
    horizontal = plane[1 - int(np.argmax(separation))]
    v_sign = float(np.sign(tips[vertical] - palm[vertical]))
    h_sign = float(np.sign(by_canonical["thumb.tip"][horizontal]
                           - by_canonical["ring.tip"][horizontal]))
    if v_sign == 0.0 or h_sign == 0.0:
        raise ValueError("hand anatomy does not disambiguate the view axes")
    return horizontal, vertical, h_sign, v_sign


def project(points: np.ndarray, hand: Hand) -> np.ndarray:
    """Hand-frame points -> view coordinates in mm, y up."""
    flat = np.atleast_2d(points)
    return np.column_stack([
        flat[:, hand.horizontal] * hand.h_sign * MM,
        flat[:, hand.vertical] * hand.v_sign * MM,
    ])


def panel_axes(patch: Patch, hand: Hand) -> tuple:
    """(u_row, u_col, unrolled) -- orthonormal 2D panel axes in view coords.

    The projected patch axes are neither unit nor orthogonal unless the patch
    happens to face the viewer, and for an edge-on patch one of them collapses
    to nothing. So the longer projection sets the panel's angle and the shorter
    one only picks the sign of its perpendicular, which keeps the panel
    undistorted while preserving the handedness a viewer would actually see.
    """
    projected_row = project(patch.axis_row, hand)[0]
    projected_col = project(patch.axis_col, hand)[0]
    length_row = float(np.linalg.norm(projected_row))
    length_col = float(np.linalg.norm(projected_col))
    unrolled = min(length_row, length_col) < EDGE_ON_FRACTION

    if length_row >= length_col:
        u_row = projected_row / length_row
        perpendicular = np.array([-u_row[1], u_row[0]])
        sign = 1.0 if float(perpendicular @ projected_col) >= 0.0 else -1.0
        u_col = perpendicular * sign
    else:
        u_col = projected_col / length_col
        perpendicular = np.array([-u_col[1], u_col[0]])
        sign = 1.0 if float(perpendicular @ projected_row) >= 0.0 else -1.0
        u_row = perpendicular * sign
    return u_row, u_col, unrolled


def panel_corners(patch: Patch, hand: Hand) -> np.ndarray:
    """The panel's drawn footprint: exactly the extent of its cells.

    No decorative padding, deliberately. Some patches really are edge to edge
    on the hand -- the sparsh palm pads are 51.3 mm apart with 50.9 mm of cells
    -- so any padding here would either overlap a neighbour or misrepresent the
    spacing. Frames of adjacent pads abutting is the honest rendering.
    """
    u_row, u_col, _ = panel_axes(patch, hand)
    rows, cols = patch.id_grid.shape
    half_row = (rows - CELL_GAP) * patch.pitch_row * MM / 2.0
    half_col = (cols - CELL_GAP) * patch.pitch_col * MM / 2.0
    centre = project(patch.centre, hand)[0]
    return np.array([
        centre + sr * half_row * u_row + sc * half_col * u_col
        for sr, sc in ((-1, -1), (-1, 1), (1, 1), (1, -1))
    ])


def cell_centre(patch: Patch, hand: Hand, row: int, column: int) -> np.ndarray:
    u_row, u_col, _ = panel_axes(patch, hand)
    rows, cols = patch.id_grid.shape
    centre = project(patch.centre, hand)[0]
    offset_row = (row - (rows - 1) / 2.0) * patch.pitch_row * MM
    offset_col = (column - (cols - 1) / 2.0) * patch.pitch_col * MM
    return centre + offset_row * u_row + offset_col * u_col


# --------------------------------------------------------------------------
# checks -- a map that is wrong should refuse to render, not render quietly
# --------------------------------------------------------------------------

def polygons_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons."""
    for polygon in (first, second):
        for index in range(len(polygon)):
            edge = polygon[(index + 1) % len(polygon)] - polygon[index]
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            near, far = first @ normal, second @ normal
            if near.max() < far.min() or far.max() < near.min():
                return False
    return True


def distance_to_hull(point: np.ndarray, hull: np.ndarray) -> float:
    """Distance from a point to a convex polygon, 0 inside it."""
    inside = True
    nearest = np.inf
    for index in range(len(hull)):
        start = hull[index]
        edge = hull[(index + 1) % len(hull)] - start
        offset = point - start
        if edge[0] * offset[1] - edge[1] * offset[0] < 0.0:
            inside = False
        step = float(np.clip(offset @ edge / (edge @ edge), 0.0, 1.0))
        nearest = min(nearest, float(np.linalg.norm(offset - step * edge)))
    return 0.0 if inside else nearest


def check_panel_fidelity(hand: Hand, patch: Patch) -> None:
    """A flat, face-on panel must draw its cells on the real taxel positions.

    This is the check that ties everything together -- panel angle, mirroring
    and cell -> id assignment all have to be right for it to pass, and it is
    exact: all 22 flat pads across both hands land within 0.00 mm.

    It only applies where "draw it where it is" is achievable. The fingertips
    are curved domes flattened onto a 6x6 canvas (0.55-0.92 of a pitch out) and
    the unrolled thumb patches are rotated out of a degenerate projection on
    purpose, so both are excluded by construction rather than by a loose bound
    that would let a genuine mislabel through.
    """
    normal = np.cross(patch.axis_row, patch.axis_col)
    taxels = patch.id_grid[patch.id_grid >= 0]
    if float(np.ptp(hand.positions[taxels] @ normal)) > FLATNESS_TOLERANCE_MM / MM:
        return
    if panel_axes(patch, hand)[2]:
        return
    rows, cols = patch.id_grid.shape
    for row in range(rows):
        for column in range(cols):
            taxel = int(patch.id_grid[row, column])
            if taxel < 0:
                continue
            drawn = cell_centre(patch, hand, row, column)
            true = project(hand.positions[taxel], hand)[0]
            error = float(np.linalg.norm(drawn - true))
            if error > PANEL_FIDELITY_MM:
                raise ValueError(
                    f"{hand.key}: {patch.name} draws taxel {taxel} "
                    f"{error:.2f} mm from where it actually is"
                )


def check_hand(hand: Hand) -> None:
    ids = np.concatenate([p.id_grid[p.id_grid >= 0].ravel() for p in hand.patches])
    if not np.array_equal(np.sort(ids), np.arange(len(hand.positions))):
        raise ValueError(f"{hand.key}: ids do not cover 0..{len(hand.positions) - 1} once")

    corners = [panel_corners(patch, hand) for patch in hand.patches]
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            if polygons_overlap(corners[i], corners[j]):
                raise ValueError(
                    f"{hand.key}: panels {hand.patches[i].name} and "
                    f"{hand.patches[j].name} overlap; the map would be misleading"
                )

    # Each pad must land on the hand, but not exactly inside a link hull: taxels
    # sit on top of the sensor (4.4 mm for a 4x4, 29 mm for a fingertip), and on
    # an edge-on patch that standoff points sideways, nudging the centroid just
    # off the projected link. Measured, that costs 0.51 mm (sparsh) / 0.60 mm
    # (leapxela) and only ever on the unrolled patches. The tolerance is there to
    # absorb the standoff while still catching a pad on the wrong finger, which
    # would miss by tens of mm.
    scaled = [hull * MM for hull in hand.hulls]
    for patch in hand.patches:
        check_panel_fidelity(hand, patch)
        centre = project(patch.centre, hand)[0]
        gap = min(distance_to_hull(centre, hull) for hull in scaled)
        if gap > HULL_TOLERANCE_MM:
            raise ValueError(
                f"{hand.key}: patch {patch.name} sits {gap:.1f} mm off every "
                "link silhouette; the projection and the meshes disagree"
            )


# --------------------------------------------------------------------------
# svg
# --------------------------------------------------------------------------

def patch_colour(canonical: str, alpha: float) -> str:
    group, segment = canonical.split(".")
    return f"hsl({GROUP_HUE[group]} 62% {SEGMENT_LIGHT[segment]}% / {alpha})"


def caption_view_y(corners: np.ndarray, others: list, label: str,
                   font: float) -> float:
    """View-space y for a patch caption, on whichever side of it is free.

    Below reads best, but the palm pads stack with no gap, so a caption fixed
    below palm.centre lands on palm.lower and hides a row of ids. Above is tried
    next; if a patch is boxed in on both sides the caption stays below and
    relies on its halo.
    """
    half_width = 0.28 * font * max(len(label), 1)
    for offset in (-1.0, 1.0):
        edge = corners[:, 1].min() if offset < 0.0 else corners[:, 1].max()
        centre = edge + offset * font * 1.15
        box = np.array([
            [corners.mean(axis=0)[0] - half_width, centre - font * 0.6],
            [corners.mean(axis=0)[0] + half_width, centre - font * 0.6],
            [corners.mean(axis=0)[0] + half_width, centre + font * 0.6],
            [corners.mean(axis=0)[0] - half_width, centre + font * 0.6],
        ])
        if not any(polygons_overlap(box, other) for other in others):
            return centre
    return corners[:, 1].min() - font * 1.15


def hand_svg(hand: Hand) -> tuple:
    """(svg markup, taxel metadata dict) for one hand."""
    pieces = []
    outline = []
    for hull in hand.hulls:
        points = " ".join(f"{x * MM:.2f},{-y * MM:.2f}" for x, y in hull)
        outline.append(f'<polygon points="{points}"/>')
    pieces.append(f'<g class="outline">{"".join(outline)}</g>')

    corners_by_patch = [panel_corners(patch, hand) for patch in hand.patches]
    extents = [hull * MM * np.array([1.0, -1.0]) for hull in hand.hulls]
    extents.extend(corners * np.array([1.0, -1.0]) for corners in corners_by_patch)
    stacked = np.concatenate(extents)
    # Captions scale with the hand, not with the patch. Sized per patch, a 4x4
    # pad's caption came out a third of the palm's and neighbouring palm labels
    # collided.
    caption_font = float(np.ptp(stacked, axis=0).max()) * CAPTION_FRACTION

    metadata = {}
    captions = []
    for patch, corners in zip(hand.patches, corners_by_patch):
        rows, cols = patch.id_grid.shape
        _, _, unrolled = panel_axes(patch, hand)
        frame = " ".join(f"{x:.2f},{-y:.2f}" for x, y in corners)
        group = [
            f'<g class="patch" data-hand="{hand.key}" data-key="{patch.canonical}">',
            f'<polygon class="frame" points="{frame}" '
            f'fill="{patch_colour(patch.canonical, 0.16)}"/>',
        ]
        width = patch.pitch_col * MM * (1.0 - CELL_GAP)
        height = patch.pitch_row * MM * (1.0 - CELL_GAP)
        # Sized so three monospace digits fit across the narrower cell dimension.
        # Fitting the taller dimension instead let 3-digit ids on the 4.25 mm
        # pads run into their neighbours and read as one long number.
        font = min(width, height) * LABEL_DIGIT_FRACTION
        for row in range(rows):
            for column in range(cols):
                taxel = int(patch.id_grid[row, column])
                if taxel < 0:
                    continue
                x, y = cell_centre(patch, hand, row, column)
                group.append(
                    f'<rect class="cell" data-id="{taxel}" '
                    f'x="{x - width / 2:.2f}" y="{-y - height / 2:.2f}" '
                    f'width="{width:.2f}" height="{height:.2f}" rx="{width * 0.16:.2f}" '
                    f'fill="{patch_colour(patch.canonical, 0.92)}"/>'
                )
                group.append(
                    f'<text class="id" x="{x:.2f}" y="{-y:.2f}" '
                    f'font-size="{font:.2f}">{taxel}</text>'
                )
                position = hand.positions[taxel] * MM
                metadata[taxel] = {
                    "patch": patch.name,
                    "key": patch.canonical,
                    "rc": [row, column],
                    "xyz": [round(float(v), 1) for v in position],
                }
        group.append("</g>")
        pieces.append("".join(group))

        # SVG paints in document order and has no z-index, so a caption emitted
        # with its own patch ends up underneath whatever patch comes next --
        # palm.centre's label was hidden by the palm.lower panel below it.
        # Captions therefore go in a layer of their own after every panel, still
        # tagged with the patch key so highlighting reaches them.
        label = patch.canonical + (" ↻" if unrolled else "")
        others = [c for c in corners_by_patch if c is not corners]
        captions.append(
            f'<g class="patch" data-hand="{hand.key}" data-key="{patch.canonical}">'
            f'<text class="caption" x="{corners.mean(axis=0)[0]:.2f}" '
            f'y="{-caption_view_y(corners, others, label, caption_font):.2f}" '
            f'font-size="{caption_font:.2f}">{html.escape(label)}</text></g>'
        )
    pieces.extend(captions)

    # Captions hang below their panel, so the box has to allow for them.
    lower = stacked.min(axis=0) - caption_font * 2.0
    upper = stacked.max(axis=0) + caption_font * 2.5
    box = f"{lower[0]:.1f} {lower[1]:.1f} {upper[0] - lower[0]:.1f} {upper[1] - lower[1]:.1f}"
    svg = (
        f'<svg id="svg-{hand.key}" data-hand="{hand.key}" viewBox="{box}" '
        f'data-home="{box}" preserveAspectRatio="xMidYMid meet">'
        f'<g class="scene">{"".join(pieces)}</g></svg>'
    )
    return svg, metadata


PAGE_CSS = """
:root { color-scheme: light dark; --bg:#fbfbfa; --fg:#1c1b19; --muted:#6b6862;
        --line:#d9d6d0; --card:#ffffff; --skin:#cfcac1; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16151a; --fg:#eceae6; --muted:#9a958d; --line:#33313a;
          --card:#1e1d23; --skin:#3a3741; }
}
* { box-sizing: border-box; }
/* Column flex rather than a hardcoded height offset: the header wraps to a
   different number of lines depending on window width, and subtracting a fixed
   132px pushed the palm and thumb below the fold. */
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 ui-sans-serif,
       -apple-system, "Segoe UI", Roboto, sans-serif;
       height:100vh; display:flex; flex-direction:column; overflow:hidden; }
header { padding:14px 20px 10px; border-bottom:1px solid var(--line); flex:none; }
h1 { margin:0 0 4px; font-size:17px; font-weight:650; letter-spacing:-0.01em; }
.note { color:var(--muted); font-size:12.5px; max-width:105ch; margin:0; }
.controls { display:flex; gap:16px; align-items:center; flex-wrap:wrap; flex:none;
            padding:10px 20px; border-bottom:1px solid var(--line); }
.controls label { display:flex; gap:6px; align-items:center; font-size:13px;
                  color:var(--muted); cursor:pointer; }
input[type=search] { width:150px; padding:5px 9px; border:1px solid var(--line);
    border-radius:7px; background:var(--card); color:var(--fg); font:inherit; font-size:13px; }
button { padding:5px 11px; border:1px solid var(--line); border-radius:7px;
         background:var(--card); color:var(--fg); font:inherit; font-size:13px; cursor:pointer; }
button:hover { border-color:var(--muted); }
.hands { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--line);
         flex:1; min-height:0; }
/* min-height:0 is required, not tidiness: a grid item defaults to
   min-height:auto, so without it the svg's own size wins and the hand runs off
   the bottom of the page instead of scaling to fit. */
.hand { background:var(--bg); display:flex; flex-direction:column;
        min-width:0; min-height:0; }
.hand h2 { margin:0; padding:9px 16px 2px; font-size:14px; font-weight:620; }
.hand p { margin:0; padding:0 16px 7px; color:var(--muted); font-size:12px; }
svg { flex:1; min-height:0; width:100%; touch-action:none; cursor:grab; }
svg.dragging { cursor:grabbing; }
.outline { fill:var(--skin); stroke:none; }
.cell { stroke:rgba(0,0,0,0.28); stroke-width:0.14; }
.id { text-anchor:middle; dominant-baseline:central; fill:#12110f;
      font-family:ui-monospace,SFMono-Regular,Menlo,monospace; pointer-events:none; }
.caption { text-anchor:middle; dominant-baseline:central; fill:var(--muted); pointer-events:none;
           font-family:ui-sans-serif,system-ui,sans-serif;
           /* halo, so a caption stays readable where it crosses a neighbouring pad */
           paint-order:stroke; stroke:var(--bg); stroke-width:0.9; stroke-linejoin:round; }
.frame { stroke:var(--muted); stroke-width:0.25; stroke-dasharray:1.2 1.2; }
body.no-ids .id { display:none; }
body.no-outline .outline { display:none; }
body.no-captions .caption { display:none; }
.patch.dim { opacity:0.2; }
.patch.match .frame { stroke:var(--fg); stroke-width:0.9; stroke-dasharray:none; }
.cell.hit { stroke:var(--fg); stroke-width:0.7; }
#tip { position:fixed; pointer-events:none; z-index:9; opacity:0; transition:opacity .08s;
       background:var(--card); border:1px solid var(--line); border-radius:9px;
       padding:8px 11px; font-size:12.5px; box-shadow:0 6px 22px rgba(0,0,0,.17);
       font-family:ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre; }
#tip.on { opacity:1; }
.legend { display:flex; gap:13px; flex-wrap:wrap; padding:8px 20px; font-size:12px;
          color:var(--muted); border-top:1px solid var(--line); flex:none; }
.swatch { display:inline-block; width:10px; height:10px; border-radius:3px;
          margin-right:5px; vertical-align:-1px; }
"""

PAGE_JS = r"""
const TAXELS = __TAXELS__;
const tip = document.getElementById('tip');

// Pan/zoom is per hand, not shared: these are different robots at different
// scales, so a common transform would line nothing up.
document.querySelectorAll('svg').forEach(svg => {
  const box = () => svg.getAttribute('viewBox').split(' ').map(Number);
  svg.addEventListener('wheel', event => {
    event.preventDefault();
    const [x, y, w, h] = box();
    const rect = svg.getBoundingClientRect();
    const fx = (event.clientX - rect.left) / rect.width;
    const fy = (event.clientY - rect.top) / rect.height;
    const k = Math.exp(event.deltaY * 0.0016);
    const nw = Math.min(Math.max(w * k, 6), 4000);
    const nh = h * (nw / w);
    svg.setAttribute('viewBox', `${x + (w - nw) * fx} ${y + (h - nh) * fy} ${nw} ${nh}`);
  }, { passive: false });

  let from = null;
  svg.addEventListener('pointerdown', event => {
    from = { x: event.clientX, y: event.clientY, box: box() };
    svg.classList.add('dragging');
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener('pointermove', event => {
    if (!from) return;
    const rect = svg.getBoundingClientRect();
    const [x, y, w, h] = from.box;
    svg.setAttribute('viewBox',
      `${x - (event.clientX - from.x) * w / rect.width} ` +
      `${y - (event.clientY - from.y) * h / rect.height} ${w} ${h}`);
  });
  const release = () => { from = null; svg.classList.remove('dragging'); };
  svg.addEventListener('pointerup', release);
  svg.addEventListener('pointercancel', release);
});

function clearHighlight() {
  document.querySelectorAll('.patch').forEach(p => p.classList.remove('dim', 'match'));
  document.querySelectorAll('.cell').forEach(c => c.classList.remove('hit'));
}

// Highlighting a patch highlights its counterpart on the other hand. The two
// hands are structurally isomorphic patch for patch, which is the only reason
// a side-by-side comparison means anything.
function highlight(key) {
  document.querySelectorAll('.patch').forEach(p => {
    const hit = p.dataset.key === key;
    p.classList.toggle('match', hit);
    p.classList.toggle('dim', !hit);
  });
}

document.addEventListener('mouseover', event => {
  const cell = event.target.closest('.cell');
  if (!cell) return;
  const patch = cell.closest('.patch');
  const hand = patch.dataset.hand;
  const meta = TAXELS[hand][cell.dataset.id];
  highlight(patch.dataset.key);
  cell.classList.add('hit');
  tip.textContent =
    `taxel id  ${cell.dataset.id}\n` +
    `patch     ${meta.patch}\n` +
    `shared    ${meta.key}\n` +
    `cell      row ${meta.rc[0]}, col ${meta.rc[1]}\n` +
    `position  ${meta.xyz[0]}, ${meta.xyz[1]}, ${meta.xyz[2]} mm`;
  tip.classList.add('on');
});

document.addEventListener('mousemove', event => {
  if (!tip.classList.contains('on')) return;
  const pad = 15;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(event.clientX + pad, innerWidth - w - 6) + 'px';
  tip.style.top = Math.min(event.clientY + pad, innerHeight - h - 6) + 'px';
});

document.addEventListener('mouseout', event => {
  if (!event.target.closest('.cell')) return;
  if (event.relatedTarget && event.relatedTarget.closest('.cell')) return;
  tip.classList.remove('on');
  clearHighlight();
});

const search = document.getElementById('search');
search.addEventListener('input', () => {
  clearHighlight();
  const wanted = search.value.trim();
  if (wanted === '') return;
  document.querySelectorAll('.cell').forEach(cell => {
    if (cell.dataset.id === wanted) {
      cell.classList.add('hit');
      cell.closest('.patch').classList.add('match');
    } else {
      cell.closest('.patch').classList.add('dim');
    }
  });
});

document.querySelectorAll('input[type=checkbox]').forEach(box => {
  box.addEventListener('change', () =>
    document.body.classList.toggle(box.dataset.off, !box.checked));
});

document.getElementById('reset').addEventListener('click', () => {
  document.querySelectorAll('svg').forEach(svg =>
    svg.setAttribute('viewBox', svg.dataset.home));
  search.value = '';
  clearHighlight();
});
"""


def legend_markup() -> str:
    items = []
    for group in ("thumb", "index", "middle", "ring", "palm"):
        colour = patch_colour(f"{group}.tip" if group != "palm" else "palm.centre", 0.92)
        items.append(f'<span><i class="swatch" style="background:{colour}"></i>{group}</span>')
    items.append("<span>lighter = further out along the finger</span>")
    items.append("<span>↻ unrolled = patch faces sideways in this view, "
                 "drawn flat so its ids stay readable</span>")
    return f'<div class="legend">{"".join(items)}</div>'


def write_hand_map_html(path: Path, hands: tuple) -> None:
    columns = []
    metadata = {}
    for hand in hands:
        check_hand(hand)
        svg, taxels = hand_svg(hand)
        metadata[hand.key] = taxels
        columns.append(
            f'<div class="hand"><h2>{html.escape(hand.title)}</h2>'
            f'<p>{html.escape(hand.subtitle)}</p>{svg}</div>'
        )

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Taxel ID map &mdash; SparshSkin vs LeapXELA</title>
<style>{PAGE_CSS}</style></head><body>
<header>
  <h1>Taxel ID maps &mdash; 368 taxels per hand</h1>
  <p class="note">Link silhouettes are the real meshes; each sensor patch is drawn as a
  flat cell grid at its true position on the hand, one taxel id per cell. Different robots
  (Allegro vs LEAP), so positions are not comparable &mdash; but the patches are, one for one.
  Hover any taxel to highlight its counterpart on the other hand.</p>
</header>
<div class="controls">
  <input id="search" type="search" placeholder="find taxel id&hellip;" inputmode="numeric">
  <label><input type="checkbox" data-off="no-ids" checked>ids</label>
  <label><input type="checkbox" data-off="no-outline" checked>hand outline</label>
  <label><input type="checkbox" data-off="no-captions" checked>patch names</label>
  <button id="reset">reset view</button>
  <span class="note">scroll to zoom &middot; drag to pan</span>
</div>
<div class="hands">{"".join(columns)}</div>
{legend_markup()}
<div id="tip"></div>
<script>{PAGE_JS.replace("__TAXELS__", json.dumps(metadata, separators=(",", ":")))}</script>
</body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
