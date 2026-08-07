"""Live taxel heatmap rendering: 3D scene markers and a 2D palm-view panel.

Both outputs are driven from the same per-taxel readings, so the markers on the
hand and the flat panel always agree. The panel is rasterised through a
precomputed label image (pixel -> taxel index), which turns each frame into a
single lookup instead of a matplotlib redraw.
"""

import cv2
import mujoco as mj
import numpy as np


CHANNELS = ("normal", "shear-x", "shear-y", "magnitude")
CHANNEL_COLUMN = {"normal": 2, "shear-x": 0, "shear-y": 1}
SIGNED_CHANNELS = ("shear-x", "shear-y")
PANEL_BACKGROUND = (24, 24, 28)
PANEL_MARGIN = 0.04
MARKER_RADIUS = 0.0026
MARKER_MIN_ALPHA = 0.65
SKIN_DIMMED_RGBA = (0.55, 0.57, 0.60, 0.55)
LUT_STEPS = 256
# Bare point-cloud view: no hand for scale, so the markers are drawn larger and
# force vectors get an arrow whose length is proportional to magnitude.
CLOUD_MARKER_RADIUS = 0.0032
ARROW_LENGTH = 0.045
ARROW_WIDTH = 0.0016
ARROW_MIN_FRACTION = 0.04


def channel_values(readings: np.ndarray, channel: str) -> np.ndarray:
    """Reduce (n, 3) taxel readings to the scalar shown for `channel`."""
    if channel == "magnitude":
        return np.linalg.norm(readings, axis=1)
    return readings[:, CHANNEL_COLUMN[channel]]


def _ramp(stops: np.ndarray, positions: np.ndarray, samples: np.ndarray):
    return np.stack([
        np.interp(samples, positions, stops[:, channel]) for channel in range(3)
    ], axis=1)


def _sequential_lut() -> np.ndarray:
    """Idle slate -> red -> yellow-white, so mid forces stay distinguishable."""
    samples = np.linspace(0.0, 1.0, LUT_STEPS)
    stops = np.array([
        [0.28, 0.30, 0.36],
        [0.85, 0.16, 0.10],
        [1.00, 0.85, 0.35],
    ])
    return _ramp(stops, np.array([0.0, 0.55, 1.0]), samples)


def _diverging_lut() -> np.ndarray:
    """Blue <- idle slate -> red for signed shear channels."""
    samples = np.linspace(-1.0, 1.0, LUT_STEPS)
    stops = np.array([
        [0.25, 0.55, 1.00],
        [0.28, 0.30, 0.36],
        [1.00, 0.30, 0.18],
    ])
    return _ramp(stops, np.array([-1.0, 0.0, 1.0]), samples)


SEQUENTIAL_LUT = _sequential_lut()
DIVERGING_LUT = _diverging_lut()


def colour_lut(channel: str) -> np.ndarray:
    return DIVERGING_LUT if channel in SIGNED_CHANNELS else SEQUENTIAL_LUT


def lut_indices(values: np.ndarray, channel: str, scale: float) -> np.ndarray:
    """Map forces to LUT rows: [-scale, scale] signed, [0, scale] unsigned."""
    if channel in SIGNED_CHANNELS:
        normalised = 0.5 * (np.clip(values / scale, -1.0, 1.0) + 1.0)
    else:
        normalised = np.clip(values / scale, 0.0, 1.0)
    return np.clip(
        (normalised * (LUT_STEPS - 1)).astype(np.int32), 0, LUT_STEPS - 1
    )


def marker_colours(
    values: np.ndarray, channel: str, scale: float
) -> np.ndarray:
    """(n, 4) RGBA for the 3D markers; idle taxels fade out so they declutter."""
    colours = colour_lut(channel)[lut_indices(values, channel, scale)]
    if channel in SIGNED_CHANNELS:
        strength = np.abs(np.clip(values / scale, -1.0, 1.0))
    else:
        strength = np.clip(values / scale, 0.0, 1.0)
    alpha = MARKER_MIN_ALPHA + (1.0 - MARKER_MIN_ALPHA) * strength
    return np.column_stack([colours, alpha]).astype(np.float32)


def dim_flex_skins(model: mj.MjModel) -> None:
    """Mute the membrane colour so the heatmap markers read against it."""
    model.flex_rgba[:] = SKIN_DIMMED_RGBA


def panel_taxel_ids(skins: list, tips: list) -> dict:
    """Taxel id per layout cell, row-major, -1 for the empty tip canvas cells."""
    ids_by_skin = {}
    for skin in skins:
        ids_by_skin[skin.definition.geom_name] = np.asarray(skin.taxel_ids)
    for tip in tips:
        cells = np.full(36, -1, dtype=np.int64)
        for index, (row, column) in enumerate(tip.grid_rowcol):
            cells[row * 6 + column] = tip.taxel_ids[index]
        ids_by_skin[f"{tip.finger}_tip"] = cells
    return ids_by_skin


def taxel_positions(model: mj.MjModel, data: mj.MjData, sensors: list) -> tuple:
    """Live world positions and matching taxel ids for every flex vertex."""
    positions, ids = [], []
    for flex_name, taxel_ids in sensors:
        flex_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_FLEX, flex_name)
        address = int(model.flex_vertadr[flex_id])
        count = int(model.flex_vertnum[flex_id])
        positions.append(data.flexvert_xpos[address : address + count])
        ids.append(np.asarray(taxel_ids))
    return np.concatenate(positions), np.concatenate(ids)


def draw_taxel_markers(
    scene: mj.MjvScene,
    positions: np.ndarray,
    colours: np.ndarray,
    radius: float,
) -> int:
    """Append one sphere per taxel to `scene`; returns how many were drawn.

    Call after `update_scene` (which resets `ngeom`) and before rendering.
    """
    size = np.array([radius, radius, radius])
    identity = np.eye(3).ravel()
    drawn = 0
    for index in range(len(positions)):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            geom,
            mj.mjtGeom.mjGEOM_SPHERE,
            size,
            positions[index],
            identity,
            colours[index].astype(np.float32),
        )
        geom.category = int(mj.mjtCatBit.mjCAT_DECOR)
        # Emissive and matte: the marker must read as its heatmap colour, not
        # as a specular highlight of the scene lighting.
        geom.emission = 0.35
        geom.specular = 0.0
        geom.shininess = 0.0
        geom.reflectance = 0.0
        scene.ngeom += 1
        drawn += 1
    return drawn


def bare_scene_option() -> mj.MjvOption:
    """Scene option that renders nothing but decor, for the point-cloud view.

    Hiding geom groups is not enough: flexes are drawn independently of
    `geomgroup`, so the membranes would still occlude the cloud.
    """
    option = mj.MjvOption()
    option.geomgroup[:] = 0
    for flag in (
        mj.mjtVisFlag.mjVIS_FLEXFACE,
        mj.mjtVisFlag.mjVIS_FLEXSKIN,
        mj.mjtVisFlag.mjVIS_FLEXEDGE,
        mj.mjtVisFlag.mjVIS_FLEXVERT,
    ):
        option.flags[flag] = 0
    return option


def draw_taxel_arrows(
    scene: mj.MjvScene,
    positions: np.ndarray,
    vectors: np.ndarray,
    colours: np.ndarray,
    scale: float,
) -> int:
    """Draw one arrow per loaded taxel, length proportional to |vector|."""
    magnitudes = np.linalg.norm(vectors, axis=1)
    drawn = 0
    for index in np.where(magnitudes > ARROW_MIN_FRACTION * scale)[0]:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(
            geom,
            mj.mjtGeom.mjGEOM_ARROW,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            colours[index].astype(np.float32),
        )
        mj.mjv_connector(
            geom,
            int(mj.mjtGeom.mjGEOM_ARROW),
            ARROW_WIDTH,
            positions[index],
            positions[index] + vectors[index] / scale * ARROW_LENGTH,
        )
        geom.category = int(mj.mjtCatBit.mjCAT_DECOR)
        geom.emission = 0.35
        geom.specular = 0.0
        scene.ngeom += 1
        drawn += 1
    return drawn


class TaxelPanel:
    """Rasterises the palm-view layout into a reusable pixel label image."""

    def __init__(
        self,
        layout: dict,
        taxel_ids_by_skin: dict,
        width: int,
        height: int,
        taxel_count: int,
    ):
        self.width = width
        self.height = height
        self.taxel_count = taxel_count
        corners_x = np.concatenate(
            [corners[0].ravel() for corners in layout.values()]
        )
        corners_y = np.concatenate(
            [corners[1].ravel() for corners in layout.values()]
        )
        span_x = float(corners_x.max() - corners_x.min())
        span_y = float(corners_y.max() - corners_y.min())
        pad_x = PANEL_MARGIN * span_x
        pad_y = PANEL_MARGIN * span_y
        self.min_x = float(corners_x.min()) - pad_x
        self.min_y = float(corners_y.min()) - pad_y
        scale = min(
            (width - 1) / (span_x + 2.0 * pad_x),
            (height - 1) / (span_y + 2.0 * pad_y),
        )
        self.scale = scale
        self.labels = np.full((height, width), -1, dtype=np.int32)
        for name, (cell_x, cell_y) in layout.items():
            ids = taxel_ids_by_skin[name]
            rows, columns = cell_x.shape[0] - 1, cell_x.shape[1] - 1
            for row in range(rows):
                for column in range(columns):
                    quad = np.array([
                        [cell_x[row, column], cell_y[row, column]],
                        [cell_x[row + 1, column], cell_y[row + 1, column]],
                        [cell_x[row + 1, column + 1], cell_y[row + 1, column + 1]],
                        [cell_x[row, column + 1], cell_y[row, column + 1]],
                    ])
                    taxel = ids[row * columns + column]
                    if taxel < 0:
                        continue
                    self._fill(quad, int(taxel))

    def _to_pixels(self, points: np.ndarray) -> np.ndarray:
        pixels = np.empty_like(points)
        pixels[:, 0] = (points[:, 0] - self.min_x) * self.scale
        # Flip y so +palm z (fingertips) is up in the image.
        pixels[:, 1] = self.height - 1 - (points[:, 1] - self.min_y) * self.scale
        return pixels

    def _fill(self, quad: np.ndarray, taxel: int) -> None:
        polygon = np.round(self._to_pixels(quad)).astype(np.int32)
        cv2.fillConvexPoly(self.labels, polygon, taxel, lineType=cv2.LINE_8)

    def render(
        self, values: np.ndarray, channel: str, scale: float
    ) -> np.ndarray:
        """Colour the layout by `values` (one entry per taxel id)."""
        lut = (colour_lut(channel) * 255.0).astype(np.uint8)
        indices = lut_indices(values, channel, scale)
        palette = np.vstack([np.array(PANEL_BACKGROUND, dtype=np.uint8),
                             lut[indices]])
        image = palette[self.labels + 1]
        label = f"{channel}  +/-{scale:.2f} N" if channel in SIGNED_CHANNELS \
            else f"{channel}  0-{scale:.2f} N"
        cv2.putText(
            image, label, (8, self.height - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (235, 235, 235), 1, cv2.LINE_AA,
        )
        return image


def paste_panel(frame: np.ndarray, panel: np.ndarray, margin: int) -> np.ndarray:
    """Composite the panel into the lower-left corner of a rendered frame."""
    out = frame.copy()
    height, width = panel.shape[:2]
    top = out.shape[0] - height - margin
    out[top : top + height, margin : margin + width] = panel
    return out
