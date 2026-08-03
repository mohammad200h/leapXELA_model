"""
Heatmap rendering of packed LeapXELA taxel frames (T, 3, 26, 31).

Adapted from the mantaray collector's taxel-image output: shear channels are
signed (TURBO, symmetric limits), the normal channel unsigned (HOT, vmin=0);
cells without a hardware taxel (id_grid == -1) render black.
"""

from __future__ import annotations

import pathlib
from typing import Dict

import cv2
import numpy as np

CHANNEL_IMAGE_NAMES = ("shear_x", "shear_y", "normal_z")
UPSCALE = 12


def packed_channel_to_image(
    channel: np.ndarray,
    vmin: float,
    vmax: float,
    signed: bool,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """One packed (26, 31) channel -> RGB heatmap with empty cells black."""
    if vmax <= vmin:
        scaled = np.zeros_like(channel, dtype=np.uint8)
    else:
        scaled = np.clip((channel - vmin) / (vmax - vmin), 0.0, 1.0)
        scaled = (scaled * 255.0).astype(np.uint8)
    colormap = cv2.COLORMAP_TURBO if signed else cv2.COLORMAP_HOT
    image_bgr = cv2.applyColorMap(scaled, colormap)
    image_bgr[~valid_mask] = 0
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = channel.shape
    return cv2.resize(
        image_rgb, (width * UPSCALE, height * UPSCALE), interpolation=cv2.INTER_NEAREST
    )


def save_taxel_images(
    packed: np.ndarray,
    id_grid: np.ndarray,
    output_dir: pathlib.Path,
) -> Dict[str, str]:
    """Save per-frame PNG heatmaps per channel under output_dir/taxel_images."""
    if packed.size == 0:
        return {}
    valid_mask = id_grid >= 0
    taxel_root = output_dir / "taxel_images"
    taxel_root.mkdir(parents=True, exist_ok=True)
    saved_dirs: Dict[str, str] = {}

    for channel_idx in range(packed.shape[1]):
        channel_name = CHANNEL_IMAGE_NAMES[channel_idx]
        channel_dir = taxel_root / f"channel_{channel_idx:02d}_{channel_name}"
        channel_dir.mkdir(parents=True, exist_ok=True)

        channel_data = packed[:, channel_idx]
        signed = channel_idx != 2
        if signed:
            limit = float(np.max(np.abs(channel_data)))
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = 0.0, float(np.max(channel_data))

        for frame_idx, frame in enumerate(channel_data):
            image = packed_channel_to_image(frame, vmin, vmax, signed, valid_mask)
            cv2.imwrite(
                str(channel_dir / f"frame_{frame_idx:04d}.png"),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            )

        summary = np.max(np.abs(channel_data), axis=0) if signed else np.max(channel_data, axis=0)
        summary_image = packed_channel_to_image(summary, vmin, vmax, signed, valid_mask)
        cv2.imwrite(
            str(channel_dir / "summary_max.png"),
            cv2.cvtColor(summary_image, cv2.COLOR_RGB2BGR),
        )
        saved_dirs[f"channel_{channel_idx}"] = str(channel_dir.relative_to(output_dir))

    return saved_dirs
