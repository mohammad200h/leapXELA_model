The uSCu ALHA fingertip is **not** a regular rectangular grid like the phalanx 4×4 or palm 4×6 patches. It is a **curved 30-taxel array** with individually placed sites.

## Grid size


| Property                   | Value                                               |
| -------------------------- | --------------------------------------------------- |
| Taxels per tip             | **30**                                              |
| Tips on hand               | **4** (IF, MF, RF, TH)                              |
| Total fingertip taxels     | **120**                                             |
| Logical visualization grid | **6 × 6**, but only **30 cells filled**             |
| Regular pitch grid?        | **No** — unlike 4×4 (4.25×4.70 mm) or 4×6 (7.25 mm) |


The 6×6 layout used for heatmaps / Sparsh-style packing (same uSCu family as `aftc`):

```text
6×6 logical grid ( · = empty, numbers = taxel index in flat stream order)

  col:  0   1   2   3   4   5
row 0:  ·   ·   1   2   ·   ·      ← 2 taxels (tip apex)
row 1:  3   ·   4   5   ·   6      ← 4 taxels
row 2:  7   8   9  10  11  12      ← 
row 3: 13  14  15  16  17  18      ← 4×6 block
row 4: 19  20  21  22  23  24      ← (24 taxels)
row 5: 25  26  27  28  29  30      ←

Count: 2 + 4 + 24 = 30
```

That pattern comes from the Sparsh/XELA `aftc` packing code:

```504:508:jepa_tactile-/sparsh-multisensory-touch-main/tactile_ssl/data/xela/utils.py
        if "aftc" in sensor_type:
            sensor_grid = np.zeros((6, 6, 3))
            sensor_grid[2:, :] = einops.rearrange(sensor_[6:], "(h w) c -> h w c", h=4, w=6)
            sensor_grid[0, 2:-2] = sensor_[0:2]
            sensor_grid[1, 1:-1] = sensor_[2:6]
```

So the “grid size” is **6×6 canvas, 30 active taxels** — a rounded fingertip footprint, not 6×6 = 36.

---



## How LeapXela places them (simulation)

In this repo, fingertip taxels are **not** placed on a regular lattice. Each of the 30 gets its own calibrated 3D pose from `leapXela_pointcloud/fingertip_magnet_pose.json`:

```179:205:jepa_tactile-/leapxela/taxel_layout.py
def _fingertip_entries(...):
    base = magnet_pose[f"{finger}_pointcloud_base_frame"]
    base_pos = np.asarray(base["pos"], dtype=np.float64)
    base_quat = np.asarray(base["quat"], dtype=np.float64)
    tip_ids: List[int] = []
    for row in range(1, 7):
        tip_ids.extend(hw_tip[str(row)])
    for k, taxel_id in enumerate(tip_ids):
        rel = magnet_pose[str(k + 1)]
        entries.append(
            TaxelEntry(
                ...
                body=body,   # always {finger}_ds (distal link)
                pos=base_pos + quat_rotate(base_quat, rel["pos"]),
                quat=quat_mul(base_quat, rel["quat"]),
            )
        )
```

Placement chain:

1. **Body**: all 30 sites on the **distal link** — `if_ds`, `mf_ds`, `rf_ds`, or `th_ds`.
2. **Base frame**: `{finger}_pointcloud_base_frame` in the JSON (aligned to the LEAP fingertip / magnet mount).
3. **Per taxel**: keys `"1"` … `"30"` give **relative** `(pos, quat)` in that base frame.
4. **Hardware ID order**: rows 1–6 from `leap_sensor_taxel_map.json` → `{finger}.tip` (which hardware IDs belong to which row).

Each taxel is a MuJoCo `<site>` with viz radius **1.2 mm** — a point, not a physical cell box.

---



## Physical location and spacing



### Module envelope (hardware)

From the XELA uSCu spec:

- **Module size**: ~**31.4 × 28.8 × 39.3 mm**
- **Taxels**: 30 tri-axial sensors
- **Typical center-to-center spacing**: **~6.5 mm overall** (manufacturer spec; not uniform on a flat plane because the skin is curved)



### 3D layout (same sensor family — Allegro URDF reference)

The repo includes full 3D coordinates for all 30 uSCu/`aftc` taxels in `xela.xacro` (same curved fingertip sensor family). Taxels sit on a **curved dome**, not a flat sheet:

```text
Side view (schematic — z rises toward nail/contact side):

        z ≈ 25.7 mm  ─── center band (taxels 12–15, 18–21)  ← most protruding
        z ≈ 20.0 mm  ─── middle ring (taxels 6–9, 22–25)
        z ≈ 12.5 mm  ─── outer/base ring (taxels 1–5, 26–30)

Plan view (y = along finger, x = across finger):

              y ≈ 34 mm (distal tip)
                 1   2
               3 4 5 6
         7 8 9 10 11 12
        13 14 15 16 17 18    ← roughly 6 columns × 6 rows
        19 20 21 22 23 24       but wings curve in x
        25 26 27 28 29 30
              y ≈ 4 mm (proximal on tip)
     x:  -14 mm ←── 0 ──→ +14 mm
```

Representative distances from the URDF coordinates (Allegro `aftc_base_link`, meters):


| Neighbor pair             | Approx. distance                       |
| ------------------------- | -------------------------------------- |
| Taxel 4 ↔ 5 (same row, y) | **6.5 mm**                             |
| Taxel 3 ↔ 4 (next row)    | **~7.0 mm**                            |
| Taxel 6 ↔ 7 (middle row)  | **6.3 mm**                             |
| Left ↔ right wing (x)     | **~8–28 mm** (curved, not a flat grid) |
| Base ring ↔ center (z)    | **~7–13 mm** height difference         |


Along a row, spacing is close to the **~6.5 mm** spec. Across the curved surface, **3D** distance varies because taxels follow the fingertip curvature and each has its own orientation (`ry`, `rx`, `rz` per taxel).

---



## Comparison to flat patches


|                  | **uSCu ALHA (tip)**        | **uSPa 44 (4×4)**    | **uSPa 46 (4×6)**    |
| ---------------- | -------------------------- | -------------------- | -------------------- |
| Taxels           | 30                         | 16                   | 24                   |
| Layout           | Curved, irregular          | Regular 4×4          | Regular 4×6          |
| Pitch            | ~6.5 mm (average)          | 4.25 × 4.70 mm       | 7.25 × 7.25 mm       |
| Placement in sim | 30 individual poses (JSON) | Regular grid formula | Regular grid formula |
| Body             | `{finger}_ds`              | varies by patch      | palm                 |


---



## Summary

- **Grid size**: **30 taxels** per tip; visualized on a **6×6** grid with **6 empty cells** (2+4+24 pattern).
- **Placement**: on the **distal finger body** (`*_ds`), each site at a **calibrated (pos, quat)** from `fingertip_magnet_pose.json` — not a uniform pitch grid.
- **Spacing**: manufacturer spec **~6.5 mm** between neighboring taxels on average; actual 3D distances are **non-uniform** because the array wraps a curved fingertip (~31×29×39 mm module).

Exact numeric poses for the LEAP hand live in the gitignored `LeapXELA_Hardware_ws-main/leapXela_pointcloud/fingertip_magnet_pose.json`. With that workspace present, you can dump all 30 `(x,y,z)` positions per finger via `build_layout()`.