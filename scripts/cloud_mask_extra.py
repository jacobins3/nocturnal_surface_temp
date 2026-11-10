# Step 6. Additional LST-based cloud mask for cloud missed by the UDM (applied
# to the industrial scene); preview first, then --apply to write the cloud bit
# into the QA raster.

from __future__ import annotations

import proj_fix  # noqa: F401  (must precede rasterio)
import logging
import shutil
from pathlib import Path
from config import DATA_ROOT

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy import ndimage as ndi

import analysis_task3_building_contrast as t3

logger = logging.getLogger(__name__)

QA_CLOUD = 1 << 3

# ---- tunables --------------------------------------------------------------
COLD_K = 3.0           # seed: robust z-score below which LST counts as cold
GROW_K = 1.5           # grow seeds through connected pixels below this z
ALBEDO_CLOUD = 0.45    # broadband albedo above which a pixel counts as bright
BUFFER_M = 60.0        # dilation of the final mask (edges + shadow offset)
CORROB_M = 100.0       # cold LST must have bright albedo within this distance
MIN_BLOB_PX = 20       # drop connected components smaller than this (10 m px)
REQUIRE_BOTH = True    # cold AND (nearby) bright; False = cold OR bright
PER_SCENE = {          # per-scene overrides, e.g. {"AD_industrial": {"COLD_K": 2.5}}
    # robust SD of this scene is ~8.5 C (very heterogeneous), so z-scores
    # cannot reach the cold tail: use absolute thresholds (deg C) instead
    "AD_industrial": {"COLD_ABS_C": 4.0, "GROW_ABS_C": 6.0, "BUFFER_M": 20.0},
    "AD_suburban": {},
    "DXB_urban": {},
}
# Optional hand-drawn polygons per scene, in the scene's analysis CRS
# (list of lists of (x, y) vertices). Leave empty unless the automatic mask
# misses an obvious patch.
MANUAL_POLYGONS: dict[str, list[list[tuple[float, float]]]] = {}
# ----------------------------------------------------------------------------


def _disk(r_px: int) -> np.ndarray:
    y, x = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
    return (x * x + y * y) <= r_px * r_px


def _polygon_mask(polys, transform, shape):
    from rasterio.features import rasterize
    if not polys:
        return np.zeros(shape, bool)
    geoms = [{"type": "Polygon", "coordinates": [p]} for p in polys]
    return rasterize([(g, 1) for g in geoms], out_shape=shape,
                     transform=transform, fill=0, dtype="uint8").astype(bool)


def build_mask(key: str, scene):
    import xarray as xr
    par = dict(COLD_K=COLD_K, GROW_K=GROW_K, ALBEDO_CLOUD=ALBEDO_CLOUD, BUFFER_M=BUFFER_M,
               CORROB_M=CORROB_M, MIN_BLOB_PX=MIN_BLOB_PX,
               REQUIRE_BOTH=REQUIRE_BOTH)
    par.update(PER_SCENE.get(key, {}))

    transform, crs, h, w = t3.load_grid(scene.predictor_nc)
    px = abs(transform.a)
    lst = t3.load_raster_on_grid(scene.lst_tif, transform, crs, (h, w),
                                 Resampling.bilinear)
    qa = t3.load_raster_on_grid(scene.qa_tif, transform, crs, (h, w),
                                Resampling.nearest)
    qa = np.nan_to_num(qa, nan=t3.QA_NODATA).astype(np.int32)
    ds = xr.open_dataset(scene.predictor_nc)
    albedo = ds["albedo"].values.astype(np.float32)
    ds.close()

    base_ok = np.isfinite(lst) & ((qa & t3.QA_EXCLUDE) == 0)
    med = np.nanmedian(lst[base_ok])
    mad = np.nanmedian(np.abs(lst[base_ok] - med)) * 1.4826
    cold_thr = par.get("COLD_ABS_C", med - par["COLD_K"] * mad)
    grow_thr = par.get("GROW_ABS_C", med - par["GROW_K"] * mad)
    logger.info("%s LST percentiles 0.5/1/2/5/10/25/50: %s", key,
                np.round(np.nanpercentile(lst[base_ok],
                                          [0.5, 1, 2, 5, 10, 25, 50]), 1))
    cold = base_ok & (lst < cold_thr)
    bright = np.isfinite(albedo) & (albedo > par["ALBEDO_CLOUD"])

    # LST-only hysteresis: the Sentinel-2 albedo cloud is from a different
    # (daytime) date, so it says nothing about the night-time HotSat cloud.
    # Seeds: robust z < -COLD_K; grown through connected pixels < -GROW_K.
    grow = base_ok & (lst < grow_thr)
    lab, n = ndi.label(grow)
    cand = np.zeros_like(cold)
    if n:
        keep = np.unique(lab[cold & (lab > 0)])
        cand = np.isin(lab, keep[keep > 0])

    manual = _polygon_mask(MANUAL_POLYGONS.get(key, []), transform, (h, w))
    cand = cand | manual

    lab, n = ndi.label(cand)
    if n:
        sizes = ndi.sum(cand, lab, index=np.arange(1, n + 1))
        small = np.isin(lab, np.where(sizes < par["MIN_BLOB_PX"])[0] + 1)
        cand = cand & ~small
    r_b = max(1, int(round(par["BUFFER_M"] / px)))
    mask = ndi.binary_dilation(cand, structure=_disk(r_b))

    stats = {"scene": key, "median_C": float(med), "robust_sd_C": float(mad),
             "cold_thr_C": float(cold_thr), "grow_thr_C": float(grow_thr),
             "cold_px": int(cold.sum()), "bright_px": int(bright.sum()),
             "mask_px": int(mask.sum()),
             "mask_frac_of_valid": float((mask & base_ok).sum() / base_ok.sum())}
    return mask, lst, albedo, base_ok, (transform, crs, h, w), stats


def _diagnostic(key, mask, lst, albedo, base_ok, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    v = lst[base_ok]
    vmin, vmax = np.nanpercentile(v, [0.5, 99.5])
    ax[0].imshow(np.where(base_ok, lst, np.nan), cmap="inferno", vmin=vmin,
                 vmax=vmax)
    ax[0].contour(mask, levels=[0.5], colors="cyan", linewidths=0.8)
    ax[0].set_title("LST + proposed mask")
    ax[1].imshow(albedo, cmap="viridis", vmin=0, vmax=0.6)
    ax[1].contour(mask, levels=[0.5], colors="red", linewidths=0.8)
    ax[1].set_title("albedo + proposed mask")
    ax[2].imshow(np.where(mask, np.nan, np.where(base_ok, lst, np.nan)),
                 cmap="inferno", vmin=vmin, vmax=vmax)
    ax[2].set_title("LST after masking")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(key)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _write_qa(scene, mask, grid, out_qa):
    transform, crs, h, w = grid
    with rasterio.open(scene.qa_tif) as src:
        prof = src.profile.copy()
        qa = src.read(1)
        # mask (analysis grid) -> HotSat grid, nearest
        dst = np.zeros(qa.shape, np.uint8)
        reproject(mask.astype(np.uint8), dst, src_transform=transform,
                  src_crs=crs, dst_transform=src.transform, dst_crs=src.crs,
                  resampling=Resampling.nearest)
    new = (qa.astype(np.int32) | np.where(dst > 0, QA_CLOUD, 0)).astype(qa.dtype)
    prof.update(driver="GTiff", compress="deflate")
    with rasterio.open(out_qa, "w", **prof) as dstf:
        dstf.write(new, 1)


def run(keys=None, apply: bool = False):
    import sys
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        hnd = logging.StreamHandler(sys.stdout)
        hnd.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(hnd)
    out_dir = t3.DATA_ROOT / "analysis" / "cloud_mask_extra"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, scene in t3.SCENES.items():
        if keys and key not in keys:
            continue
        mask, lst, albedo, base_ok, grid, st = build_mask(key, scene)
        transform, crs, h, w = grid
        _diagnostic(key, mask, lst, albedo, base_ok,
                    out_dir / f"{key}_cloud_extra.png")
        with rasterio.open(out_dir / f"{key}_cloud_extra_10m.tif", "w",
                           driver="GTiff", height=h, width=w, count=1,
                           dtype="uint8", crs=crs, transform=transform,
                           compress="deflate") as dst:
            dst.write(mask.astype(np.uint8), 1)
        logger.info("%s: %s", key, st)
        rows.append(st)
        if apply:
            orig = Path(scene.qa_tif).with_name(
                Path(scene.qa_tif).stem + "_orig.tif")
            if not orig.exists():                 # never overwrite the backup
                shutil.copy(scene.qa_tif, orig)
            src_qa = orig                         # always OR onto the original
            tmp_scene = type("S", (), {"qa_tif": src_qa})
            _write_qa(tmp_scene, mask, grid, scene.qa_tif)
            logger.info("%s: QA updated -> %s (backup %s)", key,
                        scene.qa_tif, orig.name)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table_cloud_mask_extra.csv", index=False)
    logger.info("written -> %s", out_dir)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", help="e.g. AD_industrial (default: all)")
    ap.add_argument("--apply", action="store_true",
                    help="write the mask into the QA rasters (default: preview)")
    a = ap.parse_args()
    run(a.scenes or None, apply=a.apply)
