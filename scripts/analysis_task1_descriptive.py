# Step 8. Task 1: descriptive statistics of night-time SBT per scene, segment
# and land-cover class (Table 2, Table A2, Figure 3).

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

from config import DATA_ROOT  # set via UAE_SBT_DATA_ROOT (config.py)
OUTPUT_DIR = DATA_ROOT / "analysis" / "task1_descriptive"


@dataclass
class Scene:
    key: str
    scene_id: str
    typology: str
    view_zenith_deg: float
    predictor_nc: Path
    lst_tif: Path
    qa_tif: Path


def _scene(key, scene_id, typology, vza) -> Scene:
    return Scene(
        key=key, scene_id=scene_id, typology=typology,
        view_zenith_deg=vza,
        predictor_nc=DATA_ROOT / "predictors" /
        f"{scene_id}_predictors_10m.nc",
        lst_tif=DATA_ROOT / "lst" / f"{scene_id}_LST_C.tif",
        qa_tif=DATA_ROOT / "lst" / f"{scene_id}_LST_QA.tif",
    )


SCENES = {
    "AD_suburban":   _scene("AD_suburban",
                            "20231029_213936_HS1_SBT", "sub-urban", 34.21),
    "AD_industrial": _scene("AD_industrial",
                            "20231030_212656_HS1_SBT", "industrial", 0.85),
    "DXB_urban":     _scene("DXB_urban",
                            "20231120_213935_HS1_SBT", "urban", 35.43),
}

# QA bit values (must match hotsat_lst.py)
QA_NODATA      = 1 << 0
QA_EPS_LOW     = 1 << 1
QA_OBLIQUE     = 1 << 2          # whole-scene flag -- reported, not excluded
QA_CLOUD       = 1 << 3
QA_HOT_ANOMALY = 1 << 4
QA_EXCLUDE = QA_NODATA | QA_EPS_LOW | QA_CLOUD     # excluded from main stats

# ESA WorldCover class codes and labels
WC_LABELS = {
    10: "tree", 20: "shrub", 30: "grassland", 40: "cropland",
    50: "built-up", 60: "bare/sparse", 70: "snow/ice",
    80: "water", 90: "wetland", 95: "mangrove", 100: "moss/lichen",
}

# A pixel counts as "building" if the WSF-3D building fraction exceeds this.
# fp_density is a 0-1 field; 0.25 picks pixels that are at least a quarter
# built. Tune if the building-pixel share looks implausible per scene.
BUILDING_FP_THRESHOLD = 0.25


# ===========================================================================
# Raster IO
# ===========================================================================

def load_grid_and_layer(predictor_nc: Path, layer: str):
    import xarray as xr

    ds = xr.open_dataset(predictor_nc)
    arr = ds[layer].values.astype(np.float32)
    transform = rasterio.transform.Affine(*ds.attrs["transform"])
    crs = ds.attrs["crs"]
    ds.close()
    return arr, transform, crs, arr.shape


def load_raster_on_grid(path: Path, transform, crs, shape,
                        resampling=Resampling.bilinear) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs
        src_nodata = src.nodata
    if src_nodata is not None and not np.isnan(src_nodata):
        arr = np.where(arr == src_nodata, np.nan, arr)
    dst = np.full(shape, np.nan, dtype=np.float32)
    reproject(
        source=arr, destination=dst,
        src_transform=src_transform, src_crs=src_crs, src_nodata=np.nan,
        dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


# ===========================================================================
# Statistics
# ===========================================================================

def describe(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {k: np.nan for k in
                ("n", "mean", "std", "min", "p1", "p5", "median",
                 "p95", "p99", "max", "iqr")}
    p1, p5, p25, p50, p75, p95, p99 = np.percentile(
        v, [1, 5, 25, 50, 75, 95, 99])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "p1": float(p1), "p5": float(p5),
        "median": float(p50),
        "p95": float(p95), "p99": float(p99),
        "max": float(v.max()),
        "iqr": float(p75 - p25),
    }


def analyse_scene(scene: Scene) -> dict:
    logger.info("=== %s (%s) ===", scene.key, scene.typology)

    # --- load LST, QA, land cover, building fraction on a common grid -
    ndvi, transform, crs, shape = load_grid_and_layer(
        scene.predictor_nc, "ndvi")            # used only to get the grid
    wc, _, _, _ = load_grid_and_layer(scene.predictor_nc, "wc_class")
    fp, _, _, _ = load_grid_and_layer(scene.predictor_nc, "fp_density")

    lst = load_raster_on_grid(scene.lst_tif, transform, crs, shape,
                              Resampling.bilinear)
    qa = load_raster_on_grid(scene.qa_tif, transform, crs, shape,
                             Resampling.nearest)
    qa = np.nan_to_num(qa, nan=QA_NODATA).astype(np.int32)

    # --- pixel masks --------------------------------------------------
    finite_lst = np.isfinite(lst)
    excluded = (qa & QA_EXCLUDE) != 0
    hot = (qa & QA_HOT_ANOMALY) != 0
    # main analysis pixels: finite LST, not QA-excluded, not hot-anomaly
    main = finite_lst & ~excluded & ~hot
    oblique = bool(scene.view_zenith_deg > 30.0)

    n_total = int(finite_lst.sum())
    n_main = int(main.sum())
    n_hot = int((hot & finite_lst).sum())
    logger.info("  finite LST pixels: %d", n_total)
    logger.info("  main-analysis pixels: %d (%.1f%% of finite)",
                n_main, 100.0 * n_main / max(n_total, 1))
    logger.info("  hot-anomaly pixels: %d (%.2f%% of finite)",
                n_hot, 100.0 * n_hot / max(n_total, 1))
    logger.info("  oblique acquisition (VZA %.1f deg): %s",
                scene.view_zenith_deg, oblique)

    results = {
        "scene": scene.key,
        "typology": scene.typology,
        "view_zenith_deg": scene.view_zenith_deg,
        "oblique": oblique,
        "n_finite": n_total,
        "n_main": n_main,
        "n_hot_anomaly": n_hot,
        "hot_anomaly_pct": 100.0 * n_hot / max(n_total, 1),
    }

    # --- overall distribution (main pixels) ---------------------------
    overall = describe(lst[main])
    results["overall"] = overall
    logger.info("  LST (main): median %.2f C, IQR %.2f, "
                "p1-p99 %.1f to %.1f C",
                overall["median"], overall["iqr"],
                overall["p1"], overall["p99"])

    # --- hot-anomaly population (reported separately) -----------------
    results["hot_anomaly"] = describe(lst[hot & finite_lst])

    # --- segmentation: building vs non-building -----------------------
    is_building = np.isfinite(fp) & (fp >= BUILDING_FP_THRESHOLD)
    bld = main & is_building
    nonbld = main & ~is_building
    results["building"] = describe(lst[bld])
    results["non_building"] = describe(lst[nonbld])
    contrast = (results["building"]["median"] -
                results["non_building"]["median"])
    results["building_minus_nonbuilding_median"] = float(contrast)
    logger.info("  building vs non-building median contrast: "
                "%+.2f C (building %d px, non-building %d px)",
                contrast, results["building"]["n"],
                results["non_building"]["n"])

    # --- segmentation: by land-cover class ----------------------------
    wc_round = np.round(wc).astype(np.int32)
    by_lc = {}
    for code, label in WC_LABELS.items():
        cls_mask = main & (wc_round == code)
        if int(cls_mask.sum()) >= 100:        # ignore negligible classes
            by_lc[label] = describe(lst[cls_mask])
    results["by_landcover"] = by_lc
    if by_lc:
        logger.info("  land-cover classes present (>=100 px): %s",
                    ", ".join(by_lc.keys()))

    # keep the raw LST sample for the figure
    results["_lst_main"] = lst[main]
    return results


# ===========================================================================
# Outputs
# ===========================================================================

def write_tables(all_results: list[dict], output_dir: Path) -> None:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Table 2: one row per scene x segment -------------------------
    rows = []
    for r in all_results:
        for seg in ("overall", "building", "non_building", "hot_anomaly"):
            d = r[seg]
            rows.append({
                "scene": r["scene"], "typology": r["typology"],
                "segment": seg,
                "n": d["n"], "mean_C": d["mean"], "std_C": d["std"],
                "min_C": d["min"], "p1_C": d["p1"], "p5_C": d["p5"],
                "median_C": d["median"], "p95_C": d["p95"],
                "p99_C": d["p99"], "max_C": d["max"], "iqr_C": d["iqr"],
            })
    df = pd.DataFrame(rows)
    t2 = output_dir / "table2_thermal_statistics.csv"
    df.to_csv(t2, index=False, float_format="%.3f")
    logger.info("wrote %s", t2)

    # --- land-cover breakdown -----------------------------------------
    lc_rows = []
    for r in all_results:
        for label, d in r["by_landcover"].items():
            lc_rows.append({
                "scene": r["scene"], "typology": r["typology"],
                "landcover": label,
                "n": d["n"], "median_C": d["median"],
                "iqr_C": d["iqr"], "p5_C": d["p5"], "p95_C": d["p95"],
            })
    lc = output_dir / "table2_by_landcover.csv"
    pd.DataFrame(lc_rows).to_csv(lc, index=False, float_format="%.3f")
    logger.info("wrote %s", lc)


def write_figure(all_results: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = [r["_lst_main"] for r in all_results]
    labels = [f"{r['typology']}\n(n={r['n_main']:,})"
              for r in all_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # violin
    parts = ax1.violinplot(samples, showmedians=True, showextrema=False)
    ax1.set_xticks(range(1, len(labels) + 1))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Night SBT (deg C)")
    ax1.set_title("Per-typology night-time SBT distribution")
    ax1.grid(axis="y", alpha=0.3)

    # overlaid histograms
    for r in all_results:
        ax2.hist(r["_lst_main"], bins=80, histtype="step",
                 density=True, linewidth=1.6, label=r["typology"])
    ax2.set_xlabel("Night SBT (deg C)")
    ax2.set_ylabel("Probability density")
    ax2.set_title("Night SBT histograms")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("HotSat-1 nocturnal land-surface temperature, "
                 "three UAE urban typologies (autumn 2023)")
    fig.tight_layout()
    out = output_dir / "figure3_lst_distributions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


def write_summary(all_results: list[dict], output_dir: Path) -> None:
    lines = ["HotSat-1 nocturnal SUHI -- Task 1 descriptive statistics",
             "=" * 60, ""]
    for r in all_results:
        o = r["overall"]
        lines += [
            f"{r['typology'].upper()}  ({r['scene']})",
            f"  view zenith {r['view_zenith_deg']:.1f} deg "
            f"(oblique: {r['oblique']})",
            f"  main-analysis pixels : {r['n_main']:,}",
            f"  LST median / IQR     : {o['median']:.2f} / "
            f"{o['iqr']:.2f} C",
            f"  LST p1 - p99         : {o['p1']:.2f} - {o['p99']:.2f} C",
            f"  building median      : {r['building']['median']:.2f} C "
            f"(n={r['building']['n']:,})",
            f"  non-building median  : "
            f"{r['non_building']['median']:.2f} C "
            f"(n={r['non_building']['n']:,})",
            f"  building - nonbuilding contrast : "
            f"{r['building_minus_nonbuilding_median']:+.2f} C",
            f"  hot-anomaly pixels   : {r['n_hot_anomaly']:,} "
            f"({r['hot_anomaly_pct']:.2f}%)",
            "",
        ]
    out = output_dir / "task1_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", out)
    print("\n".join(lines))


# ===========================================================================
# Driver
# ===========================================================================

def run(output_dir: Path = OUTPUT_DIR):
    import sys
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(handler)

    output_dir = Path(output_dir)
    all_results = [analyse_scene(s) for s in SCENES.values()]
    write_tables(all_results, output_dir)
    write_figure(all_results, output_dir)
    write_summary(all_results, output_dir)
    logger.info("=== Task 1 complete -> %s ===", output_dir)
    return all_results


if __name__ == "__main__":
    run()
