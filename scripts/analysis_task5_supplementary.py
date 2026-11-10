# Step 12. Task 5: paired roof-minus-ring contrast with block bootstrap, co-
# registration and oblique-view checks, directional variograms, collinearity
# and resolution degradation (Table 4, Tables A3-A6, Figure 8, Figure S1).

from __future__ import annotations

import proj_fix  # noqa: F401  (must precede rasterio)
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.windows import transform as win_transform

import analysis_task2_semivariogram as t2
import analysis_task3_building_contrast as t3
import analysis_task4_form_attribution as t4

logger = logging.getLogger(__name__)

DATA_ROOT = t3.DATA_ROOT
OUTPUT_DIR = DATA_ROOT / "analysis" / "task5_supplementary"

SEED = 42
PIXEL_M = 10.0

# ---- ring analysis -------------------------------------------------------
RING_MAIN = (10.0, 30.0)                      # inner, outer distance (m)
RING_VARIANTS = [(10.0, 20.0), (10.0, 30.0), (10.0, 50.0), (20.0, 50.0)]
EROSION_VARIANTS_M = [0.0, 2.5, 5.0, 7.5]     # roof erosion sensitivity
MIN_ROOF_PIX, MIN_RING_PIX = 3, 5
N_BOOT = 2000
N_BOOT_PIXEL_BLOCKS = 500
BLOCK_M = 1000.0
SHIFT_RANGE_PX = 4
WC_WATER = 80
STRATA = {"built-up": (50,), "bare/sparse": (60,),
          "vegetation": (10, 20, 30, 40)}

# ---- directional variograms ---------------------------------------------
DIR_ANGLES = (0.0, 45.0, 90.0, 135.0)         # 0 = east-west, 90 = north-south
DIR_TOL_DEG = 22.5
DIR_DRAWS = 5
DIR_SAMPLE = 15000
DIR_MAX_LAG = 1500.0
DIR_BIN_M = 20.0
DETREND_ORDER = 2

# ---- degradation ---------------------------------------------------------
AGG_FACTORS = [1, 3, 10, 25, 100]             # x10 m -> 10, 30, 100, 250, 1000 m
AGG_MIN_VALID_FRAC = 0.5


# ===========================================================================
# Common loading
# ===========================================================================

def _load_scene(key: str):
    s3 = t3.SCENES[key]
    transform, crs, h, w = t3.load_grid(s3.predictor_nc)
    shape = (h, w)
    lst = t3.load_raster_on_grid(s3.lst_tif, transform, crs, shape,
                                 t3.Resampling.bilinear)
    qa = t3.load_raster_on_grid(s3.qa_tif, transform, crs, shape,
                                t3.Resampling.nearest)
    qa = np.nan_to_num(qa, nan=t3.QA_NODATA).astype(np.int32)
    valid = np.isfinite(lst) & ((qa & t3.QA_EXCLUDE) == 0)
    cls, gdf = t3.build_class_mask(s3, transform, crs, shape)
    import xarray as xr
    ds = xr.open_dataset(s3.predictor_nc)
    wc = np.round(ds["wc_class"].values).astype(np.int32)
    ds.close()
    return dict(s3=s3, lst=lst, valid=valid, transform=transform, crs=crs,
                shape=shape, cls=cls, gdf=gdf, wc=wc)


# ===========================================================================
# Part 1 -- paired ring contrast
# ===========================================================================

def _window(geom, buf_m, transform, shape):
    minx, miny, maxx, maxy = geom.bounds
    inv = ~transform
    c_a, r_a = inv * (minx - buf_m, maxy + buf_m)
    c_b, r_b = inv * (maxx + buf_m, miny - buf_m)
    r0 = max(int(math.floor(min(r_a, r_b))) - 1, 0)
    r1 = min(int(math.ceil(max(r_a, r_b))) + 1, shape[0])
    c0 = max(int(math.floor(min(c_a, c_b))) - 1, 0)
    c1 = min(int(math.ceil(max(c_a, c_b))) + 1, shape[1])
    if r1 <= r0 or c1 <= c0:
        return None
    wt = win_transform(Window(c0, r0, c1 - c0, r1 - r0), transform)
    return r0, r1, c0, c1, wt


def _rast(geom, out_shape, wt):
    if geom is None or geom.is_empty:
        return np.zeros(out_shape, bool)
    return rasterize([(geom, 1)], out_shape=out_shape, transform=wt,
                     fill=0, dtype="uint8").astype(bool)


def _dominant_stratum(codes):
    if codes.size == 0:
        return "none"
    best, best_n = "other", 0
    for name, cs in STRATA.items():
        n = int(np.isin(codes, cs).sum())
        if n > best_n:
            best, best_n = name, n
    return best if best_n >= 0.5 * codes.size else "mixed"


def per_building_pairs(sc: dict, erosion_m: float, ring: tuple[float, float],
                       with_sides: bool = False) -> pd.DataFrame:
    lst, valid, cls, wc = sc["lst"], sc["valid"], sc["cls"], sc["wc"]
    transform, shape, gdf = sc["transform"], sc["shape"], sc["gdf"]
    ok_ground = valid & (cls == t3.CLS_OPEN) & (wc != WC_WATER)
    r_in, r_out = ring
    heights = gdf["height"].values if "height" in gdf else \
        np.full(len(gdf), np.nan)
    sources = gdf["source"].values if "source" in gdf else \
        np.array([""] * len(gdf))
    rows = []
    for i, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        win = _window(geom, r_out + 2 * PIXEL_M, transform, shape)
        if win is None:
            continue
        r0, r1, c0, c1, wt = win
        osh = (r1 - r0, c1 - c0)
        roof_geom = geom.buffer(-erosion_m) if erosion_m > 0 else geom
        roof_m = _rast(roof_geom, osh, wt) & valid[r0:r1, c0:c1]
        if roof_m.sum() < MIN_ROOF_PIX:
            continue
        ring_m = (_rast(geom.buffer(r_out), osh, wt)
                  & ~_rast(geom.buffer(r_in), osh, wt)
                  & ok_ground[r0:r1, c0:c1])
        if ring_m.sum() < MIN_RING_PIX:
            continue
        roof_v = lst[r0:r1, c0:c1][roof_m]
        ring_v = lst[r0:r1, c0:c1][ring_m]
        row = {"bld": i, "height": float(heights[i]), "source": sources[i],
               "n_roof": int(roof_v.size), "n_ring": int(ring_v.size),
               "roof_med": float(np.median(roof_v)),
               "ring_med": float(np.median(ring_v)),
               "stratum": _dominant_stratum(
                   wc[r0:r1, c0:c1][ring_m]),
               "cx": float(geom.centroid.x), "cy": float(geom.centroid.y)}
        row["diff"] = row["roof_med"] - row["ring_med"]
        if with_sides:
            rr, cc = np.where(ring_m)
            xs = wt.c + (cc + 0.5) * wt.a
            ys = wt.f + (rr + 0.5) * wt.e
            dx, dy = xs - geom.centroid.x, ys - geom.centroid.y
            side = np.where(np.abs(dx) >= np.abs(dy),
                            np.where(dx >= 0, "E", "W"),
                            np.where(dy >= 0, "N", "S"))
            for sd in "NESW":
                v = ring_v[side == sd]
                row[f"ring_{sd}"] = (float(np.median(v))
                                     if v.size >= 3 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_bootstrap_median(values, groups, n_boot, rng):
    values = np.asarray(values, float)
    if values.size < 5:
        return (np.nan, np.nan)
    if groups is None:
        idx = rng.integers(0, values.size, size=(n_boot, values.size))
        meds = np.median(values[idx], axis=1)
        return tuple(np.percentile(meds, [2.5, 97.5]))
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    by = {g: values[groups == g] for g in uniq}
    meds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        meds[b] = np.median(np.concatenate([by[g] for g in pick]))
    return tuple(np.percentile(meds, [2.5, 97.5]))


def _block_ids(x, y, block_m):
    return (np.floor(x / block_m).astype(int) * 100003
            + np.floor(y / block_m).astype(int))


def pixel_contrast_block_bootstrap(sc, n_boot, rng):
    lst, valid, cls, wc = sc["lst"], sc["valid"], sc["cls"], sc["wc"]
    roof = valid & (cls == t3.CLS_ROOF)
    openg = valid & (cls == t3.CLS_OPEN) & (wc != WC_WATER)
    h, w = sc["shape"]
    rr, cc = np.indices((h, w))
    blk = _block_ids(cc * PIXEL_M, rr * PIXEL_M, BLOCK_M)
    ids = np.unique(blk[roof | openg])
    roof_by = {b: lst[roof & (blk == b)] for b in ids}
    open_by = {b: lst[openg & (blk == b)] for b in ids}
    est = float(np.median(lst[roof]) - np.median(lst[openg]))
    meds = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.choice(ids, size=ids.size, replace=True)
        rv = np.concatenate([roof_by[b] for b in pick])
        ov = np.concatenate([open_by[b] for b in pick])
        meds[k] = np.median(rv) - np.median(ov) if rv.size and ov.size \
            else np.nan
    lo, hi = np.nanpercentile(meds, [2.5, 97.5])
    return est, float(lo), float(hi), int(ids.size)


def coregistration_scan(sc, max_shift=SHIFT_RANGE_PX):
    lst, valid, cls, wc = sc["lst"], sc["valid"], sc["cls"], sc["wc"]
    rows = []
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            sh = np.roll(np.roll(cls, dr, axis=0), dc, axis=1)
            # blank wrapped borders
            if dr > 0: sh[:dr, :] = t3.CLS_EDGE
            if dr < 0: sh[dr:, :] = t3.CLS_EDGE
            if dc > 0: sh[:, :dc] = t3.CLS_EDGE
            if dc < 0: sh[:, dc:] = t3.CLS_EDGE
            roof = valid & (sh == t3.CLS_ROOF)
            openg = valid & (sh == t3.CLS_OPEN) & (wc != WC_WATER)
            if roof.sum() < 50 or openg.sum() < 50:
                continue
            rows.append({"shift_row_px": dr, "shift_col_px": dc,
                         "contrast_C": float(np.median(lst[roof])
                                             - np.median(lst[openg]))})
    return pd.DataFrame(rows)


def run_ring(out_dir: Path, rng):
    summ, strat, sens, coreg, obl, scene_wide = [], [], [], [], [], []
    for key in t3.SCENES:
        logger.info("=== ring analysis: %s ===", key)
        sc = _load_scene(key)
        e_main = t3.EROSION_M_BY_SCENE.get(key, t3.EROSION_M_DEFAULT)

        # main paired estimate
        df = per_building_pairs(sc, e_main, RING_MAIN, with_sides=True)
        df.insert(0, "scene", key)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"ring_pairs_{key}.csv", index=False)
        if df.empty:
            logger.warning("  no usable buildings for %s", key)
            continue
        blk = _block_ids(df.cx.values, df.cy.values, BLOCK_M)
        lo_b, hi_b = cluster_bootstrap_median(df["diff"], None, N_BOOT, rng)
        lo_k, hi_k = cluster_bootstrap_median(df["diff"], blk, N_BOOT, rng)
        summ.append({
            "scene": key, "erosion_m": e_main,
            "ring_inner_m": RING_MAIN[0], "ring_outer_m": RING_MAIN[1],
            "n_buildings_total": len(sc["gdf"]),
            "n_buildings_used": len(df),
            "median_roof_C": df.roof_med.median(),
            "median_ring_C": df.ring_med.median(),
            "paired_median_diff_C": df["diff"].median(),
            "paired_mean_diff_C": df["diff"].mean(),
            "ci_building_lo": lo_b, "ci_building_hi": hi_b,
            "ci_block_lo": lo_k, "ci_block_hi": hi_k,
            "n_blocks": int(np.unique(blk).size),
            "frac_buildings_roof_cooler": float((df["diff"] < 0).mean()),
            "median_ring_pixels": df.n_ring.median(),
            "median_roof_pixels": df.n_roof.median()})

        # strata
        for name, g in df.groupby("stratum"):
            if len(g) < 10:
                continue
            gb = _block_ids(g.cx.values, g.cy.values, BLOCK_M)
            lo, hi = cluster_bootstrap_median(g["diff"], gb, N_BOOT, rng)
            strat.append({"scene": key, "ring_ground_cover": name,
                          "n_buildings": len(g),
                          "median_diff_C": g["diff"].median(),
                          "ci_block_lo": lo, "ci_block_hi": hi})

        # per-source (footprint provenance)
        for name, g in df.groupby("source"):
            if len(g) < 30:
                continue
            strat.append({"scene": key, "ring_ground_cover": f"source:{name}",
                          "n_buildings": len(g),
                          "median_diff_C": g["diff"].median(),
                          "ci_block_lo": np.nan, "ci_block_hi": np.nan})

        # sensitivity: erosion x ring
        for e in EROSION_VARIANTS_M:
            d_ = per_building_pairs(sc, e, RING_MAIN)
            sens.append({"scene": key, "kind": "erosion", "erosion_m": e,
                         "ring": f"{RING_MAIN[0]:.0f}-{RING_MAIN[1]:.0f}",
                         "n_buildings": len(d_),
                         "median_diff_C": d_["diff"].median()
                         if len(d_) else np.nan})
        for rg in RING_VARIANTS:
            d_ = per_building_pairs(sc, e_main, rg)
            sens.append({"scene": key, "kind": "ring", "erosion_m": e_main,
                         "ring": f"{rg[0]:.0f}-{rg[1]:.0f}",
                         "n_buildings": len(d_),
                         "median_diff_C": d_["diff"].median()
                         if len(d_) else np.nan})

        # scene-wide pixel contrast with block bootstrap
        est, lo, hi, nb = pixel_contrast_block_bootstrap(
            sc, N_BOOT_PIXEL_BLOCKS, rng)
        scene_wide.append({"scene": key, "roof_open_median_C": est,
                           "ci_block_lo": lo, "ci_block_hi": hi,
                           "n_blocks": nb})

        # co-registration
        cg = coregistration_scan(sc)
        cg.insert(0, "scene", key)
        coreg.append(cg)
        base = cg[(cg.shift_row_px == 0) & (cg.shift_col_px == 0)]
        best = cg.loc[cg.contrast_C.abs().idxmax()]
        logger.info("  co-registration: contrast at 0 shift %+.2f C; "
                    "largest |contrast| %+.2f C at shift (%d, %d) px",
                    float(base.contrast_C.iloc[0]), best.contrast_C,
                    int(best.shift_row_px), int(best.shift_col_px))

        # oblique-view checks
        from scipy import stats
        o = {"scene": key,
             "view_note": "oblique" if key != "AD_industrial" else "near-nadir"}
        h_ok = df.dropna(subset=["height"])
        if len(h_ok) > 20:
            rho, p = stats.spearmanr(h_ok.height, h_ok["diff"])
            o["spearman_diff_vs_height"] = float(rho)
            o["spearman_p"] = float(p)
            lowb = h_ok[h_ok.height < 4.0]
            o["median_diff_height_lt4m"] = (float(lowb["diff"].median())
                                           if len(lowb) >= 10 else np.nan)
            o["median_diff_height_ge4m"] = float(
                h_ok[h_ok.height >= 4.0]["diff"].median())
        for sd in "NESW":
            col = f"ring_{sd}"
            if col in df:
                o[f"roof_minus_ring_{sd}_C"] = float(
                    (df.roof_med - df[col]).median())
        sides = [o.get(f"roof_minus_ring_{sd}_C") for sd in "NESW"]
        sides = [s for s in sides if s is not None and np.isfinite(s)]
        o["side_spread_C"] = float(max(sides) - min(sides)) if sides else np.nan
        obl.append(o)

    pd.DataFrame(summ).to_csv(out_dir / "table_ring_contrast_summary.csv", index=False)
    pd.DataFrame(strat).to_csv(out_dir / "table_ring_contrast_strata.csv", index=False)
    pd.DataFrame(sens).to_csv(out_dir / "table_ring_contrast_sensitivity.csv", index=False)
    pd.DataFrame(scene_wide).to_csv(out_dir / "table_scenewide_contrast_blockboot.csv", index=False)
    if coreg:
        pd.concat(coreg).to_csv(out_dir / "table_coregistration_scan.csv", index=False)
    pd.DataFrame(obl).to_csv(out_dir / "table_oblique_checks.csv", index=False)
    logger.info("ring analysis written to %s", out_dir)


# ===========================================================================
# Part 2 -- directional variograms
# ===========================================================================

def directional_variogram(x, y, v, angles, tol_deg, max_lag, bin_m):
    n_bins = int(round(max_lag / bin_m))
    edges = np.linspace(0.0, max_lag, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sq = np.zeros((len(angles), n_bins))
    cnt = np.zeros((len(angles), n_bins))
    n = x.size
    chunk = 1500
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        dx = x[i0:i1, None] - x[None, :]
        dy = y[i0:i1, None] - y[None, :]
        dist = np.hypot(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx)) % 180.0
        dv2 = (v[i0:i1, None] - v[None, :]) ** 2
        base = (dist > 0) & (dist <= max_lag)
        b = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
        for k, a in enumerate(angles):
            da = np.abs(ang - a)
            da = np.minimum(da, 180.0 - da)
            m = base & (da <= tol_deg)
            sq[k] += np.bincount(b[m], weights=dv2[m], minlength=n_bins)
            cnt[k] += np.bincount(b[m], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = 0.5 * sq / cnt
    return centres, gamma


def fit_exponential(lag, gamma):
    from scipy.optimize import curve_fit
    m = np.isfinite(lag) & np.isfinite(gamma)
    lag, gamma = lag[m], gamma[m]
    if lag.size < 8:
        return None
    top = float(np.nanmax(gamma))

    def f(h, n, s, r):
        return n + s * (1.0 - np.exp(-3.0 * h / r))
    try:
        popt, _ = curve_fit(f, lag, gamma,
                            p0=[0.0, top, 0.3 * DIR_MAX_LAG],
                            bounds=([0, 0, PIXEL_M],
                                    [top, 5 * top, 5 * DIR_MAX_LAG]),
                            maxfev=20000)
    except Exception:  # noqa: BLE001
        return None
    return {"nugget": float(popt[0]), "partial_sill": float(popt[1]),
            "range_m": float(popt[2]),
            "identified": bool(popt[2] <= 0.8 * DIR_MAX_LAG)}


def run_dir(out_dir: Path, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, curves = [], {}
    for key, s2 in t2.SCENES.items():
        logger.info("=== directional variograms: %s ===", key)
        lst, valid, _ = t2.load_lst_and_mask(s2)
        lst, frac = t2.detrend_surface(lst, valid, DETREND_ORDER)
        gam = []
        for d in range(DIR_DRAWS):
            x, y, v = t2._sample_pixels(lst, valid, DIR_SAMPLE, rng)
            lag, g = directional_variogram(x, y * -1.0, v, DIR_ANGLES,
                                           DIR_TOL_DEG, DIR_MAX_LAG,
                                           DIR_BIN_M)
            gam.append(g)
        gmean = np.nanmean(np.stack(gam), axis=0)
        curves[key] = (lag, gmean)
        ranges = []
        for k, a in enumerate(DIR_ANGLES):
            fit = fit_exponential(lag, gmean[k])
            rows.append({"scene": key, "direction_deg": a,
                         "range_m": fit["range_m"] if fit else np.nan,
                         "partial_sill": fit["partial_sill"] if fit else np.nan,
                         "nugget": fit["nugget"] if fit else np.nan,
                         "range_identified": fit["identified"] if fit else False})
            if fit and fit["identified"]:
                ranges.append(fit["range_m"])
        rows.append({"scene": key, "direction_deg": "anisotropy_ratio_max/min",
                     "range_m": (max(ranges) / min(ranges)
                                 if len(ranges) >= 2 else np.nan),
                     "partial_sill": np.nan, "nugget": np.nan,
                     "range_identified": len(ranges) >= 2})
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "table_directional_variograms.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    labels = ["a", "b", "c"]
    for ax, (key, (lag, g)), lab in zip(axes, curves.items(), labels):
        for k, a in enumerate(DIR_ANGLES):
            ax.plot(lag, g[k], label=f"{a:.0f}°")
        ax.set_xlabel("Lag distance (m)")
        ax.set_ylabel("Semivariance (°C²)")
        ax.set_title(lab, loc="left", fontweight="normal", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].legend(title="direction (0° = E–W)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_directional_variograms.png", dpi=200)
    plt.close(fig)


# ===========================================================================
# Part 3 -- predictor correlation and VIF
# ===========================================================================

def run_corr(out_dir: Path, rng, max_n=200000):
    out_dir.mkdir(parents=True, exist_ok=True)
    vif_rows = []
    for key, s4 in t4.SCENES.items():
        logger.info("=== predictor correlations: %s ===", key)
        X, y, names, rc, res = t4.load_scene_table(s4)
        if X.shape[0] > max_n:
            idx = rng.choice(X.shape[0], max_n, replace=False)
            X = X[idx]; y = y[idx]
        df = pd.DataFrame(X, columns=names)
        pear = df.corr(method="pearson")
        spear = df.sample(min(len(df), 50000), random_state=1).corr(method="spearman")
        pear.to_csv(out_dir / f"corr_pearson_{key}.csv")
        spear.to_csv(out_dir / f"corr_spearman_{key}.csv")
        inv = np.linalg.pinv(pear.values)
        for n_, v_ in zip(names, np.diag(inv)):
            vif_rows.append({"scene": key, "predictor": n_, "VIF": float(v_),
                             "corr_with_LST": float(np.corrcoef(df[n_], y)[0, 1])})
    pd.DataFrame(vif_rows).to_csv(out_dir / "table_predictor_vif.csv", index=False)


# ===========================================================================
# Part 4 -- resolution degradation
# ===========================================================================

def block_mean(a, valid, k, min_frac=AGG_MIN_VALID_FRAC):
    h, w = a.shape
    H, W = (h // k) * k, (w // k) * k
    x = np.where(valid, a, 0.0)[:H, :W].reshape(H // k, k, W // k, k)
    m = valid[:H, :W].reshape(H // k, k, W // k, k)
    cnt = m.sum(axis=(1, 3))
    s = x.sum(axis=(1, 3))
    out = np.where(cnt >= min_frac * k * k, s / np.maximum(cnt, 1), np.nan)
    return out


def run_degrade(out_dir: Path, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in t3.SCENES:
        logger.info("=== degradation: %s ===", key)
        sc = _load_scene(key)
        lst, valid, cls, wc = sc["lst"], sc["valid"], sc["cls"], sc["wc"]
        ds = xr.open_dataset(sc["s3"].predictor_nc)
        preds = {n: ds[n].values.astype(np.float32)
                 for n in ("albedo", "svf", "mean_height") if n in ds}
        ds.close()
        ones = np.ones_like(valid, dtype=bool)
        roof_f_src = (cls == t3.CLS_ROOF).astype(np.float32)
        sd10 = float(np.nanstd(lst[valid]))
        for k in AGG_FACTORS:
            L = block_mean(lst, valid, k)
            RF = block_mean(roof_f_src, ones, k, min_frac=0.0)
            m = np.isfinite(L) & np.isfinite(RF)
            row = {"scene": key, "pixel_m": k * PIXEL_M,
                   "n_pixels": int(m.sum()),
                   "sd_C": float(np.nanstd(L[m])) if m.any() else np.nan}
            row["sd_ratio_vs_10m"] = row["sd_C"] / sd10
            row["variance_retained_pct"] = 100.0 * row["sd_ratio_vs_10m"] ** 2
            if m.sum() >= 20 and np.nanstd(RF[m]) > 0:
                slope = np.polyfit(RF[m], L[m], 1)[0]
                row["unmixing_roof_minus_other_C"] = float(slope)
            else:
                row["unmixing_roof_minus_other_C"] = np.nan
            pure_roof = m & (RF >= 0.8)
            row["n_pure_roof_pixels"] = int(pure_roof.sum())
            for n_, arr in preds.items():
                P = block_mean(arr, np.isfinite(arr), k)
                mm = m & np.isfinite(P)
                row[f"pearson_LST_{n_}"] = (float(np.corrcoef(L[mm], P[mm])[0, 1])
                                            if mm.sum() >= 20 else np.nan)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "table_degradation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for key, g in df.groupby("scene"):
        axes[0].plot(g.pixel_m, g.variance_retained_pct, marker="o", label=key)
        axes[1].plot(g.pixel_m, g.unmixing_roof_minus_other_C, marker="o", label=key)
    for ax, lab, yl in zip(axes, ["a", "b"],
                           ["Variance retained (%)",
                            "Roof-minus-other contrast by unmixing (°C)"]):
        ax.set_xscale("log"); ax.set_xlabel("Pixel size (m)"); ax.set_ylabel(yl)
        ax.set_title(lab, loc="left", fontweight="normal", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_degradation.png", dpi=200)
    plt.close(fig)


# ===========================================================================
# Driver
# ===========================================================================

def run(parts=("ring", "dir", "corr", "degrade"), output_dir: Path = OUTPUT_DIR):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "ring" in parts:
        run_ring(output_dir, rng)
    if "dir" in parts:
        run_dir(output_dir, rng)
    if "corr" in parts:
        run_corr(output_dir, rng)
    if "degrade" in parts:
        run_degrade(output_dir, rng)


if __name__ == "__main__":
    import sys
    run(parts=tuple(sys.argv[1:]) or ("ring", "dir", "corr", "degrade"))
