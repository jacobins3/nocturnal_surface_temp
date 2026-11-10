# Step 9. Task 2: empirical semivariograms (raw and degree-2 detrended),
# single and nested model fits (Table 3, Figure 5).

from __future__ import annotations

import proj_fix  # noqa: F401  (must precede rasterio)
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
OUTPUT_DIR = DATA_ROOT / "analysis" / "task2_semivariogram"

# QA bit values (must match hotsat_lst.py)
QA_NODATA      = 1 << 0
QA_EPS_LOW     = 1 << 1
QA_OBLIQUE     = 1 << 2
QA_CLOUD       = 1 << 3
QA_HOT_ANOMALY = 1 << 4
QA_EXCLUDE = QA_NODATA | QA_EPS_LOW | QA_CLOUD | QA_HOT_ANOMALY

# --- variogram settings --------------------------------------------------
N_DRAWS         = 10        # independent random subsamples per scene
SAMPLE_SIZE     = 15000     # pixels per draw
MAX_LAG_M       = 1500.0    # maximum separation analysed, metres (must exceed the range)
N_LAG_BINS      = 75        # number of distance bins (20 m wide)
PIXEL_SIZE_M    = 10.0      # analysis-grid resolution

# Large-scale trend removal. The raw LST has km-scale gradients (no sill
# within the lag window in two of three scenes), which makes the plain
# variogram non-stationary. Every scene is analysed TWICE: raw, and after
# subtracting a low-order polynomial surface fitted to the scene. The
# detrended variogram describes the LOCAL structure (the thing a 3.5 m
# sensor resolves); the raw one is kept for transparency. 0 disables.
DETREND_ORDER   = 2
# A nested (two-structure) exponential model is also fitted. It is
# preferred over the single-structure model only if its AIC is lower by
# more than this margin.
NESTED_AIC_MARGIN = 10.0
RANDOM_SEED     = 42

# Optional exhaustive cross-check (lag-capped). Off by default -- set the
# scene key to run it on one scene as a verification of the subsampled
# estimate. Uses every pixel pair within VERIFY_MAX_LAG_M.
VERIFY_SCENE      = None        # e.g. "AD_industrial"
VERIFY_MAX_LAG_M  = 300.0


@dataclass
class Scene:
    key: str
    scene_id: str
    typology: str
    predictor_nc: Path
    lst_tif: Path
    qa_tif: Path


def _scene(key, scene_id, typology) -> Scene:
    return Scene(
        key=key, scene_id=scene_id, typology=typology,
        predictor_nc=DATA_ROOT / "predictors" /
        f"{scene_id}_predictors_10m.nc",
        lst_tif=DATA_ROOT / "lst" / f"{scene_id}_LST_C.tif",
        qa_tif=DATA_ROOT / "lst" / f"{scene_id}_LST_QA.tif",
    )


SCENES = {
    "AD_suburban":   _scene("AD_suburban",
                            "20231029_213936_HS1_SBT", "sub-urban"),
    "AD_industrial": _scene("AD_industrial",
                            "20231030_212656_HS1_SBT", "industrial"),
    "DXB_urban":     _scene("DXB_urban",
                            "20231120_213935_HS1_SBT", "urban"),
}


# ===========================================================================
# Raster IO
# ===========================================================================

def load_lst_and_mask(scene: Scene):
    import xarray as xr

    ds = xr.open_dataset(scene.predictor_nc)
    transform = rasterio.transform.Affine(*ds.attrs["transform"])
    crs = ds.attrs["crs"]
    h, w = ds[list(ds.data_vars)[0]].shape
    ds.close()

    def _on_grid(path, resampling):
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            st, sc, nod = src.transform, src.crs, src.nodata
        if nod is not None and not np.isnan(nod):
            arr = np.where(arr == nod, np.nan, arr)
        dst = np.full((h, w), np.nan, dtype=np.float32)
        reproject(source=arr, destination=dst,
                  src_transform=st, src_crs=sc, src_nodata=np.nan,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=resampling)
        return dst

    lst = _on_grid(scene.lst_tif, Resampling.bilinear)
    qa = _on_grid(scene.qa_tif, Resampling.nearest)
    qa = np.nan_to_num(qa, nan=QA_NODATA).astype(np.int32)

    valid = np.isfinite(lst) & ((qa & QA_EXCLUDE) == 0)
    return lst, valid, transform


# ===========================================================================
# Empirical variogram
# ===========================================================================

def _sample_pixels(lst, valid, n, rng):
    rows, cols = np.where(valid)
    if rows.size <= n:
        idx = np.arange(rows.size)
    else:
        idx = rng.choice(rows.size, size=n, replace=False)
    r, c = rows[idx], cols[idx]
    x = c.astype(np.float64) * PIXEL_SIZE_M
    y = r.astype(np.float64) * PIXEL_SIZE_M
    v = lst[r, c].astype(np.float64)
    return x, y, v


def empirical_variogram_one_draw(x, y, v, max_lag, n_bins):
    # pairwise differences (n x n); n ~ 10k -> 1e8 floats, fine in chunks
    n = x.size
    edges = np.linspace(0.0, max_lag, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sq_sum = np.zeros(n_bins)
    cnt = np.zeros(n_bins)

    # chunk over i to keep memory bounded
    chunk = 2000
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        dx = x[i0:i1, None] - x[None, :]
        dy = y[i0:i1, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        dv2 = (v[i0:i1, None] - v[None, :]) ** 2
        # keep upper triangle only would need index care; instead just
        # accept each pair twice -- it cancels in the mean.
        in_range = (dist > 0) & (dist <= max_lag)
        bin_idx = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
        flat_b = bin_idx[in_range]
        flat_d = dv2[in_range]
        sq_sum += np.bincount(flat_b, weights=flat_d, minlength=n_bins)
        cnt += np.bincount(flat_b, minlength=n_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = 0.5 * sq_sum / cnt
    return centres, gamma, cnt


def multi_draw_variogram(lst, valid, n_draws, sample_size,
                         max_lag, n_bins, seed):
    rng = np.random.default_rng(seed)
    gammas = np.full((n_draws, n_bins), np.nan)
    centres = None
    for d in range(n_draws):
        x, y, v = _sample_pixels(lst, valid, sample_size, rng)
        centres, gamma, cnt = empirical_variogram_one_draw(
            x, y, v, max_lag, n_bins)
        gammas[d] = gamma
    return centres, gammas


# ===========================================================================
# Theoretical model fitting
# ===========================================================================

def _model(h, nugget, sill, rng_, kind):
    h = np.asarray(h, dtype=np.float64)
    if kind == "exponential":
        g = nugget + sill * (1.0 - np.exp(-3.0 * h / rng_))
    elif kind == "spherical":
        s = np.minimum(h / rng_, 1.0)
        g = nugget + sill * (1.5 * s - 0.5 * s ** 3)
    elif kind == "gaussian":
        g = nugget + sill * (1.0 - np.exp(-3.0 * (h / rng_) ** 2))
    else:
        raise ValueError(kind)
    return g


def fit_variogram_model(lag, gamma, weights=None):
    from scipy.optimize import curve_fit

    m = np.isfinite(lag) & np.isfinite(gamma)
    lag, gamma = lag[m], gamma[m]
    if weights is not None:
        weights = weights[m]
    if lag.size < 5:
        return None

    var = float(np.nanvar(gamma))
    p0 = [0.1 * var, var, 0.3 * MAX_LAG_M]
    bounds = ([0.0, 0.0, PIXEL_SIZE_M],
              [5.0 * var + 1e-6, 10.0 * var + 1e-6, 5.0 * MAX_LAG_M])

    best = None
    for kind in ("exponential", "spherical", "gaussian"):
        try:
            popt, _ = curve_fit(
                lambda h, n, s, r: _model(h, n, s, r, kind),
                lag, gamma, p0=p0, bounds=bounds, maxfev=10000)
        except Exception:  # noqa: BLE001
            continue
        pred = _model(lag, *popt, kind)
        rmse = float(np.sqrt(np.mean((pred - gamma) ** 2)))
        if best is None or rmse < best["fit_rmse"]:
            best = {
                "model": kind,
                "nugget": float(popt[0]),
                "partial_sill": float(popt[1]),
                "total_sill": float(popt[0] + popt[1]),
                "range_m": float(popt[2]),
                "fit_rmse": rmse,
                "rss": float(np.sum((pred - gamma) ** 2)),
                "n": int(lag.size),
            }
    if best:
        best["aic"] = _aic(best["rss"], best["n"], 3)
        best["range_identified"] = bool(best["range_m"] <= 0.8 * MAX_LAG_M)
    return best


def detrend_surface(lst, valid, order, max_fit=200000, seed=1):
    if order <= 0:
        return lst, 0.0
    rows, cols = np.where(valid)
    x = cols.astype(np.float64) * PIXEL_SIZE_M
    y = rows.astype(np.float64) * PIXEL_SIZE_M
    xm, xs, ym, ys = x.mean(), x.std() + 1e-9, y.mean(), y.std() + 1e-9
    xn, yn = (x - xm) / xs, (y - ym) / ys
    terms = [(i, j) for i in range(order + 1)
             for j in range(order + 1 - i)]
    A = np.column_stack([xn ** i * yn ** j for i, j in terms])
    v = lst[rows, cols].astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = (rng.choice(v.size, max_fit, replace=False)
           if v.size > max_fit else np.arange(v.size))
    coef, *_ = np.linalg.lstsq(A[idx], v[idx], rcond=None)
    trend = A @ coef
    frac = 1.0 - float(np.var(v - trend) / np.var(v))
    out = lst.copy()
    out[rows, cols] = (v - trend).astype(np.float32)
    return out, frac


def _nested_model(h, nugget, s1, r1, s2, r2):
    h = np.asarray(h, dtype=np.float64)
    return (nugget + s1 * (1.0 - np.exp(-3.0 * h / r1))
            + s2 * (1.0 - np.exp(-3.0 * h / r2)))


def _aic(rss, n, k):
    return n * np.log(max(rss, 1e-12) / n) + 2 * k


def fit_nested_model(lag, gamma):
    from scipy.optimize import curve_fit
    m = np.isfinite(lag) & np.isfinite(gamma)
    lag, gamma = lag[m], gamma[m]
    if lag.size < 8:
        return None
    var = float(np.nanmax(gamma))
    lo = [0.0, 0.0, PIXEL_SIZE_M, 0.0, 0.5 * MAX_LAG_M]
    hi = [var, 3 * var, 0.6 * MAX_LAG_M, 10 * var, 10.0 * MAX_LAG_M]
    best = None
    for p0 in ([0.02 * var, 0.7 * var, 0.15 * MAX_LAG_M, 0.3 * var,
                MAX_LAG_M],
               [0.0, 0.5 * var, 0.3 * MAX_LAG_M, 0.5 * var,
                3 * MAX_LAG_M]):
        try:
            popt, _ = curve_fit(_nested_model, lag, gamma, p0=p0,
                                bounds=(lo, hi), maxfev=20000)
        except Exception:  # noqa: BLE001
            continue
        rss = float(np.sum((_nested_model(lag, *popt) - gamma) ** 2))
        if best is None or rss < best["rss"]:
            best = {"rss": rss, "n": int(lag.size),
                    "nugget": float(popt[0]),
                    "sill1": float(popt[1]), "range1_m": float(popt[2]),
                    "sill2": float(popt[3]), "range2_m": float(popt[4]),
                    "fit_rmse": float(np.sqrt(rss / lag.size))}
    if best:
        best["aic"] = _aic(best["rss"], best["n"], 5)
    return best


# ===========================================================================
# Per-scene analysis
# ===========================================================================

def analyse_scene(scene: Scene, detrend_order: int = 0) -> dict:
    tag = f"detrended (poly order {detrend_order})" if detrend_order else "raw"
    logger.info("=== %s (%s) -- %s ===", scene.key, scene.typology, tag)
    lst, valid, transform = load_lst_and_mask(scene)
    n_valid = int(valid.sum())
    logger.info("  valid pixels: %d", n_valid)
    frac = 0.0
    if detrend_order:
        lst, frac = detrend_surface(lst, valid, detrend_order)
        logger.info("  polynomial trend removed %.1f%% of the LST variance",
                    100.0 * frac)

    centres, gammas = multi_draw_variogram(
        lst, valid, N_DRAWS, SAMPLE_SIZE, MAX_LAG_M, N_LAG_BINS,
        RANDOM_SEED)
    gamma_mean = np.nanmean(gammas, axis=0)
    gamma_std = np.nanstd(gammas, axis=0)

    rel_spread = float(np.nanmean(gamma_std) /
                       (np.nanmean(gamma_mean) + 1e-9))
    logger.info("  multi-draw spread: %.1f%% of mean signal",
                100.0 * rel_spread)

    fit = fit_variogram_model(centres, gamma_mean)
    nested = fit_nested_model(centres, gamma_mean)
    preferred = "single"
    if fit:
        logger.info("  single model: %s | range %.0f m%s | nugget %.2f | "
                    "total sill %.2f | RMSE %.3f | AIC %.1f",
                    fit["model"], fit["range_m"],
                    "" if fit["range_identified"]
                    else " (> 0.8 x max lag: NOT identified)",
                    fit["nugget"], fit["total_sill"], fit["fit_rmse"],
                    fit["aic"])
    if nested and fit:
        if nested["aic"] < fit["aic"] - NESTED_AIC_MARGIN:
            preferred = "nested"
        logger.info("  nested exp : range1 %.0f m (sill %.2f) + range2 %.0f"
                    " m (sill %.2f) | nugget %.2f | RMSE %.3f | AIC %.1f "
                    "-> preferred: %s",
                    nested["range1_m"], nested["sill1"], nested["range2_m"],
                    nested["sill2"], nested["nugget"], nested["fit_rmse"],
                    nested["aic"], preferred)

    return {
        "scene": scene.key, "typology": scene.typology,
        "detrend": detrend_order, "trend_var_frac": frac,
        "n_valid": n_valid,
        "lag": centres, "gamma_mean": gamma_mean, "gamma_std": gamma_std,
        "rel_spread": rel_spread, "fit": fit, "nested": nested,
        "preferred": preferred,
    }


def verify_exhaustive(scene: Scene, max_lag: float) -> dict:
    logger.info("=== exhaustive cross-check: %s (max lag %.0f m) ===",
                scene.key, max_lag)
    lst, valid, _ = load_lst_and_mask(scene)
    rows, cols = np.where(valid)
    x = cols.astype(np.float64) * PIXEL_SIZE_M
    y = rows.astype(np.float64) * PIXEL_SIZE_M
    v = lst[rows, cols].astype(np.float64)

    n_bins = N_LAG_BINS
    edges = np.linspace(0.0, max_lag, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sq_sum = np.zeros(n_bins)
    cnt = np.zeros(n_bins)

    # spatial blocking: only compare points in nearby cells
    cell = max_lag
    gx = np.floor(x / cell).astype(np.int64)
    gy = np.floor(y / cell).astype(np.int64)
    from collections import defaultdict
    buckets = defaultdict(list)
    for i, (a, b) in enumerate(zip(gx, gy)):
        buckets[(a, b)].append(i)

    n = x.size
    logger.info("  %d valid pixels, %d spatial cells", n, len(buckets))
    for (a, b), idx_i in buckets.items():
        idx_i = np.array(idx_i)
        neigh = []
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                neigh += buckets.get((a + da, b + db), [])
        idx_j = np.array(neigh)
        if idx_j.size == 0:
            continue
        dx = x[idx_i, None] - x[None, idx_j]
        dy = y[idx_i, None] - y[None, idx_j]
        dist = np.sqrt(dx * dx + dy * dy)
        dv2 = (v[idx_i, None] - v[None, idx_j]) ** 2
        in_range = (dist > 0) & (dist <= max_lag)
        bidx = np.clip(np.digitize(dist, edges) - 1, 0, n_bins - 1)
        sq_sum += np.bincount(bidx[in_range], weights=dv2[in_range],
                              minlength=n_bins)
        cnt += np.bincount(bidx[in_range], minlength=n_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = 0.5 * sq_sum / cnt
    fit = fit_variogram_model(centres, gamma)
    if fit:
        logger.info("  exhaustive: %s | range %.0f m | total sill %.2f",
                    fit["model"], fit["range_m"], fit["total_sill"])
    return {"lag": centres, "gamma": gamma, "fit": fit}


# ===========================================================================
# Outputs
# ===========================================================================

def write_table(results: list[dict], output_dir: Path) -> None:
    import pandas as pd
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        f = r["fit"] or {}
        n = r["nested"] or {}
        rows.append({
            "scene": r["scene"], "typology": r["typology"],
            "detrend_order": r["detrend"],
            "trend_var_removed": r["trend_var_frac"],
            "n_valid": r["n_valid"],
            "model": f.get("model"),
            "range_m": f.get("range_m"),
            "range_identified": f.get("range_identified"),
            "nugget": f.get("nugget"),
            "partial_sill": f.get("partial_sill"),
            "total_sill": f.get("total_sill"),
            "fit_rmse": f.get("fit_rmse"),
            "aic_single": f.get("aic"),
            "nested_range1_m": n.get("range1_m"),
            "nested_sill1": n.get("sill1"),
            "nested_range2_m": n.get("range2_m"),
            "nested_sill2": n.get("sill2"),
            "nested_nugget": n.get("nugget"),
            "nested_rmse": n.get("fit_rmse"),
            "aic_nested": n.get("aic"),
            "preferred_model": r["preferred"],
            "multidraw_rel_spread": r["rel_spread"],
        })
    out = output_dir / "table3_variogram_parameters.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    logger.info("wrote %s", out)


def write_figure(results: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    orders = sorted({r["detrend"] for r in results})
    fig, axes = plt.subplots(1, len(orders),
                             figsize=(7.5 * len(orders), 6), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 3))
    cmap = {k: colors[i] for i, k in
            enumerate(dict.fromkeys(r["scene"] for r in results))}

    for ax, order in zip(axes[0], orders):
        for r in [q for q in results if q["detrend"] == order]:
            col = cmap[r["scene"]]
            lag, gm, gs = r["lag"], r["gamma_mean"], r["gamma_std"]
            ax.fill_between(lag, gm - gs, gm + gs, color=col, alpha=0.18)
            ax.plot(lag, gm, "o", color=col, ms=3,
                    label=f"{r['typology']} (empirical)")
            f, n = r["fit"], r["nested"]
            hh = np.linspace(0, MAX_LAG_M, 300)
            if r["preferred"] == "nested" and n:
                ax.plot(hh, _nested_model(hh, n["nugget"], n["sill1"],
                                          n["range1_m"], n["sill2"],
                                          n["range2_m"]),
                        "-", color=col, lw=1.8,
                        label=f"{r['typology']} nested: "
                              f"{n['range1_m']:.0f} m + "
                              f"{n['range2_m']:.0f} m")
                ax.axvline(n["range1_m"], color=col, ls=":", lw=1,
                           alpha=0.7)
            elif f:
                ax.plot(hh, _model(hh, f["nugget"], f["partial_sill"],
                                   f["range_m"], f["model"]),
                        "-", color=col, lw=1.8,
                        label=(f"{r['typology']} {f['model']}, range "
                               + (f"{f['range_m']:.0f} m"
                                  if f["range_identified"]
                                  else f"> {0.8*MAX_LAG_M:.0f} m "
                                       "(not identified)")))
                if f["range_identified"]:
                    ax.axvline(f["range_m"], color=col, ls=":", lw=1,
                               alpha=0.7)
        ax.axvline(100.0, color="grey", ls="--", lw=1)
        ax.text(105, 0.02, "Landsat 100 m", rotation=90, va="bottom",
                fontsize=8, color="grey",
                transform=ax.get_xaxis_transform())
        ax.set_xlim(0, MAX_LAG_M)
        ax.set_xlabel("Lag distance (m)")
        ax.set_ylabel("Semivariance (deg C$^2$)")
        ax.set_title("raw SBT" if order == 0 else
                     f"after removing a degree-{order} polynomial trend")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Night-time SBT semivariograms -- three UAE urban "
                 f"typologies (band = spread over {N_DRAWS} subsamples)")
    fig.tight_layout()
    out = output_dir / "figure5_semivariograms.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


def write_summary(results: list[dict], output_dir: Path) -> None:
    lines = ["HotSat-1 nocturnal SUHI -- Task 2 semivariogram analysis",
             "=" * 60, ""]
    for r in results:
        f = r["fit"] or {}
        n = r["nested"] or {}
        lines += [
            f"{r['typology'].upper()}  ({r['scene']})  "
            + ("raw" if r["detrend"] == 0
               else f"detrended (order {r['detrend']}, removed "
                    f"{100*r['trend_var_frac']:.0f}% of variance)"),
            f"  valid pixels        : {r['n_valid']:,}",
            f"  single model        : {f.get('model')}, range "
            f"{f.get('range_m', float('nan')):.0f} m "
            f"(identified: {f.get('range_identified')}), total sill "
            f"{f.get('total_sill', float('nan')):.2f}, nugget "
            f"{f.get('nugget', float('nan')):.2f}",
        ]
        if n:
            lines.append(
                f"  nested exponential  : {n['range1_m']:.0f} m "
                f"(sill {n['sill1']:.2f}) + {n['range2_m']:.0f} m "
                f"(sill {n['sill2']:.2f}); preferred = {r['preferred']}")
        lines += [f"  multi-draw spread   : {100*r['rel_spread']:.1f}%", ""]
    lines += [
        "INTERPRETATION",
        "  Ranges flagged 'not identified' exceed 0.8 x the maximum lag:",
        "  the variogram has no sill inside the window, so only a lower",
        "  bound is justified. Compare raw vs detrended: scenes whose",
        "  structure is dominated by a km-scale gradient lose most of",
        "  their sill and range after detrending. The detrended",
        "  variogram is the appropriate description of LOCAL structure.",
    ]
    out = output_dir / "task2_summary.txt"
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
    orders = [0] + ([DETREND_ORDER] if DETREND_ORDER > 0 else [])
    results = [analyse_scene(s, o) for o in orders
               for s in SCENES.values()]
    write_table(results, output_dir)
    write_figure(results, output_dir)
    write_summary(results, output_dir)

    if VERIFY_SCENE is not None and VERIFY_SCENE in SCENES:
        verify_exhaustive(SCENES[VERIFY_SCENE], VERIFY_MAX_LAG_M)

    logger.info("=== Task 2 complete -> %s ===", output_dir)
    return results


if __name__ == "__main__":
    run()
