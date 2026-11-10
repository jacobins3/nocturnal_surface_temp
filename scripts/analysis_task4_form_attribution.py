# Step 11. Task 4: XGBoost + SHAP attribution of SBT to six urban-form
# predictors with spatial-block cross-validation (Tables 6-7, Figures 6-7).

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
OUTPUT_DIR = DATA_ROOT / "analysis" / "task4_form_attribution"

# QA bits (match hotsat_lst.py / tasks 1-3)
QA_NODATA      = 1 << 0
QA_EPS_LOW     = 1 << 1
QA_OBLIQUE     = 1 << 2
QA_CLOUD       = 1 << 3
QA_HOT_ANOMALY = 1 << 4
QA_EXCLUDE = QA_NODATA | QA_EPS_LOW | QA_CLOUD | QA_HOT_ANOMALY

# Urban-FORM predictors used for the attribution.
#
# CIRCULARITY CONTROL. The LST itself is retrieved in hotsat_lst.py with an
# emissivity built from the ESA WorldCover class (wc_class) and NDVI.
# Any predictor that is a function of those inputs shares information with
# the TARGET by construction, so its "importance" partly reflects the
# retrieval, not the surface physics. Therefore:
#   * wc_class is EXCLUDED (it assigns the emissivity directly);
#   * impervious_frac is EXCLUDED (it is a 30 m focal mean of the
#     WorldCover built-up class, i.e. wc_class smoothed);
#   * ndvi/ndbi/mndwi/bsi are EXCLUDED (ndvi enters the emissivity via
#     the FVC blend; the rest restate land cover).
# albedo is kept: it is computed from Sentinel-2 reflectance bands, is
# not used in the emissivity, and is a genuine surface property (its
# correlation with NDVI is reported per scene as a diagnostic).
FORM_PREDICTORS = [
    "albedo",          # broadband surface albedo (S2)
    "fp_density",      # building footprint density (WSF-3D)
    "mean_height",     # mean building height (WSF-3D)
    "svf",             # sky-view factor (WSF-3D derived)
    "built_area",      # built area per cell (WSF-3D)
    "built_volume",    # built volume per cell (WSF-3D)
]

# Sensitivity run only: the two retrieval-linked layers are ADDED back to
# show how much the ranking changes when they are allowed in. Reported
# as a robustness table, never as the headline attribution.
CIRCULAR_PREDICTORS = ["impervious_frac", "wc_class"]
RUN_SENSITIVITY = True

# --- spatial-block cross-validation -------------------------------------
# Pixels are grouped into square blocks and whole blocks are held out, so
# test pixels are not spatial neighbours of training pixels. The block
# edge must exceed the semivariogram range (Task 2: 86 / 472 / 278 m);
# 1000 m is > 2x the largest range.
BLOCK_SIZE_M = 1000.0
N_FOLDS = 5
CV_SHAP_SAMPLE = 5000          # held-out pixels explained per fold
RANDOM_CV_COMPARISON = True    # also run ordinary random K-fold to show
                               # how much autocorrelation inflates R2

# a predictor is treated as degenerate (and dropped for that scene) if
# its standard deviation over valid pixels is below this fraction of its
# mean magnitude, or essentially zero.
DEGENERATE_REL_STD = 1e-3

# pixels are subsampled for the SHAP computation (SHAP on ~300k rows is
# slow and unnecessary); the model is FITTED on all valid pixels.
SHAP_SAMPLE_SIZE = 20000
RANDOM_SEED = 42


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
# Data assembly
# ===========================================================================

def load_scene_table(scene: Scene, predictors=None):
    import xarray as xr
    predictors = list(predictors or FORM_PREDICTORS)

    ds = xr.open_dataset(scene.predictor_nc)
    transform = rasterio.transform.Affine(*ds.attrs["transform"])
    crs = ds.attrs["crs"]
    h, w = ds[predictors[0]].shape

    cube = np.stack([ds[name].values.astype(np.float32)
                     for name in predictors], axis=-1)
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

    cube_flat = cube.reshape(h * w, -1)
    lst_flat = lst.ravel()
    qa_flat = qa.ravel()

    valid = (np.isfinite(cube_flat).all(axis=1)
             & np.isfinite(lst_flat)
             & ((qa_flat & QA_EXCLUDE) == 0))
    X = cube_flat[valid]
    y = lst_flat[valid]
    rr, cc = np.divmod(np.flatnonzero(valid), w)
    rc = np.column_stack([rr, cc]).astype(np.int32)
    res_m = float(abs(transform.a))

    # drop degenerate (near-constant) predictors for this scene
    keep, names = [], []
    for j, name in enumerate(predictors):
        col = X[:, j]
        mag = max(abs(np.mean(col)), 1e-9)
        if np.std(col) / mag < DEGENERATE_REL_STD:
            logger.info("  [%s] dropping degenerate predictor '%s' "
                        "(near-constant in this scene)",
                        scene.key, name)
        else:
            keep.append(j)
            names.append(name)
    X = X[:, keep]
    logger.info("  [%s] %d valid pixels, %d form predictors used",
                scene.key, X.shape[0], X.shape[1])
    # diagnostic: albedo is partly entangled with NDVI/land cover
    if "albedo" in names:
        try:
            ds2 = xr.open_dataset(scene.predictor_nc)
            nd = ds2["ndvi"].values.astype(np.float32).ravel()[valid]
            ds2.close()
            ok = np.isfinite(nd)
            r = np.corrcoef(X[ok, names.index("albedo")], nd[ok])[0, 1]
            logger.info("  [%s] diagnostic corr(albedo, ndvi) = %+.2f",
                        scene.key, r)
        except Exception as e:                      # diagnostic only
            logger.info("  [%s] albedo-ndvi diagnostic skipped (%s)",
                        scene.key, e)
    return X, y, names, rc, res_m


# ===========================================================================
# Model + SHAP attribution
# ===========================================================================

def fit_form_model(X, y):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, n_jobs=-1, random_state=RANDOM_SEED,
        objective="reg:squarederror",
    )
    model.fit(X, y)
    pred = model.predict(X)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return model, r2


def shap_attribution(model, X, names, rng):
    # subsample rows for the SHAP computation
    if X.shape[0] > SHAP_SAMPLE_SIZE:
        idx = rng.choice(X.shape[0], SHAP_SAMPLE_SIZE, replace=False)
    else:
        idx = np.arange(X.shape[0])
    X_s = X[idx]

    try:
        import shap
    except ImportError:
        logger.warning("  shap not installed -- falling back to XGBoost "
                        "gain importance (install shap for the full "
                        "attribution).")
        imp = model.feature_importances_
        return imp, None, X_s, names

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_s)
    mean_abs = np.abs(sv).mean(axis=0)
    return mean_abs, sv, X_s, names


def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, n_jobs=-1, random_state=RANDOM_SEED,
        objective="reg:squarederror")


def _r2_rmse(y, p):
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return (1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            float(np.sqrt(np.mean((y - p) ** 2))))


def spatial_block_cv(X, y, names, rc, res_m, rng):
    bpx = max(1, int(round(BLOCK_SIZE_M / res_m)))
    block = (rc[:, 0] // bpx).astype(np.int64) * 100000 + (rc[:, 1] // bpx)
    ub, inv = np.unique(block, return_inverse=True)
    n_blocks = len(ub)
    k = min(N_FOLDS, n_blocks)
    if k < 3:
        logger.warning("  only %d spatial blocks -- block CV skipped "
                        "(shrink BLOCK_SIZE_M)", n_blocks)
        return None
    fold_of_block = rng.permutation(n_blocks) % k
    fold = fold_of_block[inv]
    logger.info("  block CV: %d blocks of %.0f m -> %d folds",
                n_blocks, BLOCK_SIZE_M, k)

    try:
        import shap
    except ImportError:
        shap = None

    r2s, rmses, fold_shap = [], [], []
    for f in range(k):
        te, tr = fold == f, fold != f
        if te.sum() < 100 or tr.sum() < 100:
            continue
        m = _xgb()
        m.fit(X[tr], y[tr])
        r2, rmse = _r2_rmse(y[te], m.predict(X[te]))
        r2s.append(r2); rmses.append(rmse)
        if shap is not None:
            idx = np.flatnonzero(te)
            if len(idx) > CV_SHAP_SAMPLE:
                idx = rng.choice(idx, CV_SHAP_SAMPLE, replace=False)
            sv = shap.TreeExplainer(m).shap_values(X[idx])
            fold_shap.append(np.abs(sv).mean(axis=0))
        else:
            fold_shap.append(m.feature_importances_)
        logger.info("    fold %d: held-out R2 %.3f  RMSE %.2f C", f,
                    r2, rmse)

    out = {"n_blocks": n_blocks, "n_folds": len(r2s),
           "block_r2_mean": float(np.mean(r2s)),
           "block_r2_sd": float(np.std(r2s)),
           "block_rmse_mean": float(np.mean(rmses)),
           "fold_shap": np.array(fold_shap)}

    if RANDOM_CV_COMPARISON:
        rf = rng.permutation(len(y)) % k
        rr2 = []
        for f in range(k):
            te, tr = rf == f, rf != f
            m = _xgb(); m.fit(X[tr], y[tr])
            rr2.append(_r2_rmse(y[te], m.predict(X[te]))[0])
        out["random_r2_mean"] = float(np.mean(rr2))
    else:
        out["random_r2_mean"] = float("nan")

    # rank stability: mean pairwise Spearman rho of fold attributions
    from scipy.stats import spearmanr
    fs = out["fold_shap"]
    rhos = [spearmanr(fs[a], fs[b])[0]
            for a in range(len(fs)) for b in range(a + 1, len(fs))]
    out["rank_rho_mean"] = float(np.nanmean(rhos)) if rhos else float("nan")
    out["cv_mean_abs_shap"] = fs.mean(axis=0)
    out["cv_sd_abs_shap"] = fs.std(axis=0)
    logger.info("  block-CV R2 %.3f +/- %.3f | random-CV R2 %.3f | "
                "fold rank stability (Spearman) %.2f",
                out["block_r2_mean"], out["block_r2_sd"],
                out["random_r2_mean"], out["rank_rho_mean"])
    return out


# ===========================================================================
# Per-scene analysis
# ===========================================================================

def analyse_scene(scene: Scene, output_dir: Path, rng,
                  predictors=None, tag="") -> dict:
    logger.info("=== %s (%s)%s ===", scene.key, scene.typology,
                f" [{tag}]" if tag else "")
    X, y, names, rc, res_m = load_scene_table(scene, predictors)

    model, r2 = fit_form_model(X, y)
    logger.info("  in-sample R2 = %.3f (optimistic; see block CV)", r2)

    mean_abs, sv, X_s, names = shap_attribution(model, X, names, rng)
    cv = spatial_block_cv(X, y, names, rc, res_m, rng)

    order = np.argsort(mean_abs)[::-1]
    logger.info("  urban-form attribution (full-fit mean |SHAP|):")
    for i in order:
        extra = ""
        if cv is not None:
            extra = (f"   block-CV {cv['cv_mean_abs_shap'][i]:.4f} "
                     f"+/- {cv['cv_sd_abs_shap'][i]:.4f}")
        logger.info("    %-16s %.4f%s", names[i], mean_abs[i], extra)

    output_dir.mkdir(parents=True, exist_ok=True)
    return {"scene": scene.key, "typology": scene.typology, "tag": tag,
            "r2": r2, "names": names, "mean_abs_shap": mean_abs,
            "shap_values": sv, "X_sample": X_s,
            "n_valid": X.shape[0], "cv": cv}


# ===========================================================================
# Outputs
# ===========================================================================

def write_table(results, output_dir, fname="table7_form_attribution.csv"):
    import pandas as pd
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        cv = r["cv"]
        for j, (name, val) in enumerate(zip(r["names"],
                                            r["mean_abs_shap"])):
            rows.append({
                "scene": r["scene"], "typology": r["typology"],
                "predictor": name,
                "mean_abs_shap_fullfit": float(val),
                "mean_abs_shap_blockcv": (float(cv["cv_mean_abs_shap"][j])
                                          if cv else np.nan),
                "sd_abs_shap_blockcv": (float(cv["cv_sd_abs_shap"][j])
                                        if cv else np.nan),
                "r2_insample": r["r2"],
                "r2_blockcv": cv["block_r2_mean"] if cv else np.nan,
                "r2_randomcv": cv["random_r2_mean"] if cv else np.nan,
            })
    out = output_dir / fname
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.5f")
    logger.info("wrote %s", out.name)


def write_cv_table(results, output_dir):
    import pandas as pd
    rows = []
    for r in results:
        cv = r["cv"]
        if cv is None:
            continue
        rows.append({
            "scene": r["scene"], "typology": r["typology"],
            "n_valid": r["n_valid"], "n_blocks": cv["n_blocks"],
            "r2_insample": r["r2"],
            "r2_randomcv": cv["random_r2_mean"],
            "r2_blockcv_mean": cv["block_r2_mean"],
            "r2_blockcv_sd": cv["block_r2_sd"],
            "rmse_blockcv_C": cv["block_rmse_mean"],
            "fold_rank_spearman": cv["rank_rho_mean"],
        })
    out = output_dir / "table8_spatial_block_cv.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    logger.info("wrote %s", out.name)


def write_sensitivity_table(base, sens, output_dir):
    import pandas as pd
    rows = []
    for b, s_ in zip(base, sens):
        def _rank(res):
            o = np.argsort(res["mean_abs_shap"])[::-1]
            return {res["names"][i]: (k + 1, float(res["mean_abs_shap"][i]))
                    for k, i in enumerate(o)}
        rb, rs = _rank(b), _rank(s_)
        for name in rs:
            rows.append({
                "scene": b["scene"], "predictor": name,
                "rank_without_circular": rb.get(name, (np.nan, np.nan))[0],
                "shap_without_circular": rb.get(name, (np.nan, np.nan))[1],
                "rank_with_circular": rs[name][0],
                "shap_with_circular": rs[name][1],
                "circular_layer": name in CIRCULAR_PREDICTORS,
            })
    out = output_dir / "table7b_circularity_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    logger.info("wrote %s", out.name)


def write_importance_figure(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        order = np.argsort(r["mean_abs_shap"])
        names = [r["names"][i] for i in order]
        vals = [r["mean_abs_shap"][i] for i in order]
        cv = r["cv"]
        xerr = [cv["cv_sd_abs_shap"][i] for i in order] if cv else None
        ax.barh(names, vals, xerr=xerr, capsize=2)
        ttl = f"{r['typology']}\n(in-sample R2 = {r['r2']:.2f}"
        if cv:
            ttl += f"; block-CV R2 = {cv['block_r2_mean']:.2f}"
        ax.set_title(ttl + ")")
        ax.set_xlabel("mean |SHAP|  (deg C)")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Urban-form attribution of night-time SBT (per-scene "
                 "SHAP; bars = full fit, whiskers = SD across spatial-"
                 "block folds)")
    fig.tight_layout()
    out = output_dir / "figure7_shap_importance.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out.name)


def write_dependence_figures(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for r in results:
        sv = r["shap_values"]
        if sv is None:
            continue
        X_s = r["X_sample"]
        names = r["names"]
        # top 3 predictors by attribution
        top = np.argsort(r["mean_abs_shap"])[::-1][:3]
        fig, axes = plt.subplots(1, len(top), figsize=(5 * len(top), 4))
        if len(top) == 1:
            axes = [axes]
        for ax, j in zip(axes, top):
            ax.scatter(X_s[:, j], sv[:, j], s=4, alpha=0.3)
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xlabel(names[j])
            ax.set_ylabel(f"SHAP for {names[j]} (deg C)")
            ax.grid(alpha=0.3)
        fig.suptitle(f"{r['typology']} -- SHAP dependence "
                     f"(marginal effect of top form predictors)")
        fig.tight_layout()
        out = output_dir / f"figure7b_shap_dependence_{r['scene']}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s", out.name)


def write_summary(results, output_dir):
    lines = ["HotSat-1 nocturnal SUHI -- Task 4 urban-form attribution",
             "=" * 62, "",
             "Per-scene, interpretable attribution of nocturnal LST to",
             "urban-form predictors (SHAP). Within-scene and explanatory",
             "only -- no transfer, no predictive claim.", ""]
    for r in results:
        order = np.argsort(r["mean_abs_shap"])[::-1]
        lines += [
            f"{r['typology'].upper()}  ({r['scene']})",
            f"  valid pixels        : {r['n_valid']:,}",
            f"  in-sample R2        : {r['r2']:.3f}",
            *([f"  random K-fold R2    : {r['cv']['random_r2_mean']:.3f}",
               f"  spatial-block R2    : {r['cv']['block_r2_mean']:.3f} "
               f"+/- {r['cv']['block_r2_sd']:.3f}  "
               f"({r['cv']['n_blocks']} blocks of {BLOCK_SIZE_M:.0f} m)",
               f"  fold rank stability : {r['cv']['rank_rho_mean']:.2f} "
               f"(mean pairwise Spearman)"] if r["cv"] else []),
            "  form attribution (mean |SHAP|, deg C, descending):",
        ]
        for i in order:
            lines.append(f"    {r['names'][i]:<16s} "
                          f"{r['mean_abs_shap'][i]:.4f}")
        lines.append("")
    lines += [
        "INTERPRETATION",
        "  mean |SHAP| is the average magnitude by which a predictor",
        "  moves the modelled nocturnal LST -- a signed, non-linear",
        "  attribution. Compare the RANKING across typologies: which",
        "  form variables govern nighttime surface temperature in",
        "  villa vs industrial vs dense-urban fabric. The SHAP",
        "  dependence figures show the shape of each effect. The model",
        "  is a within-scene attribution device; the R2 only confirms",
        "  it captured the LST-form relationship and is NOT a",
        "  predictive result.",
    ]
    out = output_dir / "task4_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote task4_summary.txt")
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

    rng = np.random.default_rng(RANDOM_SEED)
    output_dir = Path(output_dir)
    results = [analyse_scene(s, output_dir, rng)
               for s in SCENES.values()]
    write_table(results, output_dir)
    write_cv_table(results, output_dir)
    if RUN_SENSITIVITY:
        sens = [analyse_scene(s, output_dir, rng,
                              predictors=FORM_PREDICTORS + CIRCULAR_PREDICTORS,
                              tag="sensitivity: + wc_class, impervious_frac")
                for s in SCENES.values()]
        write_sensitivity_table(results, sens, output_dir)
    write_importance_figure(results, output_dir)
    write_dependence_figures(results, output_dir)
    write_summary(results, output_dir)
    logger.info("=== Task 4 complete -> %s ===", output_dir)
    return results


if __name__ == "__main__":
    run()
