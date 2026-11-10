# Step 4. HotSat-1 pre-processing: reads the SBT and UDM products, builds the
# pixel quality mask (nodata, low-confidence land cover, oblique geometry,
# cloud, hot anomaly) and writes an exploratory long-wave LST product used
# only to define the quality flags.

from __future__ import annotations

# --- PROJ environment fix (must run BEFORE rasterio is imported) ----------
# Mixed pip/conda installs can leave PROJ_DATA/PROJ_LIB pointing at a
# proj.db that is older than the libproj that gets loaded, which makes
# every EPSG lookup fail ("DATABASE.LAYOUT.VERSION.MINOR ... expected >=5").
# Point PROJ at the data directory shipped with the rasterio/pyproj wheel
# when one exists; otherwise clear the stale variables.
import importlib.util
import os


def _fix_proj():
    for pkg, sub in (("rasterio", "proj_data"),
                     ("pyproj", "proj_dir/share/proj")):
        try:
            spec = importlib.util.find_spec(pkg)
        except (ValueError, ImportError):
            continue
        if spec and spec.origin:
            d = Path(spec.origin).parent / sub
            if (d / "proj.db").exists():
                os.environ["PROJ_DATA"] = str(d)
                os.environ["PROJ_LIB"] = str(d)
                return
    for v in ("PROJ_DATA", "PROJ_LIB"):
        os.environ.pop(v, None)


import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_fix_proj()

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

logger = logging.getLogger(__name__)


# ===========================================================================
# Physical constants and sensor configuration
# ===========================================================================

H_PLANCK = 6.62607015e-34          # J s
C_LIGHT  = 2.99792458e8            # m/s
K_BOLTZ  = 1.380649e-23            # J/K
C1 = 2 * H_PLANCK * C_LIGHT ** 2   # 1.19104e-16 W m^2/sr
C2 = H_PLANCK * C_LIGHT / K_BOLTZ  # 1.43878e-2 m K

# HotSat-1 effective wavelength. Placeholder pending the SatVu SRF.
# Used by the Planck conversions and the JM-S b_gamma term.
LAMBDA_EFF_UM = 10.5
LAMBDA_EFF_M = LAMBDA_EFF_UM * 1e-6

# Instrument noise-equivalent delta-T (K). Confirm with SatVu; 0.7 K is a
# reasonable placeholder for a modern microbolometer TIR instrument.
NEDT_K = 0.7

# Physical clip range for UAE autumn-night Tb (K). Anything outside is
# fill / bad pixels (the 200 K fill values in the metadata, and the very
# hot tail that produced 85-88 C LST). 278-313 K as used in the main runs.
TB_VALID_RANGE = (278.0, 313.0)

# --- data location (see config.py) ------------------------------------------
from config import DATA_ROOT

# --- ASTER GED control -----------------------------------------------------
# The ASTER GED magnitude correction needs an interactive Earthdata login
# and a large tile download with no progress output -- it was the cause of
# silent multi-hour hangs. It only shifts absolute emissivity by <= 0.03
# (about 1-2 K in absolute LST) and does not change spatial patterns.
# Keep False unless the ASTER GED tile has been downloaded.
USE_ASTER_GED = False

# --- ERA5 control --------------------------------------------------------
# If True, fetch_era5_pwv will attempt a live cdsapi download when a
# scene has no manual pwv_kg_m2. If False, it skips the API entirely and
# uses PWV_FALLBACK. A per-scene pwv_kg_m2 ALWAYS wins over both. Set
# this False if the CDS API hangs / queues -- and supply manual values.
USE_ERA5_API = False
PWV_FALLBACK = 12.0          # autumn-UAE night climatology, kg/m^2


# ===========================================================================
# Scene configuration  (mirrors predictor_stack.SCENES)
# ===========================================================================

@dataclass
class LSTScene:
    key: str
    scene_id: str
    typology: str
    datetime_utc: str
    view_zenith_deg: float
    hotsat_tiff: Path          # HotSat-1 SBT GeoTIFF (brightness temp, K)
    predictor_nc: Path         # *_predictors_10m.nc from predictor_stack.py
    udm_tiff: Optional[Path] = None   # HotSat unusable-data mask, if present
    pwv_kg_m2: Optional[float] = None  # manual ERA5 PWV override (see below)


# Scene definitions. predictor_nc files are written by predictor_stack.py
# to DATA_ROOT/predictors/.
#
# pwv_kg_m2: per-scene ERA5 total-column water vapour (kg/m^2), looked up
# with era5_pwv_lookup.py at the acquisition time (hourly fields
# interpolated to the scene time, +/-0.25 deg box). A hard-coded value
# ALWAYS overrides the live fetch and the fallback. Late-Oct/Nov coastal
# UAE values are high (27-31 kg/m^2); the 12 kg/m^2 fallback is NOT
# representative for these dates and is used only if a value is missing.
SCENES: dict[str, LSTScene] = {
    "AD_suburban": LSTScene(
        key="AD_suburban",
        scene_id="20231029_213936_HS1_SBT",
        typology="sub-urban",
        datetime_utc="2023-10-29T21:39:36Z",
        view_zenith_deg=34.21,
        hotsat_tiff=DATA_ROOT.joinpath("hotsat/20231029_213936_HS1_SBT.tiff"),
        predictor_nc=DATA_ROOT.joinpath("predictors/"
                          "20231029_213936_HS1_SBT_predictors_10m.nc"),
        udm_tiff=DATA_ROOT.joinpath("hotsat/"
                      "20231029_213936_HS1_SBT_udm.tiff"),
        pwv_kg_m2=27.58,        # ERA5 TCWV, 2023-10-29 21:39 UTC
    ),
    "AD_industrial": LSTScene(
        key="AD_industrial",
        scene_id="20231030_212656_HS1_SBT",
        typology="industrial",
        datetime_utc="2023-10-30T21:26:56Z",
        view_zenith_deg=0.85,
        hotsat_tiff=DATA_ROOT.joinpath("hotsat/20231030_212656_HS1_SBT.tiff"),
        predictor_nc=DATA_ROOT.joinpath("predictors/"
                          "20231030_212656_HS1_SBT_predictors_10m.nc"),
        udm_tiff=DATA_ROOT.joinpath("hotsat/"
                      "20231030_212656_HS1_SBT_udm.tiff"),
        pwv_kg_m2=30.96,        # ERA5 TCWV, 2023-10-30 21:27 UTC
    ),
    "DXB_urban": LSTScene(
        key="DXB_urban",
        scene_id="20231120_213935_HS1_SBT",
        typology="urban",
        datetime_utc="2023-11-20T21:39:35Z",
        view_zenith_deg=35.43,
        hotsat_tiff=DATA_ROOT.joinpath("hotsat/20231120_213935_HS1_SBT.tiff"),
        predictor_nc=DATA_ROOT.joinpath("predictors/"
                          "20231120_213935_HS1_SBT_predictors_10m.nc"),
        udm_tiff=DATA_ROOT.joinpath("hotsat/"
                      "20231120_213935_HS1_SBT_udm.tiff"),
        pwv_kg_m2=30.09,        # ERA5 TCWV, 2023-11-20 21:40 UTC
    ),
}


# ===========================================================================
# Planck helpers
# ===========================================================================

def planck_radiance(wavelength_m: float, T_K: np.ndarray) -> np.ndarray:
    return C1 / (wavelength_m ** 5 *
                 (np.exp(C2 / (wavelength_m * T_K)) - 1.0))


def inverse_planck(wavelength_m: float, L: np.ndarray) -> np.ndarray:
    return C2 / (wavelength_m * np.log(C1 / (wavelength_m ** 5 * L) + 1.0))


# ===========================================================================
# Stage 1: load HotSat-1 Tb
# ===========================================================================

def load_tb(tb_path: Path,
            valid_range_k: tuple[float, float] = TB_VALID_RANGE):
    with rasterio.open(tb_path) as src:
        tb = src.read(1).astype(np.float32)
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "shape": (src.height, src.width),
            "bounds": src.bounds,
        }
    invalid = (np.isnan(tb) |
               (tb < valid_range_k[0]) | (tb > valid_range_k[1]))
    tb = np.where(invalid, np.nan, tb)
    logger.info("  Tb loaded: %.1f%% valid, range %.1f-%.1f K",
                100 * np.isfinite(tb).mean(),
                np.nanmin(tb), np.nanmax(tb))
    return tb, meta


# ===========================================================================
# Stage 2: emissivity map
# ===========================================================================

# ESA WorldCover class codes (as stored in the predictor-stack wc_class)
WC_TREE, WC_SHRUB, WC_GRASS, WC_CROP = 10, 20, 30, 40
WC_BUILT, WC_BARE, WC_SNOW, WC_WATER = 50, 60, 70, 80
WC_WETLAND, WC_MANGROVE, WC_MOSS = 90, 95, 100

# Per-class broadband emissivity at ~10.5 um for arid UAE surfaces, with
# 1-sigma uncertainty. Mixed vegetation/soil is handled separately via FVC.
EPS_CLASS = {
    WC_WATER:    (0.991, 0.003),
    WC_TREE:     (0.985, 0.005),
    WC_SHRUB:    (0.972, 0.008),
    WC_GRASS:    (0.972, 0.008),
    WC_CROP:     (0.975, 0.008),
    WC_BUILT:    (0.960, 0.015),
    WC_BARE:     (0.945, 0.012),
    WC_WETLAND:  (0.980, 0.008),
    WC_MANGROVE: (0.985, 0.006),
    WC_MOSS:     (0.970, 0.010),
    WC_SNOW:     (0.989, 0.005),
}
_EPS_DEFAULT = (0.960, 0.015)

# Sobrino NDVI-method parameters for the FVC blend (built / sparse pixels
# with intermediate NDVI get a vegetation/soil mixture).
NDVI_SOIL, NDVI_VEG = 0.15, 0.70
EPS_VEG, EPS_SOIL = 0.985, 0.960
DELTA_EPS = 0.015         # veg/soil contrast, scales the cavity term
NDVI_MIX_LO, NDVI_MIX_HI = 0.20, 0.50   # NDVI band treated as mixed


def read_predictor_layer(predictor_nc: Path, layer: str):
    import xarray as xr

    ds = xr.open_dataset(predictor_nc)
    if layer not in ds:
        raise KeyError(f"Layer '{layer}' not in {predictor_nc.name}. "
                       f"Available: {list(ds.data_vars)}")
    arr = ds[layer].values.astype(np.float32)
    transform = rasterio.transform.Affine(*ds.attrs["transform"])
    crs = ds.attrs["crs"]
    ds.close()
    return arr, transform, crs


def build_emissivity(scene: LSTScene,
                     tb_meta: dict,
                     aster_eps: Optional[np.ndarray] = None,
                     aster_transform=None,
                     aster_crs=None):
    # --- read NDVI + land cover from the predictor stack --------------
    ndvi, p_transform, p_crs = read_predictor_layer(scene.predictor_nc,
                                                    "ndvi")
    wc, _, _ = read_predictor_layer(scene.predictor_nc, "wc_class")

    # --- per-class emissivity on the predictor grid -------------------
    eps = np.full(ndvi.shape, _EPS_DEFAULT[0], dtype=np.float32)
    sig = np.full(ndvi.shape, _EPS_DEFAULT[1], dtype=np.float32)
    wc_round = np.round(wc).astype(np.int32)
    for code, (e, s) in EPS_CLASS.items():
        m = wc_round == code
        eps[m] = e
        sig[m] = s

    # --- Sobrino FVC blend for mixed-NDVI pixels ----------------------
    mix = (ndvi > NDVI_MIX_LO) & (ndvi < NDVI_MIX_HI)
    if mix.any():
        fvc = ((ndvi - NDVI_SOIL) / (NDVI_VEG - NDVI_SOIL + 1e-6)) ** 2
        fvc = np.clip(fvc, 0.0, 1.0)
        cavity = 4.0 * DELTA_EPS * fvc * (1.0 - fvc)
        eps[mix] = (EPS_VEG * fvc[mix] +
                    EPS_SOIL * (1.0 - fvc[mix]) + cavity[mix])
        sig[mix] = 0.010
    # Dense vegetation (high NDVI) -> vegetation emissivity
    dense = ndvi >= NDVI_MIX_HI
    eps[dense] = EPS_VEG
    sig[dense] = 0.005

    eps = np.clip(eps, 0.90, 0.999).astype(np.float32)

    # --- ASTER GED magnitude correction -------------------------------
    if aster_eps is not None:
        eps = _aster_ged_adjust(eps, p_transform, p_crs,
                                aster_eps, aster_transform, aster_crs)
        # ASTER correction adds a small systematic-uncertainty floor
        sig = np.sqrt(sig ** 2 + 0.006 ** 2).astype(np.float32)

    # --- resample emissivity onto the HotSat grid ---------------------
    eps_h = _resample(eps, p_transform, p_crs, tb_meta,
                      Resampling.bilinear)
    sig_h = _resample(sig, p_transform, p_crs, tb_meta,
                      Resampling.bilinear)
    logger.info("  emissivity: mean %.4f (range %.3f-%.3f), "
                "sigma mean %.4f",
                float(np.nanmean(eps_h)), float(np.nanmin(eps_h)),
                float(np.nanmax(eps_h)), float(np.nanmean(sig_h)))
    return eps_h, sig_h


def _resample(arr, src_transform, src_crs, dst_meta, resampling):
    dst = np.full(dst_meta["shape"], np.nan, dtype=np.float32)
    reproject(
        source=arr.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=dst_meta["transform"],
        dst_crs=dst_meta["crs"],
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


def _aster_ged_adjust(eps_fine, eps_transform, eps_crs,
                      aster_eps, aster_transform, aster_crs):
    coarse = np.full(aster_eps.shape, np.nan, dtype=np.float32)
    reproject(
        source=eps_fine, destination=coarse,
        src_transform=eps_transform, src_crs=eps_crs,
        dst_transform=aster_transform, dst_crs=aster_crs,
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    offset = np.clip(aster_eps - coarse, -0.03, 0.03)
    offset = np.nan_to_num(offset, nan=0.0)

    offset_fine = np.zeros_like(eps_fine)
    reproject(
        source=offset, destination=offset_fine,
        src_transform=aster_transform, src_crs=aster_crs,
        dst_transform=eps_transform, dst_crs=eps_crs,
        resampling=Resampling.bilinear,
    )
    return np.clip(eps_fine + offset_fine, 0.90, 0.999).astype(np.float32)


def fetch_aster_ged(tb_meta: dict):
    if not USE_ASTER_GED:
        logger.info("  ASTER GED correction disabled (USE_ASTER_GED=False)"
                    " -- emissivity from WorldCover/NDVI only")
        return None, None, None
    try:
        import earthaccess
        import h5py
    except ImportError:
        logger.warning("  earthaccess / h5py not installed -- skipping "
                        "ASTER GED correction.")
        return None, None, None

    # AOI bbox in lon/lat
    left, bottom, right, top = transform_bounds(
        tb_meta["crs"], "EPSG:4326",
        tb_meta["bounds"].left, tb_meta["bounds"].bottom,
        tb_meta["bounds"].right, tb_meta["bounds"].top,
    )
    try:
        earthaccess.login()
        results = earthaccess.search_data(
            short_name="AG100", version="003",
            bounding_box=(left, bottom, right, top),
        )
        if not results:
            logger.warning("  No ASTER GED tile for AOI -- skipping "
                            "magnitude correction.")
            return None, None, None
        files = earthaccess.download(results, "/tmp/aster_ged")
    except Exception as e:  # noqa: BLE001
        logger.warning("  ASTER GED fetch failed (%s) -- skipping "
                        "magnitude correction.", e)
        return None, None, None

    # Read emissivity bands 13 & 14 from the (first) HDF5 tile.
    # ASTER GED layout: /Emissivity/Mean  shape (5, rows, cols) for
    # bands 10-14; scale factor 0.001. Geolocation via /Geolocation.
    try:
        with h5py.File(files[0], "r") as f:
            emis = f["Emissivity/Mean"][:]            # (5, H, W), int
            lat = f["Geolocation/Latitude"][:]
            lon = f["Geolocation/Longitude"][:]
        # bands 13,14 are indices 3,4; scale 0.001
        eps = 0.5 * (emis[3] + emis[4]).astype(np.float32) * 0.001
        eps = np.where(eps > 0.5, eps, np.nan)        # mask fill

        # ASTER GED tiles are a regular 0.01 deg grid; derive transform
        res = 0.01
        west, north = float(lon.min()), float(lat.max())
        transform = rasterio.transform.from_origin(west, north, res, res)
        logger.info("  ASTER GED: %d x %d, emissivity mean %.4f",
                    eps.shape[0], eps.shape[1], float(np.nanmean(eps)))
        return eps, transform, "EPSG:4326"
    except Exception as e:  # noqa: BLE001
        logger.warning("  ASTER GED HDF parse failed (%s) -- skipping "
                        "magnitude correction.", e)
        return None, None, None


# ===========================================================================
# Stage 3: atmospheric correction
# ===========================================================================

def fetch_era5_pwv(scene: LSTScene, tb_meta: dict) -> float:
    # --- 1. manual per-scene override ---------------------------------
    if scene.pwv_kg_m2 is not None:
        logger.info("  PWV: %.1f kg/m^2 (manual per-scene value)",
                    scene.pwv_kg_m2)
        return float(scene.pwv_kg_m2)

    # --- 2. live ERA5, only if enabled --------------------------------
    if not USE_ERA5_API:
        logger.warning("  PWV: USE_ERA5_API is False and no manual value "
                        "set -- using fallback %.1f kg/m^2. Supply a real "
                        "per-scene pwv_kg_m2 for an accurate retrieval.",
                        PWV_FALLBACK)
        return PWV_FALLBACK

    try:
        import cdsapi
        import xarray as xr
    except ImportError:
        logger.warning("  PWV: cdsapi not installed -- using fallback "
                        "%.1f kg/m^2.", PWV_FALLBACK)
        return PWV_FALLBACK

    t = __import__("datetime").datetime.strptime(
        scene.datetime_utc, "%Y-%m-%dT%H:%M:%SZ")
    left, bottom, right, top = transform_bounds(
        tb_meta["crs"], "EPSG:4326",
        tb_meta["bounds"].left, tb_meta["bounds"].bottom,
        tb_meta["bounds"].right, tb_meta["bounds"].top,
    )
    out = Path(f"/tmp/era5_{scene.scene_id}.nc")
    try:
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": "total_column_water_vapour",
                "year": f"{t.year}", "month": f"{t.month:02d}",
                "day": f"{t.day:02d}", "time": f"{t.hour:02d}:00",
                "area": [top, left, bottom, right],   # N, W, S, E
                "format": "netcdf",
            },
            str(out),
        )
        ds = xr.open_dataset(out)
        var = "tcwv" if "tcwv" in ds else list(ds.data_vars)[0]
        pwv = float(ds[var].mean().values)
        ds.close()
        logger.info("  PWV: %.1f kg/m^2 (ERA5 live)", pwv)
        return pwv
    except Exception as e:  # noqa: BLE001
        logger.warning("  PWV: ERA5 fetch failed (%s) -- using fallback "
                        "%.1f kg/m^2.", e, PWV_FALLBACK)
        return PWV_FALLBACK


def atmospheric_functions(pwv_kg_m2: float,
                          view_zenith_deg: float) -> dict[str, float]:
    w = (pwv_kg_m2 / 10.0) / np.cos(np.radians(view_zenith_deg))
    psi1 = 0.04019 * w ** 2 + 0.02916 * w + 1.01523
    psi2 = -0.38333 * w ** 2 - 1.50294 * w + 0.20324
    psi3 = 0.00918 * w ** 2 + 1.36072 * w - 0.27514
    logger.info("  atm functions: psi1=%.4f psi2=%.4f psi3=%.4f "
                "(PWV_slant=%.2f g/cm^2)", psi1, psi2, psi3, w)
    return {"psi1": psi1, "psi2": psi2, "psi3": psi3}


# ===========================================================================
# Stage 4: LST retrieval (JM-S single-channel)
# ===========================================================================

def retrieve_lst(tb_k: np.ndarray,
                 eps: np.ndarray,
                 atm: dict[str, float],
                 lambda_eff_m: float = LAMBDA_EFF_M) -> np.ndarray:
    L = planck_radiance(lambda_eff_m, tb_k) * 1e-6   # W m-2 sr-1 um-1, as the psi coefficients expect
    b_gamma = C2 / lambda_eff_m
    gamma = tb_k ** 2 / (b_gamma * L)
    delta = tb_k - tb_k ** 2 / b_gamma
    lst = gamma * (1.0 / eps * (atm["psi1"] * L + atm["psi2"])
                   + atm["psi3"]) + delta
    return lst.astype(np.float32)


# ===========================================================================
# Stage 5: uncertainty propagation
# ===========================================================================

def propagate_uncertainty(lst_k: np.ndarray,
                          eps: np.ndarray,
                          sigma_eps: np.ndarray,
                          view_zenith_deg: float,
                          sigma_tb_k: float = NEDT_K) -> np.ndarray:
    s_eps = np.abs(lst_k / (4.0 * eps)) * sigma_eps
    # atmospheric term scales with slant-path factor
    slant = 1.0 / np.cos(np.radians(view_zenith_deg))
    s_atm = 0.4 * slant                       # ~0.4 K at nadir
    sigma = np.sqrt(s_eps ** 2 + s_atm ** 2 + sigma_tb_k ** 2)
    return sigma.astype(np.float32)


# ===========================================================================
# Stage 6: QA mask and output
# ===========================================================================

QA_NODATA      = 1 << 0
QA_EPS_LOW     = 1 << 1
QA_OBLIQUE     = 1 << 2
QA_CLOUD       = 1 << 3
QA_HOT_ANOMALY = 1 << 4

# Nighttime autumn-UAE land-surface temperature should not exceed ~40 C.
# Pixels above this are flagged as hot anomalies (flares, industrial heat
# sources, vehicles, sensor artefacts): retained and recorded in the QA
# mask but excluded from the training target. 40 C sits at ~p99 for the
# two affected scenes, so only the genuine tail is flagged -- the real
# urban-heat body of the Dubai scene (median 36 C) is left intact.
LST_HOT_ANOMALY_C = 40.0


def build_qa(tb_k: np.ndarray,
             eps: np.ndarray,
             view_zenith_deg: float,
             udm: Optional[np.ndarray],
             lst_c: Optional[np.ndarray] = None) -> np.ndarray:
    qa = np.zeros(tb_k.shape, dtype=np.uint8)
    qa |= (np.isnan(tb_k).astype(np.uint8) * QA_NODATA)
    qa |= (np.isnan(eps).astype(np.uint8) * QA_EPS_LOW)
    if view_zenith_deg > 30.0:
        qa |= QA_OBLIQUE
    if udm is not None:
        qa |= ((udm > 0).astype(np.uint8) * QA_CLOUD)
    if lst_c is not None:
        hot = (np.isfinite(lst_c) &
               (lst_c > LST_HOT_ANOMALY_C)).astype(np.uint8)
        qa |= (hot * QA_HOT_ANOMALY)
    return qa


def write_cog(array: np.ndarray, meta: dict, path: Path,
              dtype: str = "float32", nodata=np.nan) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": array.shape[0],
        "width": array.shape[1], "count": 1, "dtype": dtype,
        "crs": meta["crs"], "transform": meta["transform"],
        "compress": "deflate", "tiled": True,
        "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER",
    }
    if dtype != "uint8" and nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


# ===========================================================================
# Driver
# ===========================================================================

def process_scene(scene: LSTScene, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    logger.info("=== HotSat LST: %s (%s) ===", scene.key, scene.typology)

    logger.info("  reading %s", scene.hotsat_tiff)
    # Stage 1 -- load Tb
    tb, meta = load_tb(scene.hotsat_tiff)

    # Stage 2 -- emissivity (predictor-stack NDVI/LC; ASTER optional)
    logger.info("  stage 2: emissivity")
    aster_eps, aster_tr, aster_crs = fetch_aster_ged(meta)
    eps, sigma_eps = build_emissivity(scene, meta,
                                      aster_eps, aster_tr, aster_crs)

    # Stage 3 -- atmospheric correction
    pwv = fetch_era5_pwv(scene, meta)
    atm = atmospheric_functions(pwv, scene.view_zenith_deg)

    # Stage 4 -- LST retrieval
    lst_k = retrieve_lst(tb, eps, atm)
    lst_c = lst_k - 273.15
    logger.info("  LST: median %.2f C, mean %.2f C (range %.2f-%.2f C)",
                float(np.nanmedian(lst_c)), float(np.nanmean(lst_c)),
                float(np.nanmin(lst_c)), float(np.nanmax(lst_c)))

    # Stage 5 -- uncertainty
    sigma_lst = propagate_uncertainty(lst_k, eps, sigma_eps,
                                      scene.view_zenith_deg)
    logger.info("  sigma_LST: mean %.2f K", float(np.nanmean(sigma_lst)))

    # Stage 6 -- QA + write
    udm = None
    if scene.udm_tiff is not None and Path(scene.udm_tiff).exists():
        with rasterio.open(scene.udm_tiff) as src:
            udm = src.read(1)
    qa = build_qa(tb, eps, scene.view_zenith_deg, udm, lst_c=lst_c)
    hot_frac = 100.0 * ((qa & QA_HOT_ANOMALY) > 0).mean()
    logger.info("  hot-anomaly pixels flagged: %.2f%% "
                "(LST > %.0f C -- excluded from training target)",
                hot_frac, LST_HOT_ANOMALY_C)

    out = {
        "lst":   output_dir / f"{scene.scene_id}_LST_C.tif",
        "sigma": output_dir / f"{scene.scene_id}_LST_sigma_K.tif",
        "eps":   output_dir / f"{scene.scene_id}_emissivity.tif",
        "qa":    output_dir / f"{scene.scene_id}_LST_QA.tif",
    }
    write_cog(lst_c, meta, out["lst"])
    write_cog(sigma_lst, meta, out["sigma"])
    write_cog(eps, meta, out["eps"])
    write_cog(qa, meta, out["qa"], dtype="uint8", nodata=None)
    logger.info("  saved -> %s", output_dir)
    return out


def run_with_log(output_dir: Path):
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
    for scene in SCENES.values():
        process_scene(scene, output_dir)
    logger.info("=== all scenes complete -> %s ===", output_dir)


def _main():
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    out_dir = DATA_ROOT / "lst"
    logger.info("DATA_ROOT = %s", DATA_ROOT)
    for scene in SCENES.values():
        process_scene(scene, out_dir)
    logger.info("=== all scenes complete -> %s ===", out_dir)


if __name__ == "__main__":
    _main()
