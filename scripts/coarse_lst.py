# Step 5. Download and grid VIIRS VNP21A1N night LST (1 km) for the comparison
# of scene offsets. Requires a NASA Earthdata login (earthaccess).

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from config import DATA_ROOT
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

logger = logging.getLogger(__name__)


# ===========================================================================
# Scene configuration  (mirrors hotsat_lst.SCENES / predictor_stack.SCENES)
# ===========================================================================

@dataclass
class CoarseScene:
    key: str
    scene_id: str
    typology: str
    datetime_utc: str
    predictor_nc: Path          # *_predictors_10m.nc -- defines the grid


SCENES: dict[str, CoarseScene] = {
    "AD_suburban": CoarseScene(
        key="AD_suburban",
        scene_id="20231029_213936_HS1_SBT",
        typology="sub-urban",
        datetime_utc="2023-10-29T21:39:36Z",
        predictor_nc=(DATA_ROOT / "predictors/20231029_213936_HS1_SBT_predictors_10m.nc"),
    ),
    "AD_industrial": CoarseScene(
        key="AD_industrial",
        scene_id="20231030_212656_HS1_SBT",
        typology="industrial",
        datetime_utc="2023-10-30T21:26:56Z",
        predictor_nc=(DATA_ROOT / "predictors/20231030_212656_HS1_SBT_predictors_10m.nc"),
    ),
    "DXB_urban": CoarseScene(
        key="DXB_urban",
        scene_id="20231120_213935_HS1_SBT",
        typology="urban",
        datetime_utc="2023-11-20T21:39:35Z",
        predictor_nc=(DATA_ROOT / "predictors/20231120_213935_HS1_SBT_predictors_10m.nc"),
    ),
}


# ===========================================================================
# Coarse-product specifications
# ===========================================================================

@dataclass
class CoarseProduct:
    name: str                       # short label for logs / filenames
    short_name: str                 # earthaccess collection short name
    version: str
    lst_keywords: tuple[str, ...]    # substrings identifying the LST SDS
    qc_keywords: tuple[str, ...]     # substrings identifying the QC SDS
    scale: float                    # raw DN -> Kelvin
    fill_dn: float                  # raw fill value (pre-scale)
    exclude_keywords: tuple[str, ...] = field(
        default_factory=lambda: ("err", "Emis", "emis", "view", "View"))


MODIS = CoarseProduct(
    name="MODIS",
    short_name="MYD11A1",            # Aqua daily L3 1 km LST&E
    version="061",
    lst_keywords=("LST_Night_1km",),
    qc_keywords=("QC_Night",),
    scale=0.02,
    fill_dn=0.0,
)

VIIRS = CoarseProduct(
    name="VIIRS",
    short_name="VNP21A1N",           # S-NPP daily L3 1 km LST&E, night
    version="002",
    lst_keywords=("LST_1km", "LST"),
    qc_keywords=("QC",),
    scale=0.02,
    fill_dn=0.0,
)


# ===========================================================================
# Analysis grid from the predictor NetCDF
# ===========================================================================

@dataclass
class GridSpec:
    transform: object
    crs: object
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        left = self.transform.c
        top = self.transform.f
        res_x = self.transform.a
        res_y = -self.transform.e
        right = left + self.width * res_x
        bottom = top - self.height * res_y
        return transform_bounds(self.crs, "EPSG:4326",
                                left, bottom, right, top)


def grid_from_predictor_nc(predictor_nc: Path) -> GridSpec:
    import xarray as xr

    ds = xr.open_dataset(predictor_nc)
    transform = rasterio.transform.Affine(*ds.attrs["transform"])
    crs = ds.attrs["crs"]
    first = list(ds.data_vars)[0]
    h, w = ds[first].shape
    ds.close()
    logger.info("  analysis grid: %d x %d, crs=%s", h, w, crs)
    return GridSpec(transform, crs, h, w)


# ===========================================================================
# Granule search and download
# ===========================================================================

def search_granules(product: CoarseProduct,
                     grid: GridSpec,
                     scene_dt_utc: str,
                     window_days: int = 1):
    import earthaccess

    t = datetime.strptime(scene_dt_utc, "%Y-%m-%dT%H:%M:%SZ")
    t0 = (t - timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%S")
    t1 = (t + timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%S")
    bbox = grid.bounds_lonlat

    earthaccess.login()
    results = earthaccess.search_data(
        short_name=product.short_name,
        version=product.version,
        temporal=(t0, t1),
        bounding_box=tuple(bbox),
    )
    logger.info("  %s: %d granule(s) within +/-%d d of %s",
                product.name, len(results), window_days, scene_dt_utc)
    return results


def _granule_date(result) -> Optional[datetime]:
    try:
        rng = result["umm"]["TemporalExtent"]["RangeDateTime"]
        return datetime.strptime(rng["BeginningDateTime"][:10], "%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return None


def select_and_download(product: CoarseProduct,
                        results,
                        scene_dt_utc: str,
                        download_dir: Path) -> list[Path]:
    if not results:
        return []
    t = datetime.strptime(scene_dt_utc, "%Y-%m-%dT%H:%M:%SZ")

    dated = [(r, _granule_date(r)) for r in results]
    dated = [(r, d) for r, d in dated if d is not None]
    if not dated:
        chosen = results            # fall back: take everything
    else:
        # pick the single date closest to the scene UTC date
        best_date = min((d for _, d in dated),
                        key=lambda d: abs((d - t).days))
        chosen = [r for r, d in dated if d == best_date]
        logger.info("  %s: using granule date %s (%d tile[s])",
                    product.name, best_date.strftime("%Y-%m-%d"),
                    len(chosen))

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    import earthaccess
    files = earthaccess.download(chosen, str(download_dir))
    return [Path(f) for f in files]


# ===========================================================================
# Subdataset reading
# ===========================================================================

def _find_subdataset(file_path: Path,
                     keywords: tuple[str, ...],
                     exclude: tuple[str, ...] = ()) -> Optional[str]:
    with rasterio.open(file_path) as src:
        subs = src.subdatasets
    for kw in keywords:                       # keyword priority order
        for sd in subs:
            low = sd.lower()
            if kw.lower() in low and not any(x.lower() in low
                                             for x in exclude):
                return sd
    return None


def read_lst_subdataset(file_path: Path,
                        product: CoarseProduct):
    lst_sd = _find_subdataset(file_path, product.lst_keywords,
                              product.exclude_keywords)
    if lst_sd is None:
        with rasterio.open(file_path) as src:
            avail = src.subdatasets
        raise RuntimeError(
            f"No LST subdataset matching {product.lst_keywords} in "
            f"{file_path.name}. Subdatasets present: {avail}"
        )

    with rasterio.open(lst_sd) as src:
        raw = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs

    lst = raw * product.scale                  # -> Kelvin
    lst = np.where(raw == product.fill_dn, np.nan, lst)

    # QC mask (best effort -- proceed without it if QC SDS not found)
    qc_sd = _find_subdataset(file_path, product.qc_keywords,
                             product.exclude_keywords)
    if qc_sd is not None:
        with rasterio.open(qc_sd) as src:
            qc = src.read(1).astype(np.uint16)
        good = (qc & 0b11) == 0                # mandatory QA == "good"
        kept = 100.0 * good.mean()
        logger.info("    %s QC: %.1f%% pixels good quality",
                    product.name, kept)
        lst = np.where(good, lst, np.nan)
    else:
        logger.warning("    %s: no QC subdataset found -- proceeding "
                        "without QC masking.", product.name)

    # sanity range for nighttime LST (Kelvin)
    lst = np.where((lst > 250.0) & (lst < 340.0), lst, np.nan)
    return lst.astype(np.float32), transform, crs


# ===========================================================================
# Mosaic + reproject onto the analysis grid
# ===========================================================================

def coarse_lst_on_grid(product: CoarseProduct,
                       files: list[Path],
                       grid: GridSpec) -> np.ndarray:
    mosaic = np.full(grid.shape, np.nan, dtype=np.float32)

    for fp in files:
        lst_k, src_transform, src_crs = read_lst_subdataset(fp, product)
        tile = np.full(grid.shape, np.nan, dtype=np.float32)
        reproject(
            source=lst_k,
            destination=tile,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        fill = np.isnan(mosaic) & np.isfinite(tile)
        mosaic[fill] = tile[fill]

    cov = 100.0 * np.isfinite(mosaic).mean()
    lst_c = mosaic - 273.15
    if cov > 0:
        logger.info("  %s LST on grid: %.1f%% coverage, "
                    "median %.2f C (range %.2f-%.2f C)",
                    product.name, cov, float(np.nanmedian(lst_c)),
                    float(np.nanmin(lst_c)), float(np.nanmax(lst_c)))
    if cov < 90.0:
        logger.warning("  %s: only %.1f%% grid coverage -- cloud or "
                        "missing tile. Check QC / granule selection.",
                        product.name, cov)
    return lst_c


# ===========================================================================
# Output
# ===========================================================================

def write_cog(array: np.ndarray, grid: GridSpec, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": grid.height, "width": grid.width,
        "count": 1, "dtype": "float32", "crs": grid.crs,
        "transform": grid.transform, "nodata": np.nan,
        "compress": "deflate", "predictor": 2, "tiled": True,
        "blockxsize": 256, "blockysize": 256, "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)


# ===========================================================================
# Driver
# ===========================================================================

def process_scene(scene: CoarseScene,
                  output_dir: Path,
                  download_dir: Path) -> dict[str, Path]:
    logger.info("=== Coarse LST: %s (%s) ===", scene.key, scene.typology)
    grid = grid_from_predictor_nc(scene.predictor_nc)

    out: dict[str, Path] = {}
    for product in (MODIS, VIIRS):
        results = search_granules(product, grid, scene.datetime_utc)
        files = select_and_download(product, results, scene.datetime_utc,
                                    download_dir / product.name)
        if not files:
            logger.warning("  %s: no granules -- skipping this product "
                            "for %s.", product.name, scene.key)
            continue
        lst_c = coarse_lst_on_grid(product, files, grid)
        path = Path(output_dir) / (f"{scene.scene_id}_"
                                   f"{product.name}_LST_C.tif")
        write_cog(lst_c, grid, path)
        out[product.name] = path
        logger.info("  saved -> %s", path)

    return out


def _main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    output_dir = (DATA_ROOT / "coarse_lst")
    download_dir = (DATA_ROOT / "downloads")
    for scene in SCENES.values():
        process_scene(scene, output_dir, download_dir)


if __name__ == "__main__":
    _main()
