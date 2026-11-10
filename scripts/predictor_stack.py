# Step 2. Build the 10 m predictor stack per scene (Sentinel-2 albedo and
# indices, WSF-3D form layers, sky-view factor, WorldCover classes, distance
# to water, DEM) and write it to NetCDF.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from config import DATA_ROOT
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy import ndimage

logger = logging.getLogger(__name__)


# ===========================================================================
# Scene configuration
# ===========================================================================

@dataclass
class SceneConfig:
    key: str
    scene_id: str
    typology: str
    role: str                       # "train" or "test"
    datetime_utc: str
    hotsat_tiff: Path               # path to the HotSat SBT GeoTIFF
    epsg: int = 32640
    # Sentinel-2 search window (days) and cloud limit (%)
    s2_window_days: int = 7
    s2_max_cloud: float = 10.0


SCENES: dict[str, SceneConfig] = {
    "AD_suburban": SceneConfig(
        key="AD_suburban",
        scene_id="20231029_213936_HS1_SBT",
        typology="sub-urban",
        role="train",
        datetime_utc="2023-10-29T21:39:36Z",
        hotsat_tiff=(DATA_ROOT / "hotsat/20231029_213936_HS1_SBT.tiff"),
    ),
    "AD_industrial": SceneConfig(
        key="AD_industrial",
        scene_id="20231030_212656_HS1_SBT",
        typology="industrial",
        role="train",
        datetime_utc="2023-10-30T21:26:56Z",
        hotsat_tiff=(DATA_ROOT / "hotsat/20231030_212656_HS1_SBT.tiff"),
    ),
    "DXB_urban": SceneConfig(
        key="DXB_urban",
        scene_id="20231120_213935_HS1_SBT",
        typology="urban",
        role="test",
        datetime_utc="2023-11-20T21:39:35Z",
        hotsat_tiff=(DATA_ROOT / "hotsat/20231120_213935_HS1_SBT.tiff"),
    ),
}


# ===========================================================================
# Analysis grid
# ===========================================================================

@dataclass
class GridSpec:
    transform: object
    width: int
    height: int
    crs: object
    res: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left = self.transform.c
        top = self.transform.f
        right = left + self.width * self.res
        bottom = top - self.height * self.res
        return (left, bottom, right, top)


def analysis_grid_from_hotsat(hotsat_tiff: Path, res: float = 10.0) -> GridSpec:
    with rasterio.open(hotsat_tiff) as src:
        b = src.bounds
        crs = src.crs

    left = np.floor(b.left / res) * res
    top = np.ceil(b.top / res) * res
    right = np.ceil(b.right / res) * res
    bottom = np.floor(b.bottom / res) * res

    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    transform = rasterio.transform.from_origin(left, top, res, res)

    logger.info("Analysis grid: %d x %d @ %.0f m, bounds=(%.0f, %.0f, %.0f, %.0f)",
                width, height, res, left, bottom, right, top)
    return GridSpec(transform, width, height, crs, res)


def _bbox_lonlat(grid: GridSpec) -> tuple[float, float, float, float]:
    from rasterio.warp import transform_bounds
    left, bottom, right, top = grid.bounds
    return transform_bounds(grid.crs, "EPSG:4326",
                            left, bottom, right, top)


def _reproject_to_grid(src_array: np.ndarray,
                       src_transform,
                       src_crs,
                       grid: GridSpec,
                       resampling: Resampling = Resampling.bilinear,
                       src_nodata=np.nan) -> np.ndarray:
    dst = np.full(grid.shape, np.nan, dtype=np.float32)
    reproject(
        source=src_array.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    return dst


# ===========================================================================
# Section 1: Sentinel-2 predictors
# ===========================================================================

def fetch_sentinel2_items(grid: GridSpec, scene: SceneConfig):
    import planetary_computer
    import pystac_client
    from datetime import datetime, timedelta

    t = datetime.strptime(scene.datetime_utc, "%Y-%m-%dT%H:%M:%SZ")
    start = (t - timedelta(days=scene.s2_window_days)).strftime("%Y-%m-%d")
    end = (t + timedelta(days=scene.s2_window_days)).strftime("%Y-%m-%d")
    bbox = _bbox_lonlat(grid)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": scene.s2_max_cloud}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 L2A under {scene.s2_max_cloud}% cloud within "
            f"+/-{scene.s2_window_days} d of {scene.datetime_utc}. "
            "Widen s2_window_days or raise s2_max_cloud."
        )
    # Deduplicate: same MGRS tile reprocessed -> keep newest item id
    seen, dedup = set(), []
    for it in sorted(items, key=lambda x: x.id, reverse=True):
        tile = it.properties.get("s2:mgrs_tile", it.id)
        if tile not in seen:
            seen.add(tile)
            dedup.append(it)
    dedup.sort(key=lambda it: it.properties["eo:cloud_cover"])
    logger.info("[%s] Sentinel-2: %d tile(s) for mosaic -> %s",
                scene.key, len(dedup),
                ", ".join(f"{it.properties.get('s2:mgrs_tile')}"
                          f"({it.properties['eo:cloud_cover']:.1f}%)"
                          for it in dedup))
    return dedup


def _read_s2_band(item, asset_key: str, grid: GridSpec) -> np.ndarray:
    import planetary_computer
    import rioxarray  # noqa: F401

    signed = planetary_computer.sign(item.assets[asset_key].href)
    with rasterio.open(signed) as src:
        arr = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs

    # BOA offset: prefer item metadata, else default to -1000
    offset = -1000.0
    boa = item.properties.get("s2:processing_baseline")
    if boa is not None and float(str(boa).replace("N", "")) < 4.0:
        offset = 0.0
    reflectance = (arr + offset) / 10000.0
    reflectance = np.clip(reflectance, 0.0, 1.0)

    return _reproject_to_grid(reflectance, src_transform, src_crs, grid,
                              resampling=Resampling.bilinear)


def compute_s2_predictors(items, grid: GridSpec) -> dict[str, np.ndarray]:
    import planetary_computer

    # Accept a single item too, for backward compatibility
    if not isinstance(items, (list, tuple)):
        items = [items]

    band_keys = ["B02", "B03", "B04", "B08", "B11", "B12"]
    mosaic = {b: np.full(grid.shape, np.nan, dtype=np.float32)
              for b in band_keys}

    for it in items:
        it = planetary_computer.sign(it)
        if not np.isnan(mosaic["B02"]).any():
            break  # grid already fully covered
        for b in band_keys:
            tile = _read_s2_band(it, b, grid)
            fill = np.isnan(mosaic[b]) & np.isfinite(tile)
            mosaic[b][fill] = tile[fill]

    b2, b3, b4 = mosaic["B02"], mosaic["B03"], mosaic["B04"]
    b8, b11, b12 = mosaic["B08"], mosaic["B11"], mosaic["B12"]
    eps = 1e-6

    cov = 100.0 * np.isfinite(b2).mean()
    logger.info("[s2] mosaic coverage: %.1f%%", cov)
    if cov < 95.0:
        logger.warning("[s2] mosaic still %.1f%% short -- widen "
                        "s2_window_days or raise s2_max_cloud.", 100 - cov)

    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndbi = (b11 - b8) / (b11 + b8 + eps)
    mndwi = (b3 - b11) / (b3 + b11 + eps)
    bsi = ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)

    # Bonafoni (2020) Sentinel-2 broadband albedo
    albedo = (0.356 * b2 + 0.130 * b4 + 0.373 * b8
              + 0.085 * b11 + 0.072 * b12 - 0.0018) / 1.016
    albedo = np.clip(albedo, 0.0, 1.0)

    return {"ndvi": ndvi, "ndbi": ndbi, "mndwi": mndwi,
            "bsi": bsi, "albedo": albedo}


# ===========================================================================
# Section 2: buildings -> synthetic DSM
# ===========================================================================

def fetch_ms_buildings(grid: GridSpec):
    import geopandas as gpd
    import planetary_computer
    import pystac_client

    bbox = _bbox_lonlat(grid)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(collections=["ms-buildings"], bbox=bbox)
    items = list(search.items())
    if not items:
        logger.warning("[buildings] No MS Buildings tiles for AOI. "
                        "SVF and building metrics will be empty.")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    frames = []
    for it in items:
        # ms-buildings assets are GeoParquet partitioned by quadkey
        asset = it.assets.get("data")
        if asset is None:
            continue
        gdf = gpd.read_parquet(asset.href)
        frames.append(gdf)

    buildings = gpd.GeoDataFrame(
        __import__("pandas").concat(frames, ignore_index=True),
        crs="EPSG:4326",
    )
    # Clip to AOI then reproject to the analysis CRS
    from shapely.geometry import box
    aoi = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")
    buildings = gpd.overlay(buildings, aoi, how="intersection")
    buildings = buildings.to_crs(grid.crs)
    logger.info("[buildings] %d footprints in AOI", len(buildings))
    return buildings


def assign_heights_from_wsf3d(buildings, wsf3d_path: Path):
    import geopandas as gpd

    if len(buildings) == 0:
        buildings["height_m"] = []
        return buildings
    if wsf3d_path is None or not Path(wsf3d_path).exists():
        logger.warning("[buildings] WSF-3D not found at %s -- assigning a "
                        "flat fallback height of 6 m to all footprints.",
                        wsf3d_path)
        buildings = buildings.copy()
        buildings["height_m"] = 6.0
        return buildings

    with rasterio.open(wsf3d_path) as src:
        wsf_crs = src.crs
        centroids = buildings.geometry.centroid.to_crs(wsf_crs)
        coords = [(pt.x, pt.y) for pt in centroids]
        sampled = np.array([v[0] for v in src.sample(coords)],
                           dtype=np.float32)

    buildings = buildings.copy()
    # WSF-3D height of 0 over a building polygon usually means the 90 m
    # cell average is below detection; floor at a small positive value.
    buildings["height_m"] = np.where(sampled > 1.0, sampled, 3.0)
    logger.info("[buildings] heights assigned: mean %.1f m, max %.1f m",
                float(np.nanmean(buildings["height_m"])),
                float(np.nanmax(buildings["height_m"])))
    return buildings


def build_synthetic_dsm(buildings, grid: GridSpec) -> np.ndarray:
    if len(buildings) == 0:
        return np.zeros(grid.shape, dtype=np.float32)

    # Paint shortest first so taller buildings overwrite -> effective MAX
    b = buildings.sort_values("height_m")
    shapes = ((geom, float(h)) for geom, h
              in zip(b.geometry, b["height_m"]))
    dsm = rasterize(
        shapes=shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0.0,
        dtype="float32",
        all_touched=False,
    )
    return dsm


def rasterize_building_metrics(buildings,
                               grid: GridSpec,
                               focal_radius_m: float = 50.0
                               ) -> dict[str, np.ndarray]:
    if len(buildings) == 0:
        zeros = np.zeros(grid.shape, dtype=np.float32)
        return {"fp_density": zeros, "mean_height": zeros.copy()}

    # Binary footprint mask and a height raster
    mask = rasterize(
        shapes=((g, 1) for g in buildings.geometry),
        out_shape=grid.shape, transform=grid.transform,
        fill=0, dtype="uint8", all_touched=False,
    ).astype(np.float32)
    height = build_synthetic_dsm(buildings, grid)

    # Focal mean via uniform filter
    win = max(1, int(round(focal_radius_m / grid.res)))
    fp_density = ndimage.uniform_filter(mask, size=2 * win + 1,
                                        mode="nearest")
    # Focal mean height over built pixels only
    height_sum = ndimage.uniform_filter(height, size=2 * win + 1,
                                        mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_height = np.where(fp_density > 1e-3,
                               height_sum / np.maximum(fp_density, 1e-3),
                               0.0)
    return {"fp_density": fp_density.astype(np.float32),
            "mean_height": mean_height.astype(np.float32)}


# ===========================================================================
# Section 3: sky-view factor
# ===========================================================================

def compute_svf(dsm: np.ndarray,
                grid: GridSpec,
                n_directions: int = 16,
                max_radius_m: float = 200.0) -> np.ndarray:
    res = grid.res
    n_steps = max(1, int(round(max_radius_m / res)))
    horizon_sin = np.zeros(dsm.shape, dtype=np.float32)

    for d in range(n_directions):
        az = 2.0 * np.pi * d / n_directions
        dx = np.sin(az)
        dy = -np.cos(az)  # raster row index increases downward
        max_angle = np.zeros(dsm.shape, dtype=np.float32)

        for step in range(1, n_steps + 1):
            shift_x = int(round(dx * step))
            shift_y = int(round(dy * step))
            if shift_x == 0 and shift_y == 0:
                continue
            shifted = _shift_array(dsm, shift_y, shift_x)
            horiz_dist = step * res
            angle = np.arctan2(shifted - dsm, horiz_dist)
            np.maximum(max_angle, angle, out=max_angle)

        horizon_sin += np.sin(np.clip(max_angle, 0.0, np.pi / 2))

    svf = 1.0 - horizon_sin / n_directions
    return np.clip(svf, 0.0, 1.0).astype(np.float32)


def _shift_array(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(arr)
    h, w = arr.shape
    y_src = slice(max(0, -dy), h - max(0, dy))
    x_src = slice(max(0, -dx), w - max(0, dx))
    y_dst = slice(max(0, dy), h - max(0, -dy))
    x_dst = slice(max(0, dx), w - max(0, -dx))
    out[y_dst, x_dst] = arr[y_src, x_src]
    return out


# ===========================================================================
# Section 4: ESA WorldCover
# ===========================================================================

# ESA WorldCover class codes
WC_TREE, WC_SHRUB, WC_GRASS, WC_CROP = 10, 20, 30, 40
WC_BUILT, WC_BARE, WC_SNOW, WC_WATER = 50, 60, 70, 80
WC_WETLAND, WC_MANGROVE, WC_MOSS = 90, 95, 100


def fetch_worldcover(grid: GridSpec) -> dict[str, np.ndarray]:
    import planetary_computer
    import pystac_client

    bbox = _bbox_lonlat(grid)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(collections=["esa-worldcover"], bbox=bbox,
                            query={"esa_worldcover:product_version":
                                   {"eq": "2.0.0"}})
    items = list(search.items())
    if not items:
        raise RuntimeError("No ESA WorldCover tile for AOI.")

    # Mosaic tiles by reprojecting each onto the grid (modal class)
    wc = np.full(grid.shape, np.nan, dtype=np.float32)
    for it in items:
        href = it.assets["map"].href
        with rasterio.open(href) as src:
            arr = src.read(1).astype(np.float32)
            tile = _reproject_to_grid(arr, src.transform, src.crs, grid,
                                      resampling=Resampling.nearest)
        wc = np.where(np.isnan(wc), tile, wc)

    built = (wc == WC_BUILT).astype(np.float32)
    win = max(1, int(round(30.0 / grid.res)))
    impervious_frac = ndimage.uniform_filter(built, size=2 * win + 1,
                                             mode="nearest")
    return {"wc_class": wc.astype(np.float32),
            "impervious_frac": impervious_frac.astype(np.float32)}


# ===========================================================================
# Section 5: distance fields
# ===========================================================================

def compute_distance_to_water(wc_class: np.ndarray,
                              grid: GridSpec) -> np.ndarray:
    water = (wc_class == WC_WATER)
    if not water.any():
        logger.warning("[distance] No water pixels in AOI -- distance "
                        "field set to a large constant.")
        return np.full(grid.shape, 1e5, dtype=np.float32)
    # distance_transform_edt returns distance to nearest False cell;
    # invert so we measure distance to nearest water cell.
    dist_px = ndimage.distance_transform_edt(~water)
    return (dist_px * grid.res).astype(np.float32)


# ===========================================================================
# Section 6: elevation
# ===========================================================================

def fetch_dem(grid: GridSpec) -> np.ndarray:
    import planetary_computer
    import pystac_client

    bbox = _bbox_lonlat(grid)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
    items = list(search.items())
    if not items:
        logger.warning("[dem] No Copernicus DEM tile -- elevation set 0.")
        return np.zeros(grid.shape, dtype=np.float32)

    dem = np.full(grid.shape, np.nan, dtype=np.float32)
    for it in items:
        href = it.assets["data"].href
        with rasterio.open(href) as src:
            arr = src.read(1).astype(np.float32)
            tile = _reproject_to_grid(arr, src.transform, src.crs, grid,
                                      resampling=Resampling.bilinear)
        dem = np.where(np.isnan(dem), tile, dem)
    return np.nan_to_num(dem, nan=0.0).astype(np.float32)


# ===========================================================================
# Section 7: assembly
# ===========================================================================

def build_predictor_stack(scene: SceneConfig,
                           wsf3d_path: Optional[Path] = None,
                           res: float = 10.0):
    import xarray as xr

    logger.info("=== Building predictor stack: %s (%s, %s) ===",
                scene.key, scene.typology, scene.role)
    grid = analysis_grid_from_hotsat(scene.hotsat_tiff, res=res)

    # Sentinel-2
    s2_item = fetch_sentinel2_items(grid, scene)
    s2 = compute_s2_predictors(s2_item, grid)

    # Buildings -> WSF-3D gridded predictors (uniform across all scenes).
    # Replaces the Microsoft-footprints + synthetic-DSM path: WSF-3D is
    # radar-derived and sees low-contrast arid building stock that the
    # optical detectors (MS Buildings, OSM, Google Open Buildings) miss.
    from wsf3d_buildings import build_wsf3d_predictors, WSF3D_PATHS
    wsf = build_wsf3d_predictors(grid, WSF3D_PATHS)
    bmetrics = {"fp_density": wsf["fp_density"],
                "mean_height": wsf["mean_height"]}
    svf = wsf["svf"]

    # WorldCover
    wc = fetch_worldcover(grid)

    # Distance + elevation
    coast_dist = compute_distance_to_water(wc["wc_class"], grid)
    elevation = fetch_dem(grid)

    # Coordinate axes
    left, top = grid.transform.c, grid.transform.f
    xs = left + (np.arange(grid.width) + 0.5) * grid.res
    ys = top - (np.arange(grid.height) + 0.5) * grid.res

    layers = {
        "ndvi": s2["ndvi"], "ndbi": s2["ndbi"], "mndwi": s2["mndwi"],
        "bsi": s2["bsi"], "albedo": s2["albedo"],
        "fp_density": bmetrics["fp_density"],
        "mean_height": bmetrics["mean_height"],
        "svf": svf,
        "built_area": wsf.get("built_area", np.zeros(grid.shape, np.float32)),
        "built_volume": wsf.get("built_volume",
                                np.zeros(grid.shape, np.float32)),
        "impervious_frac": wc["impervious_frac"],
        "wc_class": wc["wc_class"],
        "coast_dist": coast_dist,
        "elevation": elevation,
    }
    ds = xr.Dataset(
        {name: (("y", "x"), arr) for name, arr in layers.items()},
        coords={"x": xs, "y": ys},
        attrs={
            "scene_id": scene.scene_id,
            "typology": scene.typology,
            "role": scene.role,
            "crs": str(grid.crs),
            "res_m": grid.res,
            "transform": list(grid.transform)[:6],
        },
    )
    logger.info("[%s] predictor stack complete: %d layers, %d x %d",
                scene.key, len(layers), grid.height, grid.width)
    return ds


def aggregate_to_grid(ds,
                      target_transform,
                      target_shape: tuple[int, int],
                      target_crs,
                      categorical: tuple[str, ...] = ("wc_class",)):
    import xarray as xr

    src_transform = rasterio.transform.Affine(*ds.attrs["transform"])
    src_crs = ds.attrs["crs"]
    out = {}
    for name in ds.data_vars:
        arr = ds[name].values
        resampling = (Resampling.mode if name in categorical
                      else Resampling.average)
        dst = np.full(target_shape, np.nan, dtype=np.float32)
        reproject(
            source=arr.astype(np.float32),
            destination=dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        out[name] = (("y", "x"), dst)

    return xr.Dataset(out, attrs={**ds.attrs,
                                  "aggregated_to_shape": list(target_shape)})


# ===========================================================================
# Section 8: predictor-stack QA
# ===========================================================================

# Plausible value ranges per layer. None = unbounded / not range-checked.
# Used only to flag layers that fall outside what the predictor should be.
_QA_EXPECTED = {
    "ndvi":            (-1.0, 1.0),
    "ndbi":            (-1.0, 1.0),
    "mndwi":           (-1.0, 1.0),
    "bsi":             (-1.0, 1.0),
    "albedo":          (0.0, 1.0),
    "fp_density":      (0.0, 1.0),
    "mean_height":     (0.0, 400.0),
    "svf":             (0.0, 1.0),
    "built_area":      (0.0, None),
    "built_volume":    (0.0, None),
    "impervious_frac": (0.0, 1.0),
    "wc_class":        (10.0, 100.0),
    "coast_dist":      (0.0, None),
    "elevation":       (-50.0, 2000.0),
}


def qa_predictor_stack(ds, scene_key: str = "") -> dict[str, dict]:
    report: dict[str, dict] = {}
    tag = f"[{scene_key}] " if scene_key else ""
    logger.info("%sQA: predictor-stack layer summary", tag)
    logger.info("%s  %-16s %10s %10s %10s %8s  flags",
                tag, "layer", "min", "mean", "max", "nan%")

    for name in ds.data_vars:
        arr = np.asarray(ds[name].values, dtype=np.float64)
        n_total = arr.size
        nan_frac = float(np.isnan(arr).mean()) if n_total else 1.0
        finite = arr[np.isfinite(arr)]

        if finite.size == 0:
            stats = dict(min=np.nan, mean=np.nan, max=np.nan,
                         std=np.nan, nan_frac=nan_frac)
            flags = ["ALL-NAN"]
        else:
            vmin = float(finite.min())
            vmax = float(finite.max())
            vmean = float(finite.mean())
            vstd = float(finite.std())
            stats = dict(min=vmin, mean=vmean, max=vmax,
                         std=vstd, nan_frac=nan_frac)
            flags = []

            # constant layer
            span = abs(vmax - vmin)
            if vstd < 1e-9 or span < 1e-9:
                flags.append("DEGENERATE")

            # NaN coverage
            if nan_frac > 0.20:
                flags.append(f"HIGH-NAN({nan_frac*100:.0f}%)")

            # range check
            exp = _QA_EXPECTED.get(name)
            if exp is not None:
                lo, hi = exp
                if (lo is not None and vmin < lo - 1e-6) or \
                   (hi is not None and vmax > hi + 1e-6):
                    flags.append("OUT-OF-RANGE")

            # SVF flat-DSM artefact: if every footprint was 6 m, SVF is
            # near-binary and its histogram collapses to a couple of
            # values. Detect via very low unique-value count among
            # non-unity pixels.
            if name == "svf":
                non_open = finite[finite < 0.999]
                if non_open.size > 0:
                    n_unique = np.unique(np.round(non_open, 3)).size
                    if n_unique <= 3:
                        flags.append("FLAT-DSM(check WSF-3D)")

        report[name] = {**stats, "flags": flags}
        flag_str = ", ".join(flags) if flags else "ok"
        logger.info("%s  %-16s %10.3f %10.3f %10.3f %7.1f%%  %s",
                    tag, name,
                    stats["min"], stats["mean"], stats["max"],
                    stats["nan_frac"] * 100, flag_str)

    # Scene-level verdict
    blocking = {n: r["flags"] for n, r in report.items()
                if any(f.startswith(("ALL-NAN", "DEGENERATE", "FLAT-DSM"))
                       for f in r["flags"])}
    if blocking:
        logger.warning("%sQA: %d layer(s) need attention before training: %s",
                        tag, len(blocking),
                        ", ".join(f"{n} ({'/'.join(f)})"
                                  for n, f in blocking.items()))
    else:
        logger.info("%sQA: all layers passed.", tag)

    return report


# ===========================================================================
# Driver
# ===========================================================================

def _example():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    wsf3d = (DATA_ROOT / "ancillary/WSF3D_V02_BuildingHeight.tif")
    out_dir = (DATA_ROOT / "predictors")
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, scene in SCENES.items():
        ds = build_predictor_stack(scene, wsf3d_path=wsf3d, res=10.0)
        qa_predictor_stack(ds, scene_key=key)
        nc_path = out_dir / f"{scene.scene_id}_predictors_10m.nc"
        ds.to_netcdf(nc_path)
        logger.info("[%s] saved -> %s", key, nc_path)

        # Coarse aggregation example: pass the real VNP21 / MYD11 grid
        # read from the downloaded coarse product instead of the stub below.
        #
        # with rasterio.open(vnp21_lst_path) as src:
        #     ds_750 = aggregate_to_grid(ds, src.transform,
        #                                (src.height, src.width), src.crs)
        #     ds_750.to_netcdf(out_dir / f"{scene.scene_id}_predictors_750m.nc")


if __name__ == "__main__":
    _example()
