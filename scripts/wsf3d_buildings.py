# Helper for Step 2: windowed reading of the DLR WSF-3D rasters (building
# fraction, height, area, volume) onto the 10 m analysis grid.

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from config import DATA_ROOT
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class WSF3DPaths:
    height: Path
    fraction: Path
    area: Optional[Path] = None
    volume: Optional[Path] = None


# WSF-3D rasters (download from the DLR EOC Geoservice) under DATA_ROOT/ancillary.
WSF3D_PATHS = WSF3DPaths(
    height=(DATA_ROOT / "ancillary/WSF3D_V02_BuildingHeight.tif"),
    fraction=(DATA_ROOT / "ancillary/WSF3D_V02_BuildingFraction.tif"),
    area=(DATA_ROOT / "ancillary/WSF3D_V02_BuildingArea.tif"),
    volume=(DATA_ROOT / "ancillary/WSF3D_V02_BuildingVolume.tif"),
)


# ===========================================================================
# Windowed reader
# ===========================================================================

def _read_windowed_to_grid(raster_path: Path,
                            grid,
                            resampling: Resampling = Resampling.bilinear,
                            pad_m: float = 500.0) -> np.ndarray:
    raster_path = Path(raster_path)
    if not raster_path.exists():
        raise FileNotFoundError(
            f"WSF-3D raster not found: {raster_path}. Download from "
            "https://download.geoservice.dlr.de/WSF3D/files/global/"
        )

    left, bottom, right, top = grid.bounds  # in grid.crs (UTM)

    with rasterio.open(raster_path) as src:
        # WSF-3D global rasters are EPSG:4326. Reproject the grid bbox
        # into the raster CRS to define the read window.
        win_l, win_b, win_r, win_t = transform_bounds(
            grid.crs, src.crs,
            left - pad_m, bottom - pad_m, right + pad_m, top + pad_m,
        )
        window = from_bounds(win_l, win_b, win_r, win_t,
                             transform=src.transform)
        # Round outward so the window fully contains the bbox
        window = window.round_offsets(op="floor").round_lengths(op="ceil")

        # Clamp the window to the raster extent
        window = window.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height))
        if window.width <= 0 or window.height <= 0:
            raise ValueError(
                f"Scene bbox falls outside {raster_path.name}. "
                "Check the AOI and the raster cover the same region."
            )

        data = src.read(1, window=window).astype(np.float32)
        win_transform = src.window_transform(window)
        src_crs = src.crs
        src_nodata = src.nodata

    # Treat the declared nodata as NaN for resampling, but NOT 0:
    # 0 in WSF-3D Height/Fraction means "no building", which is valid.
    if src_nodata is not None and not np.isnan(src_nodata):
        data = np.where(data == src_nodata, np.nan, data)

    # Resample the windowed array onto the analysis grid
    from rasterio.warp import reproject
    dst = np.full(grid.shape, np.nan, dtype=np.float32)
    reproject(
        source=data,
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        src_nodata=np.nan,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=np.nan,
        resampling=resampling,
    )
    # Any residual NaN over land = genuine non-built -> 0
    dst = np.nan_to_num(dst, nan=0.0)
    return dst


# ===========================================================================
# Approximate sky-view factor from WSF-3D fraction + height
# ===========================================================================

def svf_from_fraction_height(fp_density: np.ndarray,
                             mean_height: np.ndarray,
                             cell_size_m: float = 90.0) -> np.ndarray:
    eps = 1e-6
    fp = np.clip(fp_density, 0.0, 1.0)

    gap_width = cell_size_m * (1.0 - np.sqrt(fp))
    gap_width = np.maximum(gap_width, eps)
    aspect = mean_height / gap_width                 # H / W
    svf_canyon = np.cos(np.arctan(aspect))           # 0..1

    svf = (1.0 - fp) * 1.0 + fp * svf_canyon
    return np.clip(svf, 0.0, 1.0).astype(np.float32)


# ===========================================================================
# Main entry point
# ===========================================================================

def build_wsf3d_predictors(grid,
                           paths: WSF3DPaths = WSF3D_PATHS,
                           wsf3d_cell_m: float = 90.0) -> dict[str, np.ndarray]:
    logger.info("[wsf3d] reading windowed building rasters")

    # Fraction -> fp_density. Bilinear is fine for a continuous 0-1 field.
    fp_density = _read_windowed_to_grid(
        paths.fraction, grid, resampling=Resampling.bilinear)
    fp_density = np.clip(fp_density, 0.0, 1.0)

    # Height -> mean_height. Bilinear; metres.
    mean_height = _read_windowed_to_grid(
        paths.height, grid, resampling=Resampling.bilinear)
    mean_height = np.maximum(mean_height, 0.0)

    out = {"fp_density": fp_density, "mean_height": mean_height}

    # Optional Area / Volume
    if paths.area is not None and Path(paths.area).exists():
        out["built_area"] = _read_windowed_to_grid(
            paths.area, grid, resampling=Resampling.bilinear)
        out["built_area"] = np.maximum(out["built_area"], 0.0)
    if paths.volume is not None and Path(paths.volume).exists():
        out["built_volume"] = _read_windowed_to_grid(
            paths.volume, grid, resampling=Resampling.bilinear)
        out["built_volume"] = np.maximum(out["built_volume"], 0.0)

    # Approximate SVF from fraction + height
    out["svf"] = svf_from_fraction_height(
        fp_density, mean_height, cell_size_m=wsf3d_cell_m)

    # Brief summary
    for name, arr in out.items():
        finite = arr[np.isfinite(arr)]
        if finite.size:
            built = finite[finite > 0] if name != "svf" else finite[finite < 0.999]
            logger.info("[wsf3d]   %-13s min=%.3f mean=%.3f max=%.3f "
                        "(built-pixel mean=%.3f)",
                        name, float(finite.min()), float(finite.mean()),
                        float(finite.max()),
                        float(built.mean()) if built.size else 0.0)
    return out


# ===========================================================================
# Integration note
# ===========================================================================
#
# In predictor_stack.build_predictor_stack, replace the buildings block:
#
#     # OLD -- Microsoft footprints + per-building DSM
#     buildings = fetch_ms_buildings(grid)
#     buildings = assign_heights_from_wsf3d(buildings, wsf3d_path)
#     dsm = build_synthetic_dsm(buildings, grid)
#     svf = compute_svf(dsm, grid)
#     bmetrics = rasterize_building_metrics(buildings, grid)
#
#     # NEW -- WSF-3D gridded predictors, uniform across all scenes
#     from wsf3d_buildings import build_wsf3d_predictors, WSF3D_PATHS
#     wsf = build_wsf3d_predictors(grid, WSF3D_PATHS)
#     bmetrics = {"fp_density": wsf["fp_density"],
#                 "mean_height": wsf["mean_height"]}
#     svf = wsf["svf"]
#
# Then in the `layers` dict, optionally add the extra WSF-3D layers:
#
#     "built_volume": wsf.get("built_volume", np.zeros(grid.shape, np.float32)),
#
# fetch_ms_buildings, assign_heights_from_wsf3d, build_synthetic_dsm,
# rasterize_building_metrics and compute_svf are no longer called and can
# be left in place (unused) or deleted.


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    # Standalone smoke test against one scene's grid.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from predictor_stack import analysis_grid_from_hotsat, SCENES

    for key, scene in SCENES.items():
        logger.info("=== WSF-3D predictors: %s ===", key)
        grid = analysis_grid_from_hotsat(scene.hotsat_tiff, res=10.0)
        wsf = build_wsf3d_predictors(grid, WSF3D_PATHS)
