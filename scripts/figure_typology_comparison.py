# Step 13. Figure 2: overview of SBT and predictor layers for the three
# scenes.

from __future__ import annotations

import logging
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

from config import DATA_ROOT  # set via UAE_SBT_DATA_ROOT (config.py)
GBA_DIR = DATA_ROOT / "buildings_gba"
OUTPUT_DIR = DATA_ROOT / "analysis" / "figures"

# QA bits (match hotsat_lst.py)
QA_NODATA      = 1 << 0
QA_EPS_LOW     = 1 << 1
QA_OBLIQUE     = 1 << 2
QA_CLOUD       = 1 << 3
QA_HOT_ANOMALY = 1 << 4
QA_EXCLUDE = QA_NODATA | QA_EPS_LOW | QA_CLOUD | QA_HOT_ANOMALY


@dataclass
class Scene:
    key: str
    scene_id: str
    typology: str          # column label
    predictor_nc: Path
    lst_tif: Path
    qa_tif: Path
    gba_geojson: Path


def _scene(key, scene_id, typology) -> Scene:
    return Scene(
        key=key, scene_id=scene_id, typology=typology,
        predictor_nc=DATA_ROOT / "predictors" /
        f"{scene_id}_predictors_10m.nc",
        lst_tif=DATA_ROOT / "lst" / f"{scene_id}_LST_C.tif",
        qa_tif=DATA_ROOT / "lst" / f"{scene_id}_LST_QA.tif",
        gba_geojson=GBA_DIR / f"GBA_{key}_3857.geojson",
    )


# column order: sub-urban, industrial, urban
SCENES = [
    _scene("AD_suburban",   "20231029_213936_HS1_SBT", "Sub-urban"),
    _scene("AD_industrial", "20231030_212656_HS1_SBT", "Industrial"),
    _scene("DXB_urban",     "20231120_213935_HS1_SBT", "Urban"),
]

# ---------------------------------------------------------------------------
# Rows of the figure. Each row is one displayed layer.
#   kind   : "lst"        -> HotSat LST GeoTIFF, QA-masked
#            "footprint"  -> GBA building footprints, rasterized (binary)
#            "predictor"  -> a named layer from the predictor stack
#   name   : predictor-stack variable name (only for kind="predictor")
#   label  : row label shown on the left
#   cmap   : matplotlib colormap
#   units  : colourbar label
#   pct    : (low, high) percentile clip for the shared colour scale
# ---------------------------------------------------------------------------
ROWS = [
    dict(kind="lst",       name=None,             label="Night-time SBT",
         cmap="inferno",  units="SBT (\u00b0C)",          pct=(1, 99)),
    dict(kind="footprint", name=None,             label="Building footprints",
         cmap="binary",   units="building",                pct=(0, 100),
         vrange=(0.0, 1.0)),
    dict(kind="predictor", name="albedo",         label="Albedo",
         cmap="viridis",  units="albedo",                  pct=(2, 98)),
    dict(kind="predictor", name="impervious_frac",label="Impervious fraction",
         cmap="cividis",  units="fraction",                pct=(2, 98)),
    dict(kind="predictor", name="svf",            label="Sky-view factor",
         cmap="bone",     units="SVF",                     pct=(2, 98)),
    dict(kind="predictor", name="mean_height",    label="Mean building height",
         cmap="magma",    units="height (m)",              pct=(2, 98)),
]


# ===========================================================================
# Raster / vector loading
# ===========================================================================

def load_grid(predictor_nc: Path):
    import xarray as xr
    ds = xr.open_dataset(predictor_nc)
    transform = rasterio.transform.Affine(*ds.attrs["transform"])
    crs = ds.attrs["crs"]
    h, w = ds[list(ds.data_vars)[0]].shape
    ds.close()
    return transform, crs, h, w


def load_predictor_layer(predictor_nc: Path, name: str) -> np.ndarray:
    import xarray as xr
    ds = xr.open_dataset(predictor_nc)
    if name not in ds:
        ds.close()
        raise KeyError(f"'{name}' not in {predictor_nc.name}; "
                       f"available: {list(ds.data_vars)}")
    arr = ds[name].values.astype(np.float32)
    ds.close()
    return arr


def load_raster_on_grid(path, transform, crs, shape,
                        resampling=Resampling.bilinear) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        st, sc, nod = src.transform, src.crs, src.nodata
    if nod is not None and not np.isnan(nod):
        arr = np.where(arr == nod, np.nan, arr)
    dst = np.full(shape, np.nan, dtype=np.float32)
    reproject(source=arr, destination=dst,
              src_transform=st, src_crs=sc, src_nodata=np.nan,
              dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
              resampling=resampling)
    return dst


def load_lst_masked(scene: Scene, transform, crs, shape) -> np.ndarray:
    lst = load_raster_on_grid(scene.lst_tif, transform, crs, shape,
                              Resampling.bilinear)
    qa = load_raster_on_grid(scene.qa_tif, transform, crs, shape,
                             Resampling.nearest)
    qa = np.nan_to_num(qa, nan=QA_NODATA).astype(np.int32)
    return np.where((qa & QA_EXCLUDE) == 0, lst, np.nan)


def load_footprint_raster(scene: Scene, transform, crs, shape) -> np.ndarray:
    import geopandas as gpd
    from rasterio.features import rasterize

    gdf = gpd.read_file(scene.gba_geojson)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3857")
    gdf = gdf.to_crs(crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        logger.warning("[%s] no footprints to rasterize", scene.key)
        return np.zeros(shape, dtype=np.float32)
    r = rasterize(shapes, out_shape=shape, transform=transform,
                  fill=0, dtype="uint8")
    n_b = int((r == 1).sum())
    logger.info("[%s] footprints rasterized: %d building pixels "
                "(%.1f%% of grid)", scene.key, n_b,
                100.0 * n_b / r.size)
    return r.astype(np.float32)          # 1 = building, 0 = background


# ===========================================================================
# Build all layers, scene by scene and row by row
# ===========================================================================

def assemble_layers():
    grids = []
    for sc in SCENES:
        grids.append(load_grid(sc.predictor_nc))

    panels = []
    for row in ROWS:
        row_arrays = []
        for sc, (transform, crs, h, w) in zip(SCENES, grids):
            shape = (h, w)
            if row["kind"] == "lst":
                arr = load_lst_masked(sc, transform, crs, shape)
            elif row["kind"] == "footprint":
                arr = load_footprint_raster(sc, transform, crs, shape)
            elif row["kind"] == "predictor":
                arr = load_predictor_layer(sc.predictor_nc, row["name"])
            else:
                raise ValueError(row["kind"])
            row_arrays.append(arr)
            logger.info("[%s] %s loaded (%d x %d)",
                        sc.key, row["label"], h, w)
        panels.append(row_arrays)
    return panels, grids


def shared_scale(row_arrays, pct):
    pooled = np.concatenate([a[np.isfinite(a)].ravel()
                             for a in row_arrays])
    if pooled.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(pooled, [pct[0], pct[1]])
    if lo == hi:
        hi = lo + 1e-6
    return float(lo), float(hi)


# ===========================================================================
# Figure
# ===========================================================================

def build_figure(panels, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(ROWS)
    n_cols = len(SCENES)

    # size each panel ~3.3 in wide; height tracks each scene's aspect
    fig_w = 3.5 * n_cols + 1.4          # +space for row labels / colourbars
    fig_h = 3.0 * n_rows + 0.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    panel_letters = iter(string.ascii_lowercase)

    for ri, row in enumerate(ROWS):
        row_arrays = panels[ri]
        if row.get("vrange") is not None:
            vmin, vmax = row["vrange"]      # explicit fixed scale
        else:
            vmin, vmax = shared_scale(row_arrays, row["pct"])

        last_im = None
        for ci, arr in enumerate(row_arrays):
            ax = axes[ri][ci]
            disp = np.ma.masked_invalid(arr)
            im = ax.imshow(disp, cmap=row["cmap"], vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            last_im = im
            ax.set_xticks([])
            ax.set_yticks([])

            # panel label (a), (b), ... -- left-aligned, not bold,
            # no title
            letter = next(panel_letters)
            ax.text(0.02, 1.02, f"({letter})", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=11)

            # column (typology) label once, above the top row
            if ri == 0:
                ax.annotate(SCENES[ci].typology,
                            xy=(0.5, 1.10), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=12)

        # row (variable) label once, left of the first column
        axes[ri][0].annotate(
            row["label"], xy=(-0.12, 0.5),
            xycoords="axes fraction", ha="right", va="center",
            fontsize=11, rotation=90)

        # one shared colourbar per row, to the right of the last panel
        cbar = fig.colorbar(last_im, ax=axes[ri].tolist(),
                            fraction=0.025, pad=0.02)
        cbar.set_label(row["units"], fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        if row["kind"] == "footprint":
            cbar.set_ticks([0.0, 1.0])
            cbar.set_ticklabels(["background", "building"])

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "figure_typology_comparison.png"
    pdf = output_dir / "figure_typology_comparison.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", png)
    logger.info("wrote %s", pdf)


# ===========================================================================
# Driver
# ===========================================================================

def run():
    panels, _ = assemble_layers()
    build_figure(panels, OUTPUT_DIR)
    logger.info("=== comparison figure complete ===")


if __name__ == "__main__":
    run()
