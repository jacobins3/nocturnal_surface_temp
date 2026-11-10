# Step 7. Replace the exploratory LST files with the HotSat-1 SBT (deg C)
# under the same file names, so that all analyses run on SBT; --restore
# reverses this.

from __future__ import annotations

import proj_fix  # noqa: F401  (must precede rasterio)
import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

import hotsat_lst as H
import analysis_task3_building_contrast as t3

logger = logging.getLogger("use_sbt")

DATA_ROOT = t3.DATA_ROOT
LST_DIR = DATA_ROOT / "lst"
BACKUP_DIR = DATA_ROOT / "lst_LWIR_retrieval"
EXCLUDE = t3.QA_EXCLUDE


def _scene_ids():
    for key, s in t3.SCENES.items():
        yield key, Path(s.lst_tif).name.replace("_LST_C.tif", "")


def _read(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def make_sbt():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for key, sid in _scene_ids():
        lst_path = LST_DIR / f"{sid}_LST_C.tif"
        bak = BACKUP_DIR / f"{sid}_LST_C.tif"
        if not bak.exists():
            shutil.copy2(lst_path, bak)
            logger.info("%s: LST backed up -> %s", key, bak)
        else:
            logger.info("%s: backup already present (kept)", key)

        tb, meta = H.load_tb(DATA_ROOT / "hotsat" / f"{sid}.tiff")
        sbt_c = tb - 273.15
        H.write_cog(sbt_c, meta, lst_path)
        logger.info("%s: SBT written -> %s", key, lst_path)

        # comparison with the old LST on QA-valid pixels
        qa = _read(LST_DIR / f"{sid}_LST_QA.tif").astype(np.int32)
        lst = _read(bak)
        ok = ((qa & EXCLUDE) == 0) & np.isfinite(sbt_c) & np.isfinite(lst)
        s, l = sbt_c[ok], lst[ok]
        from scipy import stats
        idx = np.random.default_rng(0).choice(s.size, min(s.size, 200000),
                                              replace=False)
        rho = stats.spearmanr(s[idx], l[idx])[0]
        slope, icpt = np.polyfit(s[idx], l[idx], 1)
        logger.info("%s: SBT median %.2f C, SD %.2f C, IQR %.2f C | "
                    "old LST median %.2f C, SD %.2f C | Spearman %.4f, "
                    "LST = %.3f * SBT %+.2f",
                    key, np.median(s), np.std(s),
                    np.subtract(*np.percentile(s, [75, 25])),
                    np.median(l), np.std(l), rho, slope, icpt)


def restore():
    for key, sid in _scene_ids():
        bak = BACKUP_DIR / f"{sid}_LST_C.tif"
        if bak.exists():
            shutil.copy2(bak, LST_DIR / f"{sid}_LST_C.tif")
            logger.info("%s: LST restored", key)
        else:
            logger.warning("%s: no backup found, nothing restored", key)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true",
                    help="copy the backed-up LST back into lst/")
    a = ap.parse_args()
    restore() if a.restore else make_sbt()
