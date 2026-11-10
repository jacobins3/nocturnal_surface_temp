# Optional helper: points PROJ at the data directory bundled with
# rasterio/pyproj (fixes proj.db version clashes in mixed conda/pip
# environments). Import before rasterio.
import importlib.util
import os
import sys
from pathlib import Path


def _apply():
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
                print(f"[proj_fix] PROJ_DATA -> {d}", file=sys.stdout)
                return str(d)
    for v in ("PROJ_DATA", "PROJ_LIB"):
        os.environ.pop(v, None)
    print("[proj_fix] no bundled proj_data found; cleared PROJ_DATA/PROJ_LIB",
          file=sys.stdout)
    return None


if "rasterio" in sys.modules:
    print("[proj_fix] WARNING: rasterio was imported before proj_fix -- "
          "restart the kernel/interpreter and import proj_fix first.")
PROJ_DIR = _apply()
