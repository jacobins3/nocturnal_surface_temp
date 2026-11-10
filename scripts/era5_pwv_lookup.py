# Step 3 (optional). Look up ERA5 total-column water vapour at each HotSat-1 acquisition
# (atmospheric context; Table A1). Requires a Copernicus CDS account
# (~/.cdsapirc).
from pathlib import Path
from config import DATA_ROOT

import numpy as np
import xarray as xr

# scene centre (lon, lat) and acquisition time (UTC), from the AOI bboxes
SCENES = {
    "AD_suburban":   dict(lon=55.032, lat=24.198, t="2023-10-29T21:39:36"),
    "AD_industrial": dict(lon=54.494, lat=24.369, t="2023-10-30T21:26:56"),
    "DXB_urban":     dict(lon=55.143, lat=24.962, t="2023-11-20T21:39:35"),
}
HALF_BOX = 0.25
OUT = DATA_ROOT / "era5_pwv_downloads"


def main():
    import cdsapi
    OUT.mkdir(exist_ok=True)
    c = cdsapi.Client()
    results = {}
    for key, s in SCENES.items():
        t = np.datetime64(s["t"])
        day = str(t.astype("datetime64[D]"))
        y, m, d = day.split("-")
        h0 = int(str(t.astype("datetime64[h]"))[-2:])
        target = OUT / f"{key}_tcwv.nc"
        if not target.exists():
            c.retrieve(
                "reanalysis-era5-single-levels",
                {"product_type": "reanalysis",
                 "variable": "total_column_water_vapour",
                 "year": y, "month": m, "day": d,
                 "time": [f"{h0:02d}:00", f"{h0 + 1:02d}:00"],
                 "area": [s["lat"] + HALF_BOX, s["lon"] - HALF_BOX,
                          s["lat"] - HALF_BOX, s["lon"] + HALF_BOX],
                 "data_format": "netcdf"},
                str(target))
        ds = xr.open_dataset(target)
        var = "tcwv" if "tcwv" in ds else list(ds.data_vars)[0]
        tdim = "valid_time" if "valid_time" in ds.dims else "time"
        a = ds[var].mean(dim=[d_ for d_ in ds[var].dims if d_ != tdim])
        t0, t1 = a[tdim].values[0], a[tdim].values[1]
        w = (t - t0) / (t1 - t0)
        pwv = float((1 - w) * a.values[0] + w * a.values[1])
        results[key] = pwv
        print(f"{key:14s} {s['t']}  PWV = {pwv:5.2f} kg/m2  "
              f"(hourly: {a.values[0]:.2f} -> {a.values[1]:.2f})")
    print("\nPaste into hotsat_lst.py SCENES:")
    for k, v in results.items():
        print(f'  {k}: pwv_kg_m2={v:.2f},')


if __name__ == "__main__":
    main()
