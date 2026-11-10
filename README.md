# Nocturnal surface brightness temperature of three UAE urban typologies (HotSat-1)

Code accompanying the article:

> [Authors] ([year]). *[Title]*. Urban Climate. [DOI to be added]

The scripts analyse night-time surface brightness temperature (SBT) from the
HotSat-1 mid-wave infrared sensor (3.7–4.95 µm) over three districts in the
United Arab Emirates: a sub-urban residential district and an industrial
district in Abu Dhabi, and a dense urban district in Dubai. All analyses run
on a common 10 m grid. They comprise:

1. descriptive statistics of SBT per scene and land-cover class;
2. semivariograms of the spatial scale of thermal structure;
3. roof-to-ground contrasts from individual building footprints;
4. an XGBoost + SHAP attribution of SBT to urban-form predictors with
   spatial-block cross-validation;
5. supplementary checks: a paired roof-minus-ring contrast with a block
   bootstrap, co-registration and view-angle checks, directional
   variograms, collinearity and a resolution-degradation experiment.

## Scenes

| Key | HotSat-1 scene ID | Acquisition (UTC) | Typology |
|---|---|---|---|
| `AD_suburban` | `20231029_213936_HS1_SBT` | 2023-10-29 21:39 | sub-urban |
| `AD_industrial` | `20231030_212656_HS1_SBT` | 2023-10-30 21:26 | industrial |
| `DXB_urban` | `20231120_213935_HS1_SBT` | 2023-11-20 21:39 | urban |

## Data

No data are included. HotSat-1 imagery is a commercial product from SatVu and
cannot be redistributed. The other inputs are open and are downloaded by the
scripts or from the sources below.

| Input | Source | Obtained by |
|---|---|---|
| HotSat-1 SBT and Usable Data Mask (UDM) | SatVu | manual, placed in `hotsat/` |
| Sentinel-2 L2A, ESA WorldCover v200, Copernicus DEM | Microsoft Planetary Computer | `predictor_stack.py` |
| WSF-3D (building fraction, height, area, volume) | DLR EOC Geoservice | manual, placed in `ancillary/` |
| GlobalBuildingAtlas LoD1 footprints | TUM WFS service | `download_gba_wfs.py` |
| VIIRS VNP21A1N night LST | NASA Earthdata | `coarse_lst.py` (Earthdata login) |
| ERA5 total-column water vapour | Copernicus CDS | `era5_pwv_lookup.py` (CDS account) |

### Folder layout

Set the environment variable `UAE_SBT_DATA_ROOT` to the data folder. If it is
not set, `./data` next to this repository is used (see `scripts/config.py`).

```
<UAE_SBT_DATA_ROOT>/
├── hotsat/        <scene_id>.tiff, <scene_id>_udm.tiff           (input)
├── ancillary/     WSF3D_V02_Building{Height,Fraction,Area,Volume}.tif  (input)
├── buildings_gba/ GBA_<key>_3857.geojson                         (step 1)
├── predictors/    <scene_id>_predictors_10m.nc                   (step 2)
├── lst/           <scene_id>_LST_C.tif, <scene_id>_LST_QA.tif    (steps 4, 6, 7)
├── coarse_lst/    VIIRS products                                 (step 5)
└── analysis/      tables (CSV) and figures (PNG)                 (steps 6, 8–13)
```

## Installation

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

or, with conda, create an environment and install the same packages. On
Windows, mixed conda/pip installations can give PROJ database errors.
`scripts/proj_fix.py` is imported by the scripts before rasterio and resolves
this automatically.

## Running the pipeline

Run the scripts from the `scripts/` folder, in this order.

| Step | Command | Purpose |
|---|---|---|
| 1 | `python download_gba_wfs.py` | building footprints |
| 2 | `python predictor_stack.py` | 10 m predictor stack (uses `wsf3d_buildings.py`) |
| 3 | `python era5_pwv_lookup.py` | optional: water vapour for Table A1 |
| 4 | `python hotsat_lst.py` | quality mask and exploratory long-wave LST (see note) |
| 5 | `python coarse_lst.py` | VIIRS LST for the scene offsets |
| 6 | `python cloud_mask_extra.py AD_industrial` | preview the extra cloud mask (PNG) |
|   | `python cloud_mask_extra.py AD_industrial --apply` | write it into the QA raster |
| 7 | `python use_sbt_inputs.py` | switch the analysis input from LST to SBT |
| 8 | `python analysis_task1_descriptive.py` | Task 1: Table 2, Table A2, Figure 3 |
| 9 | `python analysis_task2_semivariogram.py` | Task 2: Table 3, Figure 5 |
| 10 | `python analysis_task3_building_contrast.py` | Task 3: Tables 4–5, Figure 4 |
| 11 | `python analysis_task4_form_attribution.py` | Task 4: Tables 6–7, Figures 6–7 |
| 12 | `python analysis_task5_supplementary.py` | Task 5: Tables 4, A3–A6, Figure 8, Figure S1 |
| 13 | `python figure_typology_comparison.py` | Figure 2 |

Step 12 needs the building-class rasters written by step 10. Parts of Task 5
can be run separately, e.g. `python analysis_task5_supplementary.py ring dir`
(parts: `ring`, `dir`, `corr`, `degrade`).

### Notes

- **SBT, not LST.** The analyses use the provider's SBT product, which is
  corrected for atmospheric effects but not for emissivity. Step 4 also
  writes an exploratory long-wave LST. It is used only to define two quality
  flags: the hot-anomaly flag (LST > 40 °C) and the seeds of the extra cloud
  mask (LST below 4 °C, grown through connected pixels below 6 °C and
  dilated by 20 m; about 7.5 and 9 °C in SBT). Step 7 then replaces it with
  SBT in °C under the same file names (`*_LST_C.tif`), so every later step
  reads SBT. The original files are kept in `lst_LWIR_retrieval/`, and
  `python use_sbt_inputs.py --restore` reverses the switch. Do not re-run
  step 4 after step 7, because it would overwrite the SBT.
- **Quality bits** in `*_LST_QA.tif`: 1 nodata, 2 low-confidence land cover,
  4 oblique view, 8 cloud (UDM or extra mask), 16 hot anomaly. Pixels with
  bit 1, 2, 8 or 16 set are excluded from all analyses.
- **Randomness.** Subsampling, cross-validation and bootstrap use fixed
  seeds. Small differences can still occur between library versions.
- **Internet access** is required for steps 1–3 and 5 (Planetary Computer,
  TUM WFS, Copernicus CDS, NASA Earthdata).

## Citation

If you use this code, please cite the article above.

## License

[Add a license, e.g. MIT for code.]
