# Step 1. Download GlobalBuildingAtlas (GBA) LoD1 building footprints for the
# study areas via the TUM WFS service.

from __future__ import annotations

import logging
from pathlib import Path
from config import DATA_ROOT
from urllib.parse import urlencode

import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

WFS_BASE = "https://tubvsig-so2sat-vm1.srv.mwn.de/geoserver/ows"

OUT_DIR = DATA_ROOT / "buildings_gba"

# AOIs as (name, lon_min, lat_min, lon_max, lat_max) in EPSG:4326,
# with a small margin so edge buildings are kept.
AOIS = {
    "AD_suburban":   (54.985, 24.165, 55.078, 24.230),
    "AD_industrial": (54.466, 24.347, 54.521, 24.392),
    "DXB_urban":     (55.096, 24.929, 55.190, 24.995),
}
AOI_MARGIN_DEG = 0.01

# Filled in automatically by discover_layers(); or set by hand if the
# automatic pick is wrong (see the printed layer list).
LAYER_NAME: str | None = None

REQUEST_TIMEOUT = 120


# ===========================================================================
# Step 1 -- discover what the WFS offers
# ===========================================================================

def discover_layers() -> list[str]:
    params = {
        "service": "WFS", "version": "2.0.0",
        "request": "GetCapabilities",
    }
    url = f"{WFS_BASE}?{urlencode(params)}"
    logger.info("querying WFS GetCapabilities ...")
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    # parse feature type names without assuming a namespace prefix
    import xml.etree.ElementTree as ET
    root = ET.fromstring(r.text)
    names = []
    for el in root.iter():
        if el.tag.endswith("}Name") or el.tag == "Name":
            parent_is_ft = True   # Name elements appear inside FeatureType
            if el.text and parent_is_ft:
                names.append(el.text.strip())
    # FeatureType/Name entries usually contain a ':' (workspace:layer)
    layers = sorted(set(n for n in names if ":" in n or n))
    logger.info("WFS advertises %d name entries; candidate layers:",
                len(layers))
    for n in layers:
        logger.info("   %s", n)
    return layers


def describe_layer(layer: str) -> list[str]:
    params = {
        "service": "WFS", "version": "2.0.0",
        "request": "DescribeFeatureType", "typeNames": layer,
    }
    url = f"{WFS_BASE}?{urlencode(params)}"
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        logger.warning("DescribeFeatureType failed for %s: %s", layer, e)
        return []
    import xml.etree.ElementTree as ET
    root = ET.fromstring(r.text)
    attrs = []
    for el in root.iter():
        if el.tag.endswith("}element") or el.tag == "element":
            nm = el.attrib.get("name")
            if nm:
                attrs.append(nm)
    logger.info("layer %s attributes: %s", layer, attrs)
    return attrs


# ===========================================================================
# Step 2 -- fetch features for one AOI bounding box
# ===========================================================================

def fetch_aoi(layer: str, name: str, bbox4326, out_dir: Path):
    lon0, lat0, lon1, lat1 = bbox4326
    m = AOI_MARGIN_DEG
    # WFS 2.0 with a 4326 CRS URI expects BBOX as lat,lon (axis order);
    # stating the CRS makes the server interpret it correctly.
    bbox = f"{lat0 - m},{lon0 - m},{lat1 + m},{lon1 + m}," \
           f"urn:ogc:def:crs:EPSG::4326"
    params = {
        "service": "WFS", "version": "2.0.0",
        "request": "GetFeature", "typeNames": layer,
        "bbox": bbox, "outputFormat": "application/json",
        "srsName": "EPSG:3857",
    }
    url = f"{WFS_BASE}?{urlencode(params)}"
    logger.info("[%s] requesting features ...", name)
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        logger.error("[%s] WFS returned HTTP %d: %s",
                      name, r.status_code, r.text[:300])
        return None
    if not r.text.strip().startswith("{"):
        # server returned an XML exception rather than JSON
        logger.error("[%s] WFS did not return JSON. First 300 chars:\n%s",
                      name, r.text[:300])
        return None

    raw = out_dir / f"GBA_{name}_raw3857.geojson"
    raw.write_text(r.text, encoding="utf-8")

    import geopandas as gpd
    gdf = gpd.read_file(raw)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3857")     # GBA is always 3857
    logger.info("[%s] %d buildings, columns=%s",
                name, len(gdf), list(gdf.columns))

    # save a clean 3857 copy and a 4326 copy
    gdf.to_file(out_dir / f"GBA_{name}_3857.geojson", driver="GeoJSON")
    gdf.to_crs("EPSG:4326").to_file(
        out_dir / f"GBA_{name}_4326.geojson", driver="GeoJSON")
    return gdf


# ===========================================================================
# Driver
# ===========================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    layers = discover_layers()

    layer = LAYER_NAME
    if layer is None:
        # heuristic: prefer a layer whose name mentions building / lod1
        for cand in layers:
            cl = cand.lower()
            if "lod1" in cl or "building" in cl or "gba" in cl:
                layer = cand
                break
    if layer is None:
        raise SystemExit(
            "Could not auto-pick a building layer. Set LAYER_NAME to one "
            "of the layers printed above and re-run.")
    logger.info("using layer: %s", layer)

    attrs = describe_layer(layer)
    has_height = any(a.lower() in ("height", "var", "height_m")
                     for a in attrs)
    if has_height:
        logger.info("OK -- layer carries a height attribute.")
    else:
        logger.warning("layer does NOT appear to carry a height "
                        "attribute. Footprints will still be saved; "
                        "heights then need the tile + produce_lod1.py "
                        "route.")

    any_ok = False
    for name, bbox in AOIS.items():
        gdf = fetch_aoi(layer, name, bbox, OUT_DIR)
        if gdf is not None and len(gdf) > 0:
            any_ok = True

    if any_ok:
        logger.info("Done. Inspect a GBA_*_4326.geojson file -- confirm "
                    "the height attribute name and units, then proceed "
                    "to the footprint-ingestion step.")
    else:
        logger.error("No AOI returned features. Check the layer name and "
                      "the WFS endpoint, or fall back to the tile route.")


if __name__ == "__main__":
    main()
