from pathlib import Path

import geopandas as gpd
import rioxarray
from platformdirs import user_cache_dir

CACHE_DIR = Path(user_cache_dir("vhs"))
HUC_ID = "1805000203"


def load_sample_data():
    """Download and cache sample DEM and flowlines for a small HUC watershed.

    Requires streamkit as an optional dependency.
    Returns (dem, flowlines) suitable for passing to prepare_inputs.
    """
    try:
        from streamkit.datasets import download_huc_data
    except ImportError:
        raise ImportError(
            "load_sample_data requires streamkit. "
            'Install it with: pip install "valley-floor[streamkit] @ git+https://github.com/avkoehl/valley-floor.git"'
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dem_path = CACHE_DIR / "sample_dem.tif"
    flowlines_path = CACHE_DIR / "sample_flowlines.gpkg"

    if not dem_path.exists() or not flowlines_path.exists():
        print(f"Sample data not found in cache. Downloading to {CACHE_DIR}...")
        dem, flowlines = download_huc_data(HUC_ID, nhd_layer="high", crs="EPSG:3310")
        dem.rio.to_raster(dem_path, compress="LZW")
        flowlines.to_file(flowlines_path, driver="GPKG")
    else:
        print(f"Loading sample data from cache at {CACHE_DIR}...")

    dem = rioxarray.open_rasterio(dem_path, masked=True).squeeze()
    flowlines = gpd.read_file(flowlines_path)
    return dem, flowlines
