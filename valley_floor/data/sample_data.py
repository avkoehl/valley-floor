from platformdirs import user_cache_dir
from pathlib import Path
import rioxarray

from streamkit.datasets import get_huc_data
from streamkit.extraction import heads_from_features

CACHE_DIR = Path(user_cache_dir("valley_floor"))


def load_sample_data():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dem_path = CACHE_DIR / "sample_dem.tif"
    channel_heads_path = CACHE_DIR / "sample_channel_heads.tif"

    if not dem_path.exists() or not channel_heads_path.exists():
        print(f"Sample data not found in cache. Downloading to {CACHE_DIR}...")
        dem, flowlines = get_huc_data("1805000203", nhd_layer="high", crs="EPSG:3310")
        heads = heads_from_features(flowlines, dem)
        dem.rio.to_raster(dem_path, compress="LZW")
        heads.rio.to_raster(channel_heads_path, compress="LZW")

    else:
        print(f"Loading sample data from cache at {CACHE_DIR}...")

    dem = rioxarray.open_rasterio(dem_path).squeeze()
    channel_heads = rioxarray.open_rasterio(channel_heads_path).squeeze()
    return dem, channel_heads
