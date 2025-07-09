example:

```python
import matplotlib.pyplot as plt
import rioxarray as rxr
import geopandas as gpd

from valley_floor.flood import flood_extent
from valley_floor.region import low_slope_region
from valley_floor.postprocess import process_floor
from valley_floor.dynamic_flood.valley_transition import find_wallpoints
from valley_floor.dynamic_flood.flood_threshold import determine_flood_threshold


dem = rxr.open_rasterio("./data/1805000203_dem.tif", masked=True).squeeze()
flowlines = gpd.read_file("./data/1805000203_flowlines.gpkg")


# STREAMTOOLS PREPROCESSING
from streamtools.core import process_catchment_hydro
from streamtools.smooth import gaussian_smooth_raster
from streamtools.wbw import compute_slope
from streamtools.wbw import upstream_channel_length

catchment_data = process_catchment_hydro(dem, flowlines)

# REGION FLOOR
upstream_channel_len = upstream_channel_length(
    catchment_data["pointer"], catchment_data["raster_channels"]
)
trimmed_channel_network = catchment_data["raster_channels"].where(
    upstream_channel_len > 1000
)
smoothed_dem_region = gaussian_smooth_raster(dem, spatial_radius=90, sigma=30)
smoothed_slope_region = compute_slope(smoothed_dem_region)
connect = low_slope_region(
    smoothed_slope_region,
    trimmed_channel_network,
    slope_threshold=3.5,
    dilation_radius=3,
)

# RIVER VALLEY FLOOR
smoothed_dem = gaussian_smooth_raster(dem, spatial_radius=30, sigma=10)
smoothed_slope = compute_slope(smoothed_dem)
wallpoints = find_wallpoints(
    catchment_data["profiles"],
    smoothed_dem,
    smoothed_slope,
    min_slope=6,
    elevation_threshold=3,
)
thresholds = determine_flood_threshold(
    wallpoints,
    catchment_data["subbasins"],
    catchment_data["hand"],
    min_points=10,
    percentile=80,
    buffer=1,
    default_threshold=10,
)
flood = flood_extent(
    catchment_data["hand"], 
    smoothed_slope, 
    catchment_data['raster_channels'], 
    10, 
    thresholds, 
    catchment_data["subbasins"]
)

# COMBINE AND CLEAN
processed = process_floor(
    flood,
    connect,
    catchment_data["raster_channels"],
    40000,
    catchment_data["subbasins"],
)

```
