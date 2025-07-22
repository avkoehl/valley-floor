# Valley-Floor
This repository contains a python package for delineating valley floors from digital elevation models (DEMs) and hydrologic data.

There are two main methods for delineating valley floors:
1. **Region Floor**: This method identifies low-slope regions that are connected to the river network, suitable for larger (unconfined) valley floors.
2. **Flood Extent Floor**: This method identifies the valley floor based on flood stage thresholds, suitable for smaller (confined) valley floors.

The two can be combined to create a comprehensive valley floor delineation for a watershed.

For optimal results it is recommended to use the [`streamtools`](https://github.com/avkoehl/streamtools) package for preprocessing the DEM and hydrologic data. 
The examples below demonstrate how to use the `valley_floor` package in conjunction with `streamtools` for valley floor delineation.
If you are not using `streamtools`, you can still use the `valley_floor` package, but you will need to provide the necessary prepared inputs manually.

# Installation 
```bash
git clone https://github.com/avkoehl/streamtools.git
cd streamtools
poetry install

```

# Requirements
- Python 3.11 or higher
- Poetry for package management


# Valley Floor Delineation Example

## Preprocessing with `streamtools` package

```python
import rioxarray as rxr
import geopandas as gpd

from streamtools.core import process_catchment_hydro
from streamtools.core import process_catchment_hydro
from streamtools.smooth import gaussian_smooth_raster
from streamtools.wbw import compute_slope
from streamtools.wbw import upstream_channel_length

raw_usgs_dem = rxr.open_rasterio("./data/1805000203_dem.tif", masked=True).squeeze()
raw_nhd_flowlines = gpd.read_file("./data/1805000203_flowlines.gpkg")
results = process_catchment_hydro(
    raw_usgs_dem, raw_nhd_flowlines
)
flow_pointer = results["pointer"]
channel_network = results["raster_channels"]
dem = results["conditioned_dem"]
detrended_dem = results["hand"]
subbasins = results["subbasins"]
xs_coordinates = results["profiles"]

coarse_dem = gaussian_smooth_raster(
    dem, spatial_radius=90, sigma=30
)  # Smooth the DEM for region floor delineation
coarse_slope = compute_slope(coarse_dem)  # Compute slope for region floor delineation
smoothed_dem = gaussian_smooth_raster(
    dem, spatial_radius=30, sigma=10
)  # Smooth the DEM for flood extent floor delineation
smoothed_slope = compute_slope(smoothed_dem)  # Compute slope for flood extent floor delineation

upstream_channel_len = upstream_channel_length(
    flow_pointer, channel_network
)  # Compute upstream channel length for trimming the channel network
trimmed_channel_network = channel_network.where(
    upstream_channel_len > 1000
)  # Trim the channel network to remove small headwater channels
```

## Region Floor
```python
from valley_floor.region import low_slope_region

region_floor = low_slope_region(
    coarse_slope,
    trimmed_channel_network,
    slope_threshold=3.5,
    dilation_radius=3,
)
```

## Flood Extent Floor Fixed Threshold
```python
from valley_floor.flood import flood_extent

flood_floor = flood_extent(
    hand,
    smoothed_slope,
    channel_network,
    10,
    thresholds,
    subbasins
)

```

## Flood Extent Floor Dynamic Threshold

```python
from valley_floor.dynamic_flood.valley_transition import label_wallpoints
from valley_floor.dynamic_flood.flood_threshold import determine_flood_threshold

labeled_xsections = label_wallpoints(
    xs_coordinates,
    smoothed_dem,
    smoothed_slope,
    min_slope=6,
    elevation_threshold=3,
)
thresholds = determine_flood_threshold(
    labeled_xsections.loc[labeled_xsections["is_wallpoint"]],
    subbasins,
    hand,
    min_points=10,
    percentile=80,
    buffer=1,
    default_threshold=10,
)
flood_floor = flood_extent(
    hand,
    smoothed_slope,
    channel_network,
    10,
    thresholds,
    subbasins
)

```


## Postprocessing

```python
from valley_floor.postprocess import process_floor

processed = process_floor(
    flood_floor,
    region_floor,
    channel_network,
    40000,
    subbasins,
)

```
