# vhs (Valley Hillslope Separator)

A Python package for delineating valley floors from digital elevation models (DEMs).

Documentation: https://avkoehl.github.io/valley-floor/

Valley floors are the topographic region between valley walls shaped mainly by fluvial
processes, composed of floodplains, terraces, alluvial fans, and channels.

The method combines two components:

1. **Region growing** — low-slope pixels connected to the channel network. Captures
   wider, unconfined valley floors with multiple channels and floodplains.
2. **Reach flooding** — reach-specific relative elevation thresholds derived from
   cross-section analysis. Captures narrow, confined valley floors bounded by steep
   hillslopes.

## Installation

Core package: 
```bash
pip install "valley-floor @ git+https://github.com/avkoehl/valley-floor.git"
```

With the optional preprocessing pipeline and example dataset (requires streamkit):
```bash
pip install "valley-floor[streamkit] @ git+https://github.com/avkoehl/valley-floor.git"
```

## Usage

### Core usage 

If you have your own HAND, channel network, and subbasins:

```python
from vhs import map_valley_floor, Parameters

# all four inputs are xarray.DataArrays on the same grid
valley_floor = map_valley_floor(
    dem=dem,
    hand=hand,
    channel_network=channel_network,  # reach-labeled raster, IDs match subbasins
    subbasins=subbasins,
    params=Parameters(),              # optional, defaults shown
)
# valley_floor is an xarray.DataArray (uint8, nodata=255)
```

For access to intermediate outputs (region floor, flood floor, slope break points,
per-reach HAND thresholds):

```python
from vhs import map_valley_floor_detailed

result = map_valley_floor_detailed(dem, hand, channel_network, subbasins)
result.valley_floor       # final binary raster
result.region_floor       # region growing component
result.flood_floor        # reach flooding component
result.slope_break_pts    # GeoDataFrame of detected valley wall points
result.reach_thresholds   # dict of {reach_id: HAND threshold}
```

### Convenience usage — with streamkit

If you have a raw DEM and vector flowlines, `prepare_inputs` runs a streamkit-based
workflow to produce the four required inputs:

```python
from vhs import map_valley_floor, prepare_inputs

dem, hand, channel_network, subbasins = prepare_inputs(dem, flowlines)
valley_floor = map_valley_floor(dem, hand, channel_network, subbasins)
```

### Tuning parameters

```python
from vhs import Parameters

params = Parameters(
    region_slope_threshold=2.0,   # degrees, tighter region growing
    flood_percentile=90.0,        # higher HAND threshold per reach
    min_hole_size=20_000,         # m², smaller holes filled
)
valley_floor = map_valley_floor(dem, hand, channel_network, subbasins, params=params)
```

## Developement

Install the package with development dependencies and register the package's
virtual environment as a Jupyter kernel:
```bash
git clone git@github.com:avkoehl/valley-floor.git
cd valley-floor
uv sync                        # core only
uv sync --extra streamkit --group dev  # include streamkit and development dependencies
uv run python -m ipykernel install --user --name valley-floor # register the package's virtual environment as a Jupyter kernel
```


Run tests with pytest:
```bash
uv run pytest -v
```

Build the documentation and preview it locally:
```bash
uv run quartodoc build
quarto preview
```

## Contact

Arthur Koehl — avkoehl at ucdavis.edu
