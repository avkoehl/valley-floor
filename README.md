# valley-floor

A Python package for delineating valley floors from digital elevation models (DEMs).

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

Basic usage:
```python
from valley_floor import (
    Parameters, 
    PreProcessingParameters,
    PostProcessingParameters,
    delineate_from_dem_and_flowlines
)

# dem is an xarray.DataArray of elevation values
# flowlines is a gpd.GeoDataFrame of flowline geometries 
result = delineate_from_dem_and_flowlines(
    dem=dem,
    flowlines=flowlines,
    params=Parameters(),
    preprocessing_params=PreProcessingParameters(),
    postprocessing_params=PostProcessingParameters()
)
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
