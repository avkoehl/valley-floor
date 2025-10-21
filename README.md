# Valley-Floor
This repository contains a python package for delineating valley floors from digital elevation models (DEMs) and hydrologic data.

There are two main methods for delineating valley floors:
1. **Region Floor**: This method identifies low-slope regions that are connected to the river network, suitable for larger (unconfined) valley floors.
2. **Flood Extent Floor**: This method identifies the valley floor based on flood stage thresholds, suitable for smaller (confined) valley floors.

The two can be combined to create a comprehensive valley floor delineation for a mountainous watershed.

# Installation

## From github to use as a dependency in your project

e.g. for poetry:
```toml
[tool.poetry.dependencies]
valley_floor = { git = "ssh://git@github.com/avkoehl/valley-floor.git", branch = "main" }
```

## From source for development

e.g. with poetry
```bash
git clone git@github.com:avkoehl/valley-floor.git
cd valley-floor
poetry install
```

# Usage

## Basic Usage

Import and use the package as follows:

```python
from valley_floor import Config
from valley_floor import load_sample_dem
from valley_floor import load_sample_flowlines
from valley_floor import delineate_valley_floor

config = Config()
dem = load_sample_dem()
flowlines = load_sample_flowlines()
valley_floors = delineate_valley_floor(dem, flowlines, config)
```

## Configuration
You can customize the delineation process by modifying the `Config` object:

```python
config = Config()
config.region_delineation.slope_threshold = 5
```

# Contact
Arthur Koehl
avkoehl at ucdavis.edu
