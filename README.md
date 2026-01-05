# Valley-Floor

This repository contains a python package for delineating valley floors from
digital elevation models (DEMs) of mountainous watersheds. Valley floors are
the topographic region between valley walls (hillslopes) that are mainly shaped by
fluvial processes and are composed of alluvial fans, floodplains, terraces, and
channels.

The method predicts valley floor pixels by combining pixels identified by either of two components:

1. Low-slope pixels with connectivity to the channel network. This is
   especially useful for larger flatter unconfined valley floors where multiple
channels and floodplains exist.
2. Reach specific relative elevation to channel thresholding, where the
   thresholds are determined from analysis of cross sections detecting
transition to sustained high slopes (valley walls). This is especially useful
for confined steep valley floors where floodplains are narrow (but may be high
slope and have roughness) and closely bound by hillslopes.


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
from valley_floor import delineate_valley_floor
from valley_floor.data import load_sample_data

config = Config()
dem, channel_heads = load_sample_data()
valley_floors = delineate_valley_floor(dem, channel_heads, config)
```

## Configuration
You can customize the delineation process by modifying the `Config` object:

```python
config = Config()
config.cross_section.interval_distance = 50 # meters between cross sections
```

# Contact
Arthur Koehl
avkoehl at ucdavis.edu
