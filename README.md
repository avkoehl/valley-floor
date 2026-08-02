# vhs 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21685156.svg)](https://doi.org/10.5281/zenodo.21685156)

vhs (valley hillslope separator) is a Python package for delineating valley floors from digital elevation models (DEMs).

![Valley floor delineation from a conditioned DEM and a labeled channel network](https://raw.githubusercontent.com/avkoehl/vhs/main/assets/graphical_abstract.png)

Valley floors are the topographic region between valley walls shaped mainly by fluvial
processes, composed of floodplains, terraces, alluvial fans, and channels. From a
hydrologically conditioned DEM and a reach-labeled channel network, `vhs` produces a
binary valley floor raster. 

The method combines two components:

1. **Region growing** — low-slope pixels connected to the channel network. Captures
   wider, unconfined valley floors with multiple channels and floodplains.
2. **Reach flooding** — reach-specific relative elevation thresholds derived from
   cross-section analysis. Captures narrow, confined valley floors bounded by steep
   hillslopes.

## Installation

```bash
pip install valleyfloor
```

> **Note:** `valleyfloor` is the PyPI distribution name; the import package name is `vhs`,
> i.e. `import vhs`.

## Usage

With your own conditioned DEM and channel network (both `xarray.DataArray`s on
the same grid). `dem` must be hydrologically conditioned - D8 flow directions
are computed from it internally,  `channel_network` (a reach-labeled raster)
must align with the hydro conditioned dem.

```python
from vhs import map_valley_floor, Parameters

valley_floor = map_valley_floor(
    dem=dem,
    channel_network=channel_network,
    params=Parameters(),   # optional; defaults shown in the table below
)
```

### Parameters

```python
from vhs import Parameters

params = Parameters(
    region_slope_threshold=2.0,   # degrees, tighter region growing
    flood_percentile=90.0,        # higher flood threshold per reach
    min_hole_size=20_000,         # m², smaller holes filled
)
valley_floor = map_valley_floor(dem, channel_network, params=params)
```

Parameters can be saved and reloaded as JSON with `params.to_json(path)` and
`Parameters.from_json(path)`.

## Configuration parameters

All parameters live on the `Parameters` dataclass, grouped by pipeline stage.

#### Headwater filtering

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `headwater_min_length` | 1,000 | m | Tip reaches shorter than this are treated as headwaters and dropped from valley-floor mapping (their channel pixels are reattached at the end). |
| `headwater_max_mean_slope` | 5.0 | degrees | Tip reaches whose mean channel slope exceeds this are treated as headwaters and dropped. |

#### Region growing

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `region_smooth_sigma` | 90 | m | Gaussian smoothing length applied to the slope surface before region growing; larger values bridge small rough patches. |
| `region_slope_threshold` | 3.0 | degrees | Maximum slope for a pixel to be grown into the valley floor from the channel network; lower values give tighter, more confined floors. |

#### Cross-section sampling

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `xs_interval_distance` | 100 | m | Spacing between cross-sections sampled along each reach. |
| `xs_length` | 1,500 | m | Total length of each cross-section (extends this far to either side of the channel). |
| `xs_point_spacing` | 10 | m | Spacing between elevation sample points along each cross-section. |

#### Reach flooding

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `flood_steep_slope` | 10.0 | degrees | Minimum DEM slope (direction-independent, not measured along the cross-section line) for a point to count as part of a valley wall when detecting slope breaks. |
| `flood_slope_window` | 30.0 | m | Distance over which the DEM slope raster is smoothed before slope breaks are detected, converted internally to a pixel window based on the DEM's resolution; damps single-cell noise that would otherwise interrupt an otherwise-steep run. Set to 0 to disable. |
| `flood_min_elevation_gain` | 10.0 | m | Minimum elevation gain across a steep segment to confirm it as a valley wall (slope-break point). |
| `flood_default_hand` | 10 | m | Fallback elevation-gain threshold used for a reach when too few valid slope-break points are found. |
| `flood_percentile` | 85.0 | percentile | Percentile of slope-break elevation-gain values used as the reach's flood threshold; higher values flood wider. |
| `flood_min_points` | 10 | count | Minimum number of valid slope-break points a reach needs before its threshold is computed from data instead of the default. |

#### Postprocessing

| Parameter | Default | Unit | Description |
|---|---|---|---|
| `min_hole_size` | 40,000 | m² | Holes in the valley floor smaller than this are filled; set to 0 to disable hole filling. |
| `max_slope` | 15.0 | degrees | Pixels steeper than this are removed from the final valley floor. |
