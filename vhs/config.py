import dataclasses
from dataclasses import dataclass, field
import json

import geopandas as gpd
import xarray as xr


@dataclass
class Parameters:
    # headwater filtering
    headwater_min_length: int = 1_000  # meters
    headwater_max_mean_slope: float = 5.0  # degrees

    # region growing
    region_smooth_sigma: int = 90  # meters
    region_slope_threshold: float = 3.0  # degrees

    # cross section sampling
    xs_interval_distance: int = 100  # meters
    xs_length: int = 1_500  # meters
    xs_point_spacing: int = 10  # meters

    # reach flooding
    flood_steep_slope: float = 10.0  # degrees
    flood_min_elevation_gain: float = 10.0  # meters
    flood_default_hand: int = 10  # meters
    flood_percentile: float = 85.0
    flood_min_points: int = 10

    # postprocessing
    min_hole_size: int = 40_000  # m²
    max_slope: float = 15.0  # degrees

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ValleyFloorDetailed:
    valley_floor: xr.DataArray
    region_floor: xr.DataArray
    flood_floor: xr.DataArray
    slope_break_pts: gpd.GeoDataFrame
    reach_thresholds: dict[int, float]
