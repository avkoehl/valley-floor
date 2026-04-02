import dataclasses
from dataclasses import dataclass
import json


@dataclass
class PreprocessingParameters:
    # reach segmentation
    reach_penalty: int = 5
    reach_min_length: int = 1000
    reach_smooth_window: int = 5
    reach_threshold_degrees: float = 1.0
    # headwater filtering
    headwater_min_catchment_area: int = 1_500_000
    headwater_max_mean_slope: float = 5.0
    # cross section generation
    xs_interval_distance: int = 100
    xs_length: int = 1500
    xs_point_spacing: int = 10

    def to_json(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class Parameters:
    # region growing
    region_smooth_radius: int = 90
    region_smooth_sigma: int = 30
    region_slope_threshold: float = 3.0
    region_dilation_radius: int = 3
    # reach flooding
    flood_steep_slope: float = 10.0
    flood_min_elevation_gain: float = 10.0
    flood_default_elevation: int = 10
    flood_percentile: float = 85.0
    flood_min_points: int = 10

    def to_json(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class PostprocessingParameters:
    min_size: int = 40_000
    max_slope: float = 12.0

    def to_json(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
