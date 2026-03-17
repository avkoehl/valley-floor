import dataclasses
from dataclasses import dataclass, field
import json


@dataclass
class ReachParameters:
    penalty: int = 5
    min_length: int = 1000
    smooth_window: int = 5
    threshold_degrees: float = 1.0


@dataclass
class HeadwaterFilterParameters:
    min_catchment_area: int = 1_500_000
    max_mean_slope: float = 5.0


@dataclass
class RegionParameters:
    smooth_radius: int = 90
    smooth_sigma: int = 30
    slope_threshold: float = 3.0
    dilation_radius: int = 3


@dataclass
class CrossSectionParameters:
    interval_distance: int = 100
    length: int = 1500
    point_spacing: int = 10


@dataclass
class SlopeBreakParameters:
    steep_slope: float = 10.0
    min_elevation_gain: float = 10.0


@dataclass
class ThresholdParameters:
    default_elevation: int = 10
    percentile: float = 85.0
    min_points: int = 10


@dataclass
class PostProcessParameters:
    min_size: int = 40_000
    max_slope: float = 12.0


@dataclass
class Config:
    reach: ReachParameters = field(default_factory=ReachParameters)
    headwater_filter: HeadwaterFilterParameters = field(
        default_factory=HeadwaterFilterParameters
    )
    region: RegionParameters = field(default_factory=RegionParameters)
    cross_section: CrossSectionParameters = field(
        default_factory=CrossSectionParameters
    )
    slope_break: SlopeBreakParameters = field(default_factory=SlopeBreakParameters)
    threshold: ThresholdParameters = field(default_factory=ThresholdParameters)
    post_process: PostProcessParameters = field(default_factory=PostProcessParameters)

    def to_json(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
        return cls(
            reach=ReachParameters(**data["reach"]),
            headwater_filter=HeadwaterFilterParameters(**data["headwater_filter"]),
            region=RegionParameters(**data["region"]),
            cross_section=CrossSectionParameters(**data["cross_section"]),
            slope_break=SlopeBreakParameters(**data["slope_break"]),
            threshold=ThresholdParameters(**data["threshold"]),
            post_process=PostProcessParameters(**data["post_process"]),
        )
