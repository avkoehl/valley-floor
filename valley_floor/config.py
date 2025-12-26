from dataclasses import dataclass, field


@dataclass
class ReachParameters:
    penalty: int = 5
    min_length: int = 1000
    smooth_window: int = 5
    threshold_degrees: float = 1.0


@dataclass
class HeadwaterFilterParameters:
    min_stream_order: int = 2
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
