import dataclasses
from dataclasses import dataclass
import json


@dataclass
class PreprocessingParameters:
    """Parameters for the streamkit-based preprocessing pipeline.

    Controls reach segmentation, headwater filtering, and cross section generation.
    All spatial units are in the CRS units of the input DEM (typically meters).

    Attributes:
        reach_penalty: PELT algorithm penalty for reach segmentation. Higher values
            produce fewer, longer reaches.
        reach_min_length: Minimum reach length in meters.
        reach_smooth_window: Window size for smoothing slope before segmentation.
        reach_threshold_degrees: Merge adjacent reaches if slope difference is below
            this threshold in degrees.
        headwater_min_catchment_area: Minimum contributing area in square meters to
            retain a headwater reach.
        headwater_max_mean_slope: Maximum mean slope in degrees for a headwater reach
            to be retained.
        xs_interval_distance: Distance between cross sections in meters.
        xs_length: Total length of each cross section in meters.
        xs_point_spacing: Distance between sample points along each cross section in meters.
    """

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
    """Parameters for the core valley floor delineation algorithm.

    Controls the region growing and reach flooding components.
    All spatial units are in the CRS units of the input DEM (typically meters).

    Attributes:
        region_smooth_radius: Radius of the Gaussian smoothing kernel in meters,
            applied to the DEM before slope calculation in region growing.
        region_smooth_sigma: Standard deviation of the Gaussian smoothing kernel
            in pixels.
        region_slope_threshold: Maximum slope in degrees for a pixel to be
            included in the region growing component.
        region_dilation_radius: Radius in pixels for dilating the channel network
            before region growing connectivity analysis.
        flood_steep_slope: Minimum slope in degrees to identify a valley wall
            in cross section analysis.
        flood_min_elevation_gain: Minimum elevation gain in meters for a steep
            segment to be considered a valley wall.
        flood_default_elevation: Default HAND threshold in meters applied to
            reaches with insufficient cross section data.
        flood_percentile: Percentile of slope break elevations used to derive
            the HAND threshold per reach.
        flood_min_points: Minimum number of slope break points required to
            derive a reach threshold from data rather than the default.
    """

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
    """Parameters for postprocessing the valley floor raster.

    Attributes:
        min_size: Holes smaller than this size in square meters are filled in as valley floor.
        max_slope: Maximum slope in degrees. Pixels exceeding this are excluded
            from the valley floor regardless of other criteria.
    """

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
