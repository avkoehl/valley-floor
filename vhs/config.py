import dataclasses
from dataclasses import dataclass, field, fields
import json

import geopandas as gpd
import xarray as xr


def _param(default, group, unit, description):
    """A dataclass field carrying table metadata (group, unit, description)."""
    return field(
        default=default,
        metadata={"group": group, "unit": unit, "description": description},
    )


@dataclass
class Parameters:
    # headwater filtering
    headwater_min_length: int = _param(
        1_000, "Headwater filtering", "m",
        "Tip reaches shorter than this are treated as headwaters and dropped "
        "from valley-floor mapping (their channel pixels are reattached at the end).",
    )
    headwater_max_mean_slope: float = _param(
        5.0, "Headwater filtering", "degrees",
        "Tip reaches whose mean channel slope exceeds this are treated as "
        "headwaters and dropped.",
    )

    # region growing
    region_smooth_sigma: int = _param(
        90, "Region growing", "m",
        "Gaussian smoothing length applied to the slope surface before region "
        "growing; larger values bridge small rough patches.",
    )
    region_slope_threshold: float = _param(
        3.0, "Region growing", "degrees",
        "Maximum slope for a pixel to be grown into the valley floor from the "
        "channel network; lower values give tighter, more confined floors.",
    )

    # cross-section sampling
    xs_interval_distance: int = _param(
        100, "Cross-section sampling", "m",
        "Spacing between cross-sections sampled along each reach.",
    )
    xs_length: int = _param(
        1_500, "Cross-section sampling", "m",
        "Total length of each cross-section (extends this far to either side "
        "of the channel).",
    )
    xs_point_spacing: int = _param(
        10, "Cross-section sampling", "m",
        "Spacing between elevation sample points along each cross-section.",
    )

    # reach flooding
    flood_steep_slope: float = _param(
        10.0, "Reach flooding", "degrees",
        "Minimum slope for a cross-section segment to count as a valley wall "
        "when detecting slope breaks.",
    )
    flood_min_elevation_gain: float = _param(
        10.0, "Reach flooding", "m",
        "Minimum elevation gain across a steep segment to confirm it as a "
        "valley wall (slope-break point).",
    )
    flood_default_hand: int = _param(
        10, "Reach flooding", "m",
        "Fallback HAND threshold used for a reach when too few valid "
        "slope-break points are found.",
    )
    flood_percentile: float = _param(
        85.0, "Reach flooding", "percentile",
        "Percentile of slope-break HAND values used as the reach's flood "
        "threshold; higher values flood wider.",
    )
    flood_min_points: int = _param(
        10, "Reach flooding", "count",
        "Minimum number of valid slope-break points a reach needs before its "
        "threshold is computed from data instead of the default.",
    )

    # postprocessing
    min_hole_size: int = _param(
        40_000, "Postprocessing", "m²",
        "Holes in the valley floor smaller than this are filled; set to 0 to "
        "disable hole filling.",
    )
    max_slope: float = _param(
        15.0, "Postprocessing", "degrees",
        "Pixels steeper than this are removed from the final valley floor.",
    )

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def describe(cls) -> list[dict]:
        """Return per-parameter metadata rows (name, default, unit, group,
        description) for building documentation tables."""
        rows = []
        for f in fields(cls):
            if not f.metadata:
                continue
            rows.append(
                {
                    "name": f.name,
                    "default": f.default,
                    "unit": f.metadata["unit"],
                    "group": f.metadata["group"],
                    "description": f.metadata["description"],
                }
            )
        return rows


@dataclass
class ValleyFloorDetailed:
    valley_floor: xr.DataArray
    region_floor: xr.DataArray
    flood_floor: xr.DataArray
    slope_break_pts: gpd.GeoDataFrame
    reach_thresholds: dict[int, float]
