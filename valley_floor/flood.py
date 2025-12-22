"""

Use thresholds of elevation above stream and slope to delineate floodplain and terrace (floor) extent.

inputs:
    - detrended DEM (likely HAND) (meters)
    - slope raster (degrees)
    - slope threshold (optional, default 10 degrees)
    - elevation threshold (optional, int or dict, default 10 meters)
    - subbasin raster (optional, for subbasin-specific floor delineation)

outputs:
    - valley floor raster (boolean, 1 for floor, 0 for non-floor)
"""

import numpy as np

from valley_floor.postprocess import burnin_streams
from valley_floor.postprocess import remove_isolated_areas


def flood_extent(
    detrended_dem,
    slope,
    channel_network,
    slope_threshold=10.0,
    elevation_threshold=10.0,
    subbasin=None,
):
    floor = detrended_dem.copy(deep=True)
    floor.data = np.zeros(detrended_dem.shape, dtype=bool)
    floor.rio.write_nodata(0)

    if subbasin is not None:
        if not isinstance(elevation_threshold, dict):
            raise ValueError(
                "If subbasin is provided, elevation_threshold must be a dict with subbasin keys."
            )

        subbasins = np.unique(subbasin.data)
        subbasins = subbasins[np.isfinite(subbasins)]

        if not all(key in elevation_threshold.keys() for key in subbasins):
            raise ValueError(
                "Not all subbasins have an elevation threshold defined in elevation_threshold."
            )

        # apply floor delineation for each subbasin
        for subbasin_id, threshold in elevation_threshold.items():
            if subbasin_id in subbasin:
                # apply floor delineation for this subbasin
                sub_mask = (slope <= slope_threshold) & (detrended_dem <= threshold)
                floor.data[subbasin == subbasin_id] = sub_mask.data[
                    subbasin == subbasin_id
                ]
    else:
        # apply floor delineation globally
        if isinstance(elevation_threshold, dict):
            raise ValueError(
                "If subbasin is None, elevation_threshold must be a single value."
            )
        sub_mask = (slope <= slope_threshold) & (detrended_dem <= elevation_threshold)
        floor.data = sub_mask

    floor = burnin_streams(floor, channel_network)
    floor = remove_isolated_areas(floor, channel_network)
    floor = floor.astype(np.uint8)
    return floor
