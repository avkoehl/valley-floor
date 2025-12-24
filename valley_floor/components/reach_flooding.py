import numpy as np
import xarray as xr


def flood_reaches():
    wallpoints = find_transition_points()
    reach_thresholds = derive_reach_thresholds()
    flooded_reaches = flood_reaches(detrended_dem, reach_thresholds, subbasins)
    return flooded_reaches


def apply_flooding(detrended_dem, elevation_thresholds, subbasins):
    floor = detrended_dem.copy(data=np.zeros_like(detrended_dem.data), dtype=np.uint8)

    subbasin_ids = np.unique(subbasins.data)
    subbasin_ids = subbasin_ids[np.isfinite(subbasin_ids)]
    # confirm each subbasin_id has a corresponding threshold on elevation_threshold dict
    keys_set = set(elevation_thresholds.keys())
    for subbasin_id in subbasin_ids:
        if subbasin_id not in keys_set:
            raise ValueError(
                f"Missing elevation threshold for subbasin ID {subbasin_id}"
            )

    for subbasin_id, threshold in elevation_thresholds.items():
        sub_mask = subbasins == subbasin_id
        flood_mask = (detrended_dem <= threshold) & sub_mask
        floor.data[flood_mask.data] = 1

    return floor


def find_transition_points(network_gdf, elevation, xs_params, transition_params):
    pass


def derive_reach_thresholds(points_gdf, detrended_dem, subbasins)::
    pass
