"""
Determine flood threshold for each stream segment based on wall points and detrended elevation data.
Output is a dictionary with stream segment IDs as keys and their respective flood thresholds as values.
To be used as an input in `valley_floor.flood`.

"""

import xarray as xr
import numpy as np


def determine_flood_threshold(
    wallpoints,
    subbasin_raster,
    detrended_elevation_raster,
    min_points=10,
    percentile=80,
    buffer=0.0,
    default_threshold=10,
):
    wallpoints["subbasin"] = subbasin_raster.sel(
        x=xr.DataArray(wallpoints["x_coord"]),
        y=xr.DataArray(wallpoints["y_coord"]),
        method="nearest",
    ).values

    wallpoints["detrended"] = detrended_elevation_raster.sel(
        x=xr.DataArray(wallpoints["x_coord"]),
        y=xr.DataArray(wallpoints["y_coord"]),
        method="nearest",
    ).values

    unique_subbasins = np.unique(subbasin_raster.data)
    unique_subbasins = unique_subbasins[np.isfinite(unique_subbasins)]

    flood_thresholds = {}
    for sid in unique_subbasins:
        subbasin_points = wallpoints[wallpoints["subbasin"] == sid]
        values = subbasin_points["detrended"].values
        values = values[np.isfinite(values)]

        # remove outliers
        if len(values) >= min_points:
            print(f"Processing subbasin {sid} with {len(values)} points.")
            print(f"Values before outlier removal: {values}")
            print(f"Mean: {np.mean(values)}, Std Dev: {np.std(values)}")
            values = values[np.abs(values - np.mean(values)) < 3 * np.std(values)]

        if len(values) < min_points:
            flood_threshold = default_threshold
        else:
            flood_threshold = np.percentile(values, percentile)
            flood_threshold = flood_threshold + buffer
        flood_thresholds[sid] = flood_threshold

    return flood_thresholds
