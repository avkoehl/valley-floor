"""
Analyze cross sections centered on river segments to identify the transition points where valley floor meets steep sustained slopes indicating valley wall boundary. Steep slopes are defined by a minimum elevation gain over a continuous segment of steep slope. Continuous segments are defined by sequential points with slope greater than a specified minimum slope.


`find_wallpoints` is a helper function that encapsulates the whole workflow - converts cross section linestrings into profiles and then processes each profile to find the transition points.
See `streamtools.profile` for details on how profiles are generated from cross section linestrings.
"""

import xarray as xr
import pandas as pd


def find_wallpoints(
    xs_coords,
    elevation_raster,
    slope_raster,
    min_slope,
    elevation_threshold,
):
    """
    See `streamtools.xs` and `streamtools.profile` for details on how to generate cross sections from a river network.

    Returns a DataFrame with coordinates of identified wall locations.
    """
    wall_points = process_profiles(
        xs_coords, elevation_raster, slope_raster, min_slope, elevation_threshold
    )
    return wall_points


def process_profiles(xs_coords, elevation, slope, min_slope, elevation_threshold):
    """
    Processes cross section profiles to identify wall locations based on elevation and slope data.
    Returns a DataFrame with the coordinates of the identified wall locations.
    """
    xs_coords["x_coord"] = xs_coords.geometry.x
    xs_coords["y_coord"] = xs_coords.geometry.y

    xs_coords["elevation"] = elevation.sel(
        x=xr.DataArray(xs_coords["x_coord"]),
        y=xr.DataArray(xs_coords["y_coord"]),
        method="nearest",
    ).values

    xs_coords["slope"] = slope.sel(
        x=xr.DataArray(xs_coords["x_coord"]),
        y=xr.DataArray(xs_coords["y_coord"]),
        method="nearest",
    ).values

    wall_locs = []
    for (xs_id, side), half_profile in xs_coords.groupby(["xs_id", "side"]):
        if side == "center":
            continue

        # otherwise,  add center point and sort by absolute distance from center
        # in ascending order of distance
        center = xs_coords.loc[
            (xs_coords["xs_id"] == xs_id) & (xs_coords["side"] == "center")
        ]
        half_profile = pd.concat([center, half_profile])
        half_profile = half_profile.sort_values("distance", key=abs)

        elevations = half_profile["elevation"].values
        slopes = half_profile["slope"].values

        integer_pos = find_transition(
            elevations, slopes, elevation_threshold, min_slope=min_slope
        )
        if integer_pos is not None:
            wall_locs.append(half_profile.index[integer_pos])

    if not wall_locs:
        return None
    else:
        results = xs_coords.loc[wall_locs]
        results = results.drop_duplicates(subset=["x_coord", "y_coord"])
        return results


def find_transition(elevations, slopes, height_threshold, min_slope):
    steep = slopes >= min_slope

    for i in range(len(steep)):
        if steep[i]:
            # Find end of this steep segment
            j = i
            while j < len(steep) and steep[j]:
                j += 1

            # Check elevation gain over this segment
            delta_h = elevations[j - 1] - elevations[i]
            if delta_h > height_threshold:
                return i

            i = j - 1  # Skip to end of this segment

    return None
