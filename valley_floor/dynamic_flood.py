"""
Determine flood threshold for each stream segment based on wall points and detrended elevation data.
Output is a dictionary with stream segment IDs as keys and their respective flood thresholds as values.
To be used as an input in `valley_floor.flood`.

"""

import xarray as xr
import numpy as np


def dynamic_flood_thresholds(
    xscoords,
    dem,
    slope,
    hand,
    subbasins,
    min_slope=10.0,
    elevation_threshold=10.0,
    min_points=10,
    percentile=80,
    buffer=0.0,
    default_threshold=10,
    return_wallpoints=False,
):
    labeled_xsections = _label_wallpoints(
        xscoords, dem, slope, min_slope, elevation_threshold
    )
    wallpoints = labeled_xsections[labeled_xsections["is_wallpoint"]].copy()
    thresholds = _determine_flood_threshold(
        wallpoints,
        subbasins,
        hand,
        min_points,
        percentile,
        buffer,
        default_threshold,
    )
    if return_wallpoints:
        return thresholds, wallpoints
    return thresholds


def _determine_flood_threshold(
    wallpoints,
    subbasin_raster,
    detrended_elevation_raster,
    min_points,
    percentile,
    buffer,
    default_threshold,
):
    x_coords = xr.DataArray(wallpoints["geometry"].x.values)
    y_coords = xr.DataArray(wallpoints["geometry"].y.values)

    wallpoints["subbasin"] = subbasin_raster.sel(
        x=x_coords,
        y=y_coords,
        method="nearest",
    ).values

    wallpoints["detrended"] = detrended_elevation_raster.sel(
        x=x_coords,
        y=y_coords,
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
            values = values[np.abs(values - np.mean(values)) < 3 * np.std(values)]

        if len(values) < min_points:
            flood_threshold = default_threshold
        else:
            flood_threshold = np.percentile(values, percentile)
            flood_threshold = flood_threshold + buffer
        flood_thresholds[sid] = flood_threshold

    return flood_thresholds


def _label_wallpoints(
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
    cols = ["xs_id", "side", "geometry", "distance"]
    # confirm cols exist in xs_coords
    for col in cols:
        if col not in xs_coords.columns:
            raise ValueError(f"Column '{col}' not found in xs_coords DataFrame.")

    coord_copy = xs_coords.copy(deep=True)
    coord_copy["point_id"] = np.arange(len(coord_copy))

    # add elevation and slope data to xs_coords
    xcoords = xr.DataArray(coord_copy["geometry"].x.values)
    ycoords = xr.DataArray(coord_copy["geometry"].y.values)
    coord_copy["interp_elevation"] = elevation_raster.interp(
        x=xcoords, y=ycoords, method="linear"
    ).values
    coord_copy["raster_slope"] = slope_raster.sel(
        x=xcoords, y=ycoords, method="nearest"
    ).values

    # Iterate through each cross section to find wall points
    wall_points_all = []
    for _, points in coord_copy.groupby("xs_id"):
        # confirm atleast 5 points in the cross section
        if len(points) < 5:
            continue

        wall_points = _find_wallpoints_in_cross_section(
            points, min_slope, elevation_threshold
        )
        if wall_points is not None:
            for point_id in wall_points:
                wall_points_all.append(point_id)

    xs_coords["is_wallpoint"] = False
    xs_coords.loc[coord_copy["point_id"].isin(wall_points_all), "is_wallpoint"] = True
    return xs_coords


def _find_wallpoints_in_cross_section(points, min_slope, height_threshold):
    # first check if its cross section along a steep hillslope (cascade)
    if points.loc[points["distance"].abs() < 30, "raster_slope"].mean() > min_slope:
        neg_wp_id = points.loc[points["side"] == "negative", "point_id"].iloc[0]
        pos_wp_id = points.loc[points["side"] == "positive", "point_id"].iloc[0]
        return [neg_wp_id, pos_wp_id]
    else:
        wall_points = []
        negative_points = points[
            (points["side"] == "negative") | (points["side"] == "center")
        ].sort_values(by="distance", key=abs, ascending=True)
        positive_points = points[
            (points["side"] == "positive") | (points["side"] == "center")
        ].sort_values(by="distance", ascending=True)

        # Find wall points in negative side
        neg_wallpoint = _find_wallpoint_half_profile(
            negative_points, min_slope, height_threshold
        )
        pos_wallpoint = _find_wallpoint_half_profile(
            positive_points, min_slope, height_threshold
        )
        if neg_wallpoint:
            wall_points.append(neg_wallpoint)
        if pos_wallpoint:
            wall_points.append(pos_wallpoint)

        if len(wall_points) == 0:
            return None
        else:
            return wall_points


def _find_wallpoint_half_profile(points, min_slope_degrees, height_threshold):
    elevations = points["interp_elevation"].values
    lateral_slopes = np.gradient(elevations, points["distance"].values)
    lateral_slopes = np.abs(lateral_slopes)
    # Convert min_slope from degrees to a ratio
    min_slope = np.tan(np.radians(min_slope_degrees))
    steep = lateral_slopes >= min_slope
    for i in range(len(steep)):
        if steep[i]:
            # Find end of this steep segment
            j = i
            while j < len(steep) and steep[j]:
                j += 1
            # Check elevation gain over this segment
            delta_h = elevations[j - 1] - elevations[i]
            if delta_h > height_threshold:
                return points.iloc[i][
                    "point_id"
                ]  # Return the first point of the steep segment
            i = j - 1  # Skip to end of this segment
    return None
