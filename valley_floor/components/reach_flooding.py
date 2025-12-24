import numpy as np
import xarray as xr
import geopandas as gpd

from streamkit.network import network_cross_sections, sample_cross_sections

from .slope_breaks import find_slope_breaks


def flood_reaches(
    network_gdf,
    detrended_dem,
    subbasins,
    xs_params,
    transition_params,
    threshold_params,
):
    transitions = find_transition_points(
        network_gdf, detrended_dem, xs_params, transition_params
    )
    reach_thresholds = derive_reach_thresholds(
        transitions, detrended_dem, subbasins, threshold_params
    )
    flooded_reaches = apply_flooding(detrended_dem, reach_thresholds, subbasins)
    return {"flooded_reaches": flooded_reaches, "reach_thresholds": reach_thresholds}


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
    # create cross section lines
    xs_lines = network_cross_sections(
        network_gdf.geometry,
        xs_params["interval_distance"],
        xs_params["length"],
        smoothed=True,
    )
    xs_coords = sample_cross_sections(xs_lines, xs_params["point_spacing"])
    xs_coords["point_id"] = np.arange(len(xs_coords))
    xs_coords["interpolated_elevation"] = elevation.interp(
        x=xr.DataArray(xs_coords["geometry"].x.values),
        y=xr.DataArray(xs_coords["geometry"].y.values),
        method="linear",
    ).values

    results_list = []
    for xs_id, xs in xs_coords.groupby("xs_id"):
        breaks = find_slope_breaks(
            xs, transition_params["min_slope"], transition_params["min_elevation_gain"]
        )
        for side in ["left", "right"]:
            found_id = breaks[side]
            if found_id is not None:
                matched_row = xs_coords[xs_coords["point_id"] == found_id]
                results_list.append(
                    {
                        "xs_id": xs_id,
                        "side": side,
                        "geometry": matched_row.geometry.values[0],
                        "elevation": matched_row.interpolated_elevation.values[0],
                    }
                )
    if results_list:
        points_gdf = gpd.GeoDataFrame(
            results_list, geometry="geometry", crs=network_gdf.crs
        )
    else:
        points_gdf = gpd.GeoDataFrame(
            columns=["xs_id", "side", "geometry", "elevation"],
            geometry="geometry",
            crs=network_gdf.crs,
        )
    return points_gdf


def derive_reach_thresholds(points_gdf, detrended_dem, subbasins, threshold_params):
    points_gdf["subbasin"] = subbasins.sel(
        x=xr.DataArarray(points_gdf.geometry.x.values),
        y=xr.DataArray(points_gdf.geometry.y.values),
        method="nearest",
    )
    points_gdf["elevation"] = detrended_dem.sel(
        x=xr.DataArarray(points_gdf.geometry.x.values),
        y=xr.DataArray(points_gdf.geometry.y.values),
        method="nearest",
    )

    reach_thresholds = {}
    for subbasin_id, group in points_gdf.groupby("subbasin"):
        if not group.empty:
            values = group["elevation"].values
            values = values[np.isfinite(values)]
            if len(values) == 0:
                threshold = threshold_params["default_threshold"]
            else:
                threshold = np.percentile(
                    values, threshold_params["percentile_threshold"]
                )
        else:
            threshold = threshold_params["default_threshold"]
        reach_thresholds[subbasin_id] = threshold
    return reach_thresholds
