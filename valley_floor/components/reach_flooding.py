import numpy as np
import xarray as xr
import geopandas as gpd

from streamkit.network import network_cross_sections, sample_cross_sections


def flood_reaches(
    network_gdf,
    elevation,
    detrended_dem,
    subbasins,
    xs_params,
    transition_params,
    threshold_params,
):
    xs_coords = generate_cross_sections(
        network_gdf,
        elevation,
        xs_params["interval_distance"],
        xs_params["length"],
        xs_params["point_spacing"],
    )
    slope_break_pts = detect_slope_breaks(
        xs_coords,
        transition_params["steep_slope"],
        transition_params["min_elevation_gain"],
    )
    reach_thresholds = derive_reach_thresholds(
        slope_break_pts,
        detrended_dem,
        subbasins,
        threshold_params["default_elevation"],
        threshold_params["percentile"],
    )
    flooded_reaches = apply_flooding(detrended_dem, reach_thresholds, subbasins)
    return {
        "flooded_reaches": flooded_reaches,
        "slope_break_pts": slope_break_pts,
        "reach_thresholds": reach_thresholds,
    }


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

    # where dem is nan, set floor to 255
    floor.data[np.isnan(detrended_dem.data)] = 255
    floor = floor.rio.set_nodata(255)
    floor = floor.rio.write_nodata(255)
    floor.attrs["_FillValue"] = 255

    return floor


def generate_cross_sections(
    network_gdf, elevation, interval_distance, length, point_spacing
):
    # create cross section lines
    xs_lines = network_cross_sections(
        network_gdf.geometry,
        interval_distance,
        length,
        smoothed=True,
    )
    xs_coords = sample_cross_sections(xs_lines, point_spacing)
    xs_coords["point_id"] = np.arange(len(xs_coords))
    xs_coords["interpolated_elevation"] = elevation.interp(
        x=xr.DataArray(xs_coords["geometry"].x.values),
        y=xr.DataArray(xs_coords["geometry"].y.values),
        method="linear",
    ).values
    return xs_coords


def detect_slope_breaks(xs_coords, steep_slope, min_elevation_gain):
    results_list = []
    for xs_id, xs in xs_coords.groupby("xs_id"):
        breaks = find_slope_breaks(xs, steep_slope, min_elevation_gain)
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
            results_list, geometry="geometry", crs=xs_coords.crs
        )
    else:
        points_gdf = gpd.GeoDataFrame(
            columns=["xs_id", "side", "geometry", "elevation"],
            geometry="geometry",
            crs=xs_coords.crs,
        )
    return points_gdf


def derive_reach_thresholds(
    points_gdf, detrended_dem, subbasins, default_elevation, percentile
):
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
    for subbasin_id, group in np.unique(subbasins.values):
        if np.isnan(subbasin_id) or subbasin_id == 0:
            continue
        if not group.empty:
            values = group["elevation"].values
            values = values[np.isfinite(values)]
            if len(values) == 0:
                threshold = default_elevation
            else:
                threshold = np.percentile(values, percentile)
        else:
            threshold = default_elevation
        reach_thresholds[subbasin_id] = threshold
    return reach_thresholds


def find_slope_breaks(gdf, min_slope_degrees, min_elevation_gain):
    """
    Identifies the start of the first sustained steep segment on both sides
    of a cross section.
    """
    min_slope_ratio = np.tan(np.radians(min_slope_degrees))

    left_bank = gdf[gdf["distance"] <= 0].copy()
    right_bank = gdf[gdf["distance"] >= 0].copy()

    results = {}

    for side_name, df in [("left", left_bank), ("right", right_bank)]:
        if df.empty:
            results[side_name] = None
            continue

        # Sort by absolute distance, reset index for clean positional access
        df = df.copy()
        df["abs_dist"] = df["distance"].abs()
        df = df.sort_values("abs_dist").reset_index(drop=True)

        # Calculate slope to next point
        delta_z = (
            df["interp_elevation"].diff().shift(-1)
        )  # diff with next, not previous
        delta_x = df["abs_dist"].diff().shift(-1)
        slopes = (delta_z / delta_x).fillna(0)

        # Identify steep segments
        is_steep = slopes >= min_slope_ratio
        segment_ids = (is_steep != is_steep.shift()).cumsum()

        # Find first segment with sufficient elevation gain
        found_point = None
        steep_segments = df[is_steep].groupby(segment_ids)

        for seg_id in sorted(steep_segments.groups.keys()):
            group = steep_segments.get_group(seg_id)
            # Elevation gain within the steep segment
            gain = (
                group["interp_elevation"].iloc[-1] - group["interp_elevation"].iloc[0]
            )

            if gain > min_elevation_gain:
                found_point = group.iloc[0]["point_id"]
                break

        results[side_name] = found_point

    return results
