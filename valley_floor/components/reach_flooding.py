import numpy as np
import xarray as xr
import geopandas as gpd

from valley_floor.config import Parameters


def flood_reaches(
    xs_coords: gpd.GeoDataFrame,
    detrended_dem: xr.DataArray,
    subbasins: xr.DataArray,
    params: Parameters,
) -> tuple[xr.DataArray, gpd.GeoDataFrame, dict]:
    slope_break_pts = detect_slope_breaks(
        xs_coords,
        params.flood_steep_slope,
        params.flood_min_elevation_gain,
    )
    reach_thresholds, slope_break_pts = derive_reach_thresholds(
        slope_break_pts,
        detrended_dem,
        subbasins,
        params.flood_default_elevation,
        params.flood_percentile,
        params.flood_min_points,
    )
    flooded_reaches = apply_flooding(detrended_dem, reach_thresholds, subbasins)
    return flooded_reaches, slope_break_pts, reach_thresholds


def apply_flooding(detrended_dem, elevation_thresholds, subbasins):
    floor = detrended_dem.copy(data=np.zeros_like(detrended_dem.data, dtype=np.uint8))

    for subbasin_id, threshold in elevation_thresholds.items():
        sub_mask = subbasins == subbasin_id
        flood_mask = (detrended_dem <= threshold) & sub_mask
        floor.data[flood_mask.data] = 1

    floor.data[np.isnan(detrended_dem.data)] = 255
    floor = floor.rio.set_nodata(255)
    floor = floor.rio.write_nodata(255, encoded=True)
    return floor


def detect_slope_breaks(xs_coords, steep_slope, min_elevation_gain):
    point_id_index = xs_coords.set_index("point_id")

    results_list = []
    for xs_id, xs in xs_coords.groupby("xs_id"):
        breaks = find_slope_breaks(xs, steep_slope, min_elevation_gain)
        for side in ["left", "right"]:
            found_id = breaks[side]
            if found_id is not None:
                matched_row = point_id_index.loc[found_id]
                results_list.append(
                    {
                        "xs_id": xs_id,
                        "side": side,
                        "geometry": matched_row.geometry.values[0],
                        "elevation": matched_row.interp_elevation.values[0],
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
    points_gdf, detrended_dem, subbasins, default_elevation, percentile, min_points
):
    def _mad_filter(data):
        data = data[np.isfinite(data)]
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return median + 3 * mad

    points_gdf["subbasin"] = subbasins.sel(
        x=xr.DataArray(points_gdf.geometry.x.values),
        y=xr.DataArray(points_gdf.geometry.y.values),
        method="nearest",
    )
    points_gdf["elevation"] = detrended_dem.sel(
        x=xr.DataArray(points_gdf.geometry.x.values),
        y=xr.DataArray(points_gdf.geometry.y.values),
        method="nearest",
    )

    reach_thresholds = {}
    points_gdf["is_outlier"] = False
    for subbasin_id, group in points_gdf.groupby("subbasin"):
        if np.isnan(subbasin_id) or subbasin_id == 0:
            continue
        if not group.empty:
            values = group["elevation"].values
            cutoff = _mad_filter(values)
            is_above = group["elevation"] > cutoff
            points_gdf.loc[is_above.index, "is_outlier"] = is_above
            values = values[values <= cutoff]
            values = values[np.isfinite(values)]
            threshold = (
                np.percentile(values, percentile)
                if len(values) >= min_points
                else default_elevation
            )
        else:
            threshold = default_elevation
        reach_thresholds[subbasin_id] = threshold

    for subbasin_id in np.unique(subbasins.values):
        if subbasin_id == 0 or np.isnan(subbasin_id):
            continue
        if subbasin_id not in reach_thresholds:
            reach_thresholds[int(subbasin_id)] = default_elevation

    return reach_thresholds, points_gdf


def find_slope_breaks(gdf, min_slope_degrees, min_elevation_gain):
    min_slope_ratio = np.tan(np.radians(min_slope_degrees))

    left_bank = gdf[gdf["distance"] <= 0].copy()
    right_bank = gdf[gdf["distance"] >= 0].copy()

    results = {}
    for side_name, df in [("left", left_bank), ("right", right_bank)]:
        if df.empty:
            results[side_name] = None
            continue

        df = df.copy()
        df["abs_dist"] = df["distance"].abs()
        df = df.sort_values("abs_dist").reset_index(drop=True)

        delta_z = df["interp_elevation"].diff().shift(-1)
        delta_x = df["abs_dist"].diff().shift(-1)
        slopes = (delta_z / delta_x).fillna(0)

        is_steep = slopes >= min_slope_ratio
        segment_ids = (is_steep != is_steep.shift()).cumsum()

        found_point = None
        steep_segments = df[is_steep].groupby(segment_ids)
        for seg_id in sorted(steep_segments.groups.keys()):
            group = steep_segments.get_group(seg_id)
            gain = (
                group["interp_elevation"].iloc[-1] - group["interp_elevation"].iloc[0]
            )
            if gain > min_elevation_gain:
                if group.iloc[0]["distance"] == 0:
                    found_point = (
                        group.iloc[1]["point_id"]
                        if len(group) > 1
                        else group.iloc[0]["point_id"]
                    )
                else:
                    found_point = group.iloc[0]["point_id"]
                break

        results[side_name] = found_point

    return results
