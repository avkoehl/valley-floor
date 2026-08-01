from rasterio.transform import rowcol
import numpy as np
import xarray as xr
import geopandas as gpd

from vhs.utils.cross_sections import sample_cross_sections
from vhs.utils.routing import route_points_to_reach, flood_from_reaches
from vhs.config import Parameters


def flood_reaches(
    channel_network: xr.DataArray,
    dem: xr.DataArray,
    flow_directions: xr.DataArray,
    params: Parameters,
) -> tuple[xr.DataArray, gpd.GeoDataFrame, dict]:
    xs_coords = sample_cross_sections(channel_network, dem, params)
    slope_break_pts = _detect_slope_breaks(
        xs_coords,
        params.flood_steep_slope,
        params.flood_min_elevation_gain,
    )
    reach_thresholds, slope_break_pts = _derive_reach_thresholds(
        slope_break_pts,
        channel_network,
        dem,
        flow_directions,
        params.flood_default_hand,
        params.flood_percentile,
        params.flood_min_points,
    )
    flood_floor = flood_from_reaches(channel_network, flow_directions, dem, reach_thresholds)
    return flood_floor, slope_break_pts, reach_thresholds


def _detect_slope_breaks(
    xs_coords: gpd.GeoDataFrame,
    steep_slope: float,
    min_elevation_gain: float,
) -> gpd.GeoDataFrame:
    point_id_index = xs_coords.set_index("point_id")
    results = []
    for xs_id, xs in xs_coords.groupby("xs_id"):
        breaks = _find_slope_breaks(xs, steep_slope, min_elevation_gain)
        for side in ["left", "right"]:
            found_id = breaks[side]
            if found_id is not None:
                row = point_id_index.loc[found_id]
                results.append(
                    {
                        "xs_id": xs_id,
                        "side": side,
                        "geometry": row.geometry,
                        "elevation": row.interp_elevation,
                        "reach_id": int(row.linestring_id),
                    }
                )
    if results:
        return gpd.GeoDataFrame(results, geometry="geometry", crs=xs_coords.crs)
    return gpd.GeoDataFrame(
        columns=["xs_id", "side", "geometry", "elevation", "reach_id"],
        geometry="geometry",
        crs=xs_coords.crs,
    )


def _derive_reach_thresholds(
    slope_break_pts: gpd.GeoDataFrame,
    channel_network: xr.DataArray,
    dem: xr.DataArray,
    flow_directions: xr.DataArray,
    default_hand: float,
    percentile: float,
    min_points: int,
) -> tuple[dict, gpd.GeoDataFrame]:
    def _mad_cutoff(values):
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return median + 3 * mad

    pts = slope_break_pts.copy()
    pts["gain"] = np.nan
    pts["routed"] = False
    if not pts.empty:
        # Each breakpoint's own reach is already known from the cross
        # section it came from (reach_id); route it back to that reach
        # rather than sampling a precomputed HAND raster, since a
        # breakpoint can sit closer to a different reach's channel than to
        # its own.
        rows, cols = rowcol(
            dem.rio.transform(), pts.geometry.x.values, pts.geometry.y.values
        )
        gains, valid = route_points_to_reach(
            np.asarray(rows),
            np.asarray(cols),
            pts["reach_id"].values,
            dem,
            channel_network,
            flow_directions,
        )
        pts["gain"] = gains
        pts["routed"] = valid

    reach_thresholds = {}
    pts["is_outlier"] = False
    if not pts.empty:
        for reach_id, group in pts[pts["routed"]].groupby("reach_id"):
            values = group["gain"].values
            values = values[np.isfinite(values)]
            if len(values) >= min_points:
                cutoff = _mad_cutoff(values)
                is_outlier = group["gain"] > cutoff
                pts.loc[is_outlier.index[is_outlier], "is_outlier"] = True
                values = values[values <= cutoff]
                threshold = (
                    np.percentile(values, percentile)
                    if len(values) >= min_points
                    else default_hand
                )
            else:
                threshold = default_hand
            reach_thresholds[int(reach_id)] = threshold

    # only assign default threshold to reaches actually present in the network
    active_reach_ids = set(
        int(v) for v in np.unique(channel_network.values) if v != 0 and np.isfinite(v)
    )
    for reach_id in active_reach_ids:
        if reach_id not in reach_thresholds:
            reach_thresholds[reach_id] = default_hand

    return reach_thresholds, pts


def _find_slope_breaks(
    gdf: gpd.GeoDataFrame,
    min_slope_degrees: float,
    min_elevation_gain: float,
) -> dict:
    min_slope_ratio = np.tan(np.radians(min_slope_degrees))
    results = {}
    for side_name, mask in [
        ("left", gdf["distance"] <= 0),
        ("right", gdf["distance"] >= 0),
    ]:
        df = gdf[mask].copy()
        if df.empty:
            results[side_name] = None
            continue
        df["abs_dist"] = df["distance"].abs()
        df = df.sort_values("abs_dist").reset_index(drop=True)

        delta_z = df["interp_elevation"].diff().shift(-1)
        delta_x = df["abs_dist"].diff().shift(-1)
        slopes = (delta_z / delta_x).fillna(0)
        is_steep = slopes >= min_slope_ratio
        segment_ids = (is_steep != is_steep.shift()).cumsum()

        found_point = None
        for seg_id, group in df[is_steep].groupby(segment_ids[is_steep]):
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
