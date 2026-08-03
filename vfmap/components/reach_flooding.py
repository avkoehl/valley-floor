from rasterio.transform import rowcol
import numba
import numpy as np
import xarray as xr
import geopandas as gpd

from vfmap.utils.cross_sections import sample_cross_sections
from vfmap.utils.raster import calculate_slope
from vfmap.utils.routing import route_points_to_reach, flood_from_reaches
from vfmap.config import Parameters


def flood_reaches(
    channel_network: xr.DataArray,
    dem: xr.DataArray,
    flow_directions: xr.DataArray,
    params: Parameters,
) -> tuple[xr.DataArray, gpd.GeoDataFrame, dict]:
    xs_coords = sample_cross_sections(channel_network, dem, params)
    slope_raster = calculate_slope(dem, smooth_window=params.flood_slope_window).values
    slope_break_pts = _detect_slope_breaks(
        xs_coords,
        dem,
        slope_raster,
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
    dem: xr.DataArray,
    slope_raster: np.ndarray,
    steep_slope: float,
    min_elevation_gain: float,
) -> gpd.GeoDataFrame:
    """For every cross section, walk outward from the channel on each side
    looking for the first steep run (see `_find_slope_breaks_numba`) and
    return its point as that side's slope-break.

    Steepness is the DEM's own local slope (`slope_raster`, from
    `calculate_slope`) sampled at each cross-section point - direction-
    independent, unlike a naive elevation difference measured along the
    cross-section's own line, which underreads the true slope whenever that
    line isn't perpendicular to the wall.

    All cross sections are processed in one pass: rather than looping per
    `xs_id` in Python, every side of every cross section is sorted into one
    flat array (grouped by (xs_id, side), ordered by distance from the
    channel) and handed to a numba kernel that scans each group's run in a
    single pass.
    """
    empty = gpd.GeoDataFrame(
        columns=["xs_id", "side", "geometry", "elevation", "reach_id"],
        geometry="geometry",
        crs=xs_coords.crs,
    )
    if xs_coords.empty:
        return empty

    # The center point (distance == 0) belongs to both the left (<= 0) and
    # right (>= 0) walk, so it's duplicated into both halves below - matching
    # the original per-xs_id code, which ran the same two overlapping masks.
    left_mask = (xs_coords["distance"] <= 0).to_numpy()
    right_mask = (xs_coords["distance"] >= 0).to_numpy()

    xs_id = xs_coords["xs_id"].to_numpy()
    distance = xs_coords["distance"].to_numpy()
    elevation = xs_coords["interp_elevation"].to_numpy()
    point_id = xs_coords["point_id"].to_numpy()

    rows, cols = rowcol(dem.rio.transform(), xs_coords.geometry.x.values, xs_coords.geometry.y.values)
    rows = np.clip(np.asarray(rows), 0, slope_raster.shape[0] - 1)
    cols = np.clip(np.asarray(cols), 0, slope_raster.shape[1] - 1)
    point_slope = slope_raster[rows, cols]

    half_side = np.concatenate(
        [np.zeros(left_mask.sum(), dtype=np.int64), np.ones(right_mask.sum(), dtype=np.int64)]
    )
    h_xs_id = np.concatenate([xs_id[left_mask], xs_id[right_mask]])
    h_distance = np.concatenate([distance[left_mask], distance[right_mask]])
    h_abs_dist = np.concatenate([-distance[left_mask], distance[right_mask]])
    h_elevation = np.concatenate([elevation[left_mask], elevation[right_mask]])
    h_point_id = np.concatenate([point_id[left_mask], point_id[right_mask]])
    h_true_slope = np.concatenate([point_slope[left_mask], point_slope[right_mask]])

    # sort by (xs_id, half_side, abs_dist) so each (xs_id, side) group is
    # contiguous and ordered outward from the channel
    order = np.lexsort((h_abs_dist, half_side, h_xs_id))
    h_xs_id = h_xs_id[order]
    half_side = half_side[order]
    h_distance = h_distance[order]
    h_elevation = h_elevation[order]
    h_point_id = h_point_id[order]
    h_true_slope = h_true_slope[order]

    group_key = h_xs_id.astype(np.int64) * 2 + half_side
    group_start = np.flatnonzero(
        np.r_[True, group_key[1:] != group_key[:-1]]
    ).astype(np.int64)
    group_end = np.r_[group_start[1:], len(group_key)].astype(np.int64)

    found_point_id = _find_slope_breaks_numba(
        h_true_slope.astype(np.float64),
        h_elevation.astype(np.float64),
        h_distance.astype(np.float64),
        h_point_id.astype(np.int64),
        group_start,
        group_end,
        float(steep_slope),
        float(min_elevation_gain),
    )

    valid = found_point_id >= 0
    if not np.any(valid):
        return empty

    result_xs_id = h_xs_id[group_start][valid]
    result_side = np.where(half_side[group_start][valid] == 0, "left", "right")
    result_point_id = found_point_id[valid]

    point_lookup = xs_coords.set_index("point_id")
    rows = point_lookup.loc[result_point_id]
    return gpd.GeoDataFrame(
        {
            "xs_id": result_xs_id,
            "side": result_side,
            "geometry": rows["geometry"].to_numpy(),
            "elevation": rows["interp_elevation"].to_numpy(),
            "reach_id": rows["linestring_id"].to_numpy().astype(int),
        },
        geometry="geometry",
        crs=xs_coords.crs,
    )


@numba.njit(cache=True)
def _find_slope_breaks_numba(
    true_slope, elevation, distance, point_id, group_start, group_end,
    min_slope_deg, min_elevation_gain,
):
    """For each (xs_id, side) group - contiguous, sorted by distance from
    the channel ascending - find the first run of consecutive steep
    segments (DEM slope at the segment's start point >= `min_slope_deg`,
    and rising outward) whose flagged span gains more than
    `min_elevation_gain`, and return the point_id of that run's break point
    (-1 if none found).

    A segment i -> i+1 is "steep" if the DEM's own local slope at point i
    meets the threshold and elevation increases outward across it - using
    the DEM's slope directly (rather than rise-over-run measured along the
    cross-section's own, possibly oblique, line) avoids underreading the
    true steepness whenever the cross-section isn't perpendicular to the
    wall. A run is a maximal contiguous block of steep-flagged segments.
    The run's gain is measured between its first and last *flagged* point -
    i.e. it does not include the endpoint of the run's final segment, only
    where that segment starts. This mirrors the original implementation's
    `group.iloc[-1] - group.iloc[0]` over rows where `is_steep` is True,
    preserved here rather than "corrected" since it's existing, validated
    behavior.

    If the run starts exactly at the channel (`distance == 0`, i.e. the
    center point), its own point is skipped in favor of the run's second
    point, same as the original - a break can't be "at" the channel itself.
    """
    n_groups = len(group_start)
    found = np.full(n_groups, -1, dtype=np.int64)

    for g in range(n_groups):
        start = group_start[g]
        end = group_end[g]
        n = end - start
        if n < 2:
            continue

        run_start = -1
        for i in range(n - 1):
            idx = start + i
            rising = elevation[idx + 1] >= elevation[idx]
            is_steep = rising and (true_slope[idx] >= min_slope_deg)

            if is_steep:
                if run_start == -1:
                    run_start = i
            elif run_start != -1:
                last_flagged = i - 1
                gain = elevation[start + last_flagged] - elevation[start + run_start]
                if gain > min_elevation_gain:
                    if distance[start + run_start] == 0.0 and last_flagged > run_start:
                        found[g] = point_id[start + run_start + 1]
                    else:
                        found[g] = point_id[start + run_start]
                    break
                run_start = -1

        if found[g] == -1 and run_start != -1:
            last_flagged = n - 2
            gain = elevation[start + last_flagged] - elevation[start + run_start]
            if gain > min_elevation_gain:
                if distance[start + run_start] == 0.0 and last_flagged > run_start:
                    found[g] = point_id[start + run_start + 1]
                else:
                    found[g] = point_id[start + run_start]

    return found


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
