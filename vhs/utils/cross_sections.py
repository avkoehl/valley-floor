from typing import Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import xarray as xr
from shapelysmooth import chaikin_smooth, taubin_smooth

from vhs.config import Parameters
from vhs.utils.raster import raster_network_to_vector


# --- public entry point ---


def sample_cross_sections(
    channel_network: xr.DataArray,
    dem: xr.DataArray,
    params: Parameters,
) -> gpd.GeoDataFrame:
    network_gdf = raster_network_to_vector(channel_network)
    xs_lines = _create_cross_sections(
        network_gdf.geometry,
        params.xs_interval_distance,
        params.xs_length,
        linestring_ids=network_gdf["stream_id"],
    )
    xs_points = _cross_sections_to_points(xs_lines, params.xs_point_spacing)

    x = xr.DataArray(xs_points.geometry.x.values, dims="points")
    y = xr.DataArray(xs_points.geometry.y.values, dims="points")
    xs_points["interp_elevation"] = dem.sel(x=x, y=y, method="nearest").values
    xs_points = xs_points.dropna(subset=["interp_elevation"]).reset_index(drop=True)
    xs_points["point_id"] = np.arange(1, len(xs_points) + 1)

    return xs_points


# --- cross section generation ---


def _create_cross_sections(
    linestrings: gpd.GeoSeries,
    interval_distance: float,
    width: float,
    linestring_ids: Optional[Sequence] = None,
    smoothed: bool = True,
) -> gpd.GeoDataFrame:
    if linestring_ids is None:
        linestring_ids = linestrings.index

    xsections = []
    for cid, linestring in zip(linestring_ids, linestrings):
        channel_xsections = _create_xs_for_linestring(
            linestring, interval_distance, width, crs=linestrings.crs, smoothed=smoothed
        )
        channel_xsections = gpd.GeoDataFrame(
            geometry=channel_xsections, crs=linestrings.crs
        )
        channel_xsections["linestring_id"] = cid
        xsections.append(channel_xsections)

    xsections = pd.concat(xsections)
    xsections["xs_id"] = np.arange(1, len(xsections) + 1)
    return xsections


def _cross_sections_to_points(
    xs_linestrings: gpd.GeoDataFrame,
    point_interval: float,
) -> gpd.GeoDataFrame:
    if "xs_id" not in xs_linestrings.columns:
        xs_linestrings["xs_id"] = np.arange(1, len(xs_linestrings) + 1)

    geoms = xs_linestrings.geometry.to_numpy()
    lengths = shapely.length(geoms)

    # Each line's own (float-jittered) length determines its own distance
    # grid - as in the original per-line derivation - so this can't be
    # collapsed onto one shared grid without risking an off-by-one point at
    # the ends when two "identical" lines differ by a ULP or two. The grid
    # math itself is cheap numpy, so it stays a plain loop; only the
    # expensive part (interpolating points on the geometry) is batched into
    # one vectorized shapely call below.
    line_index_chunks = []
    distance_chunks = []
    side_chunks = []
    for i, length in enumerate(lengths):
        distances, sides = _cross_section_distance_grid(length, point_interval)
        line_index_chunks.append(np.full(len(distances), i))
        distance_chunks.append(distances)
        side_chunks.append(sides)

    line_index = np.concatenate(line_index_chunks) if line_index_chunks else np.empty(0, dtype=int)
    all_distances = np.concatenate(distance_chunks) if distance_chunks else np.empty(0)
    all_sides = np.concatenate(side_chunks) if side_chunks else np.empty(0, dtype=object)
    rel_distances = all_distances - lengths[line_index] / 2

    points = shapely.line_interpolate_point(geoms[line_index], all_distances)
    xy = shapely.get_coordinates(points)

    other_cols = [c for c in xs_linestrings.columns if c != "geometry"]
    data = {
        "side": all_sides,
        "geometry": gpd.points_from_xy(xy[:, 0], xy[:, 1]) if len(xy) else [],
        "distance": rel_distances,
    }
    for c in other_cols:
        data[c] = xs_linestrings[c].to_numpy()[line_index]

    xs_points = pd.DataFrame(data)
    return gpd.GeoDataFrame(xs_points, crs=xs_linestrings.crs, geometry="geometry")


def _cross_section_distance_grid(length, interval):
    """Distances (and their side labels) at which a cross section of this
    `length` is sampled, matching the original per-line derivation: a center
    point plus points spaced `interval` apart out to each end."""
    center = length / 2
    pos_dists = np.arange(center + interval, length + interval, interval)
    neg_dists = np.arange(center - interval, -interval, -interval)
    distances = np.concatenate([[center], pos_dists, neg_dists])
    sides = np.array(
        ["center"] + ["positive"] * len(pos_dists) + ["negative"] * len(neg_dists)
    )

    valid_mask = (distances >= 0) & (distances <= length)
    return distances[valid_mask], sides[valid_mask]


def _create_xs_for_linestring(
    linestring, interval_distance, width, crs=None, smoothed=True
):
    if smoothed:
        angles, points = _perpendicular_angles_smoothed(linestring, interval_distance)
    else:
        angles, points = _perpendicular_angles(linestring, interval_distance)
    lines = _xs_linestrings(points, angles, width)
    series = gpd.GeoSeries(lines)
    if crs:
        series.crs = crs
    return series


def _perpendicular_angles(linestring, interval_distance):
    distances = np.arange(0, linestring.length + interval_distance, interval_distance)
    distances = distances[distances <= linestring.length]
    if len(distances) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=object)

    delta = 1.0
    left_dist = np.where(distances - delta >= 0, distances - delta, 0.0)
    right_dist = np.where(
        distances + delta <= linestring.length, distances + delta, linestring.length
    )

    points = shapely.line_interpolate_point(linestring, distances)
    left_xy = shapely.get_coordinates(shapely.line_interpolate_point(linestring, left_dist))
    right_xy = shapely.get_coordinates(shapely.line_interpolate_point(linestring, right_dist))

    angles = (
        np.arctan2(right_xy[:, 1] - left_xy[:, 1], right_xy[:, 0] - left_xy[:, 0])
        + np.pi / 2
    )
    return angles, points


def _perpendicular_angles_smoothed(linestring, interval_distance):
    smoothed = chaikin_smooth(taubin_smooth(linestring))
    angles, points = _perpendicular_angles(smoothed, interval_distance)
    if len(points) == 0:
        return angles, points

    # `angles` is returned unchanged below in every branch of the original
    # per-point logic; only the sampled `points` themselves get snapped onto
    # the (unsmoothed) `linestring`.
    candidate_lines = _xs_linestrings(points, angles, width=200)
    intersections = shapely.intersection(candidate_lines, linestring)
    type_ids = shapely.get_type_id(intersections)

    new_points = np.array(points, dtype=object)

    is_point = type_ids == shapely.GeometryType.POINT
    new_points[is_point] = intersections[is_point]

    is_multi = type_ids == shapely.GeometryType.MULTIPOINT
    for i in np.flatnonzero(is_multi):
        geoms = list(intersections[i].geoms)
        new_points[i] = min(geoms, key=lambda p: points[i].distance(p))

    other = ~is_point & ~is_multi
    if np.any(other):
        proj = shapely.line_locate_point(linestring, np.array(points, dtype=object)[other])
        new_points[other] = shapely.line_interpolate_point(linestring, proj)

    return angles, new_points


def _xs_linestrings(points, angles, width) -> np.ndarray:
    points = np.asarray(points, dtype=object)
    angles = np.asarray(angles, dtype=float)
    xy = shapely.get_coordinates(points)
    dx = width / 2 * np.cos(angles)
    dy = width / 2 * np.sin(angles)
    starts = xy - np.column_stack([dx, dy])
    ends = xy + np.column_stack([dx, dy])
    coords = np.stack([starts, ends], axis=1)
    return shapely.linestrings(coords)
