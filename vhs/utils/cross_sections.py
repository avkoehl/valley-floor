from typing import Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import LineString
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

    xs_points = []
    for xs_id, xs_linestring in xs_linestrings.groupby("xs_id"):
        for _, linestring in xs_linestring.iterrows():
            points = _points_along_linestring(
                linestring.geometry, point_interval, crs=xs_linestrings.crs
            )
            points["xs_id"] = xs_id
            for col in xs_linestring.columns:
                if col != "geometry":
                    points[col] = linestring[col]
            xs_points.append(points)

    return gpd.GeoDataFrame(
        pd.concat(xs_points), crs=xs_linestrings.crs, geometry="geometry"
    ).reset_index(drop=True)


def _points_along_linestring(linestring, interval, crs=None) -> gpd.GeoDataFrame:
    center = linestring.length / 2
    pos_dists = np.arange(center + interval, linestring.length + interval, interval)
    neg_dists = np.arange(center - interval, -interval, -interval)
    distances = np.concatenate([[center], pos_dists, neg_dists])
    sides = ["center"] + ["positive"] * len(pos_dists) + ["negative"] * len(neg_dists)

    valid_mask = (distances >= 0) & (distances <= linestring.length)
    points = [linestring.interpolate(d) for d in distances[valid_mask]]
    distances = distances[valid_mask] - center

    return gpd.GeoDataFrame(
        {
            "side": [s for s, v in zip(sides, valid_mask) if v],
            "geometry": points,
            "distance": distances,
        },
        crs=crs,
    )


def _create_xs_for_linestring(
    linestring, interval_distance, width, crs=None, smoothed=True
):
    if smoothed:
        angles, points = _perpendicular_angles_smoothed(linestring, interval_distance)
    else:
        angles, points = _perpendicular_angles(linestring, interval_distance)
    lines = [
        _xs_linestring(point, angle, width) for point, angle in zip(points, angles)
    ]
    series = gpd.GeoSeries(lines)
    if crs:
        series.crs = crs
    return series


def _perpendicular_angles(linestring, interval_distance):
    distances = np.arange(0, linestring.length + interval_distance, interval_distance)
    distances = distances[distances <= linestring.length]
    angles, points = [], []
    for dist in distances:
        point = linestring.interpolate(dist)
        left, right = _points_on_either_side(linestring, dist)
        angle = np.arctan2(right.y - left.y, right.x - left.x) + np.pi / 2
        angles.append(angle)
        points.append(point)
    return angles, points


def _perpendicular_angles_smoothed(linestring, interval_distance):
    smoothed = chaikin_smooth(taubin_smooth(linestring))
    angles, points = _perpendicular_angles(smoothed, interval_distance)
    new_points, new_angles = [], []
    for point, angle in zip(points, angles):
        line = _xs_linestring(point, angle, width=200)
        intersection = linestring.intersection(line)
        if intersection.geom_type == "Point":
            new_points.append(intersection)
            new_angles.append(angle)
        elif intersection.geom_type == "MultiPoint":
            geoms = list(intersection.geoms)
            closest = min(geoms, key=lambda p: point.distance(p))
            new_points.append(closest)
            new_angles.append(angle)
        else:
            new_points.append(linestring.interpolate(linestring.project(point)))
            new_angles.append(angle)
    return new_angles, new_points


def _points_on_either_side(linestring, distance, delta=1):
    left_delta = delta if distance - delta >= 0 else 0
    right_delta = delta if distance + delta <= linestring.length else linestring.length
    return linestring.interpolate(distance - left_delta), linestring.interpolate(
        distance + right_delta
    )


def _xs_linestring(point, angle, width) -> LineString:
    dx = width / 2 * np.cos(angle)
    dy = width / 2 * np.sin(angle)
    return LineString([(point.x - dx, point.y - dy), (point.x + dx, point.y + dy)])
