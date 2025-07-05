"""
This module provides functions to post-process a floor raster.
"""


def combine_floors(
    flood_extent_floor,
    low_slope_floor,
):
    pass


def burnin_streams(
    floor,
    channel_network,
):
    pass


def close_holes(
    floor,
    min_size,
):
    pass


def label_by_subbasin(
    floor,
    subbasins,
):
    pass


def convert_to_polygon(
    floor,
):
    pass


def remove_boundary_slopes(floor, slope_threshold):
    pass


def process_floor(
    flood_extent_floor,
    low_slope_floor,
    channel_network,
    slope_threshold=None,
    min_size=0,
    subbasins=None,
    return_polygon=False,
):
    floor = combine_floors(flood_extent_floor, low_slope_floor)

    if slope_threshold is not None:
        floor = remove_boundary_slopes(floor, slope_threshold)

    floor = burnin_streams(floor, channel_network)

    if min_size > 0:
        floor = close_holes(floor, min_size)

    if subbasins is not None:
        floor = label_by_subbasin(floor, subbasins)

    if return_polygon:
        floor = convert_to_polygon(floor)

    return floor
