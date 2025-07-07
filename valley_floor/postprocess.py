"""
This module provides functions to post-process a floor raster.
"""

from skimage.morphology import remove_small_holes


def combine_floors(
    flood_extent_floor,
    low_slope_floor,
):
    return flood_extent_floor | low_slope_floor


def burnin_streams(
    floor,
    channel_network,
):
    return floor + (channel_network > 0)


def close_holes(
    floor,
    max_fill_area,
):
    filled = floor.copy(deep=True)
    num_cells = max_fill_area / floor.rio.resolution()[0] ** 2
    filled.data = remove_small_holes(filled.data, num_cells)
    return filled


def label_by_subbasin(
    floor,
    subbasins,
):
    return (floor > 0) * subbasins


def convert_to_polygon(
    floor,
):
    # TODO
    pass


def remove_boundary_slopes(floor, slope_threshold):
    # TODO
    pass


def process_floor(
    flood_extent_floor,
    low_slope_floor,
    channel_network,
    min_size=0,
    subbasins=None,
):
    floor = combine_floors(flood_extent_floor, low_slope_floor)

    floor = burnin_streams(floor, channel_network)

    if min_size > 0:
        floor = close_holes(floor, min_size)

    if subbasins is not None:
        floor = label_by_subbasin(floor, subbasins)

    # if return_polygon:
    #    floor = convert_to_polygon(floor)

    return floor
