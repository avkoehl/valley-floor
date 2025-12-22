"""
This module provides functions to post-process a floor raster.
"""

from skimage.morphology import remove_small_holes
from skimage.morphology import label
import numpy as np


def remove_isolated_areas(binary, flowpaths):
    """
    Keep only connected components in a binary raster that intersect with flowpaths.
    """
    fp = flowpaths > 0
    combined = fp + binary
    combined = combined > 0

    con = label(combined, connectivity=2)
    con = con.astype(np.float64)

    values = np.unique(con[flowpaths > 0])
    values = values[np.isfinite(values)]

    result = flowpaths.copy()
    result.data = con
    result = result.where(np.isin(con, values))
    result = result > 0
    return result


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
    # floor is raster with True or False values
    # subbasins is a raster with integer labels for each subbasin
    # want labeled to be a raster with subbasin labels where floor is True
    # though there are some 'floors' that don't intersect with any subbasin, they should be labeled with the max subbasin + 1
    labels = floor.copy(data=np.zeros_like(floor.data))
    floor_mask = floor.data > 0
    labels.data[floor_mask] = subbasins.data[floor_mask]

    orphaned = floor_mask & (labels.data == 0)
    max_label = np.nanmax(subbasins.data) + 1
    labels.data[orphaned] = max_label
    return labels


def process_floor(
    flood_extent_floor,
    low_slope_floor,
    channel_network,
    min_size=0.0,
    subbasins=None,
):
    floor = combine_floors(flood_extent_floor, low_slope_floor)

    floor = burnin_streams(floor, channel_network)
    floor = remove_isolated_areas(floor, channel_network)

    if min_size > 0:
        floor = close_holes(floor, min_size)

    if subbasins is not None:
        floor = label_by_subbasin(floor, subbasins)

    floor = floor > 0
    floor = floor.astype(np.uint8)
    return floor
