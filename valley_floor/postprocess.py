"""
This module provides functions to post-process a floor raster.
"""

from skimage.morphology import remove_small_holes
from skimage.morphology import label
import numpy as np
import xarray as xr

from .config import PostProcessParameters


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
    region_floor: xr.DataArray,
    flood_floor: xr.DataArray,
    channel_network: xr.DataArray,
    slope: xr.DataArray,
    params: PostProcessParameters,
) -> xr.DataArray:
    floor = (region_floor == 1) | (flood_floor == 1)

    floor.data[slope > params.max_slope] = 0
    floor = floor == 1

    if params.min_size > 0:
        floor = close_holes(floor, params.min_size)

    floor = remove_isolated_areas(floor, channel_network)

    floor.data[channel_network > 0] = 1
    floor = floor.astype(np.uint8)

    floor.data[np.isnan(slope.data)] = 255
    floor = floor.rio.write_nodata(255)
    floor = floor.rio.set_nodata(255)
    floor.attrs["_FillValue"] = 255
    return floor
