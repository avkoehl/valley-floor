import numpy as np
import xarray as xr
from skimage.morphology import remove_small_holes

from vfmap.config import Parameters
from vfmap.utils import remove_isolated_areas
from vfmap.utils import calculate_slope


def postprocess(
    floor: xr.DataArray,
    channel_network: xr.DataArray,
    dem: xr.DataArray,
    params: Parameters,
) -> xr.DataArray:
    slope = calculate_slope(dem)

    floor = floor.copy(deep=True)
    floor.data[slope.data > params.max_slope] = 0
    floor = floor == 1

    if params.min_hole_size > 0:
        floor = _close_holes(floor, params.min_hole_size)

    floor = remove_isolated_areas(floor, channel_network)
    floor.data[channel_network.data > 0] = 1
    floor = floor.astype(np.uint8)
    floor.data[np.isnan(dem.data)] = 255
    floor = floor.rio.write_nodata(255)
    floor = floor.rio.set_nodata(255)
    floor.attrs["_FillValue"] = 255
    return floor


def _close_holes(floor: xr.DataArray, min_hole_size: float) -> xr.DataArray:
    resolution = floor.rio.resolution()[0]
    num_cells = int(min_hole_size / resolution**2)
    filled = floor.copy(deep=True)
    filled.data = remove_small_holes(filled.data, max_size=num_cells)
    return filled
