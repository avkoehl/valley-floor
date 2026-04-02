import numpy as np
import xarray as xr
from xrspatial import slope as compute_slope

from .config import PostprocessingParameters
from .utils import close_holes, remove_isolated_areas


def postprocess(
    region_floor: xr.DataArray,
    flood_floor: xr.DataArray,
    channel_network: xr.DataArray,
    dem: xr.DataArray,
    params: PostprocessingParameters = PostprocessingParameters(),
) -> xr.DataArray:
    slope = compute_slope(dem)

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
