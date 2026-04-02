import numpy as np
from skimage.morphology import isotropic_dilation
from xrspatial import slope as compute_slope
import xarray as xr

from valley_floor.config import Parameters
from valley_floor.utils import remove_isolated_areas
from valley_floor.utils import smooth_raster


def grow_region(
    elevation: xr.DataArray,
    channel_network: xr.DataArray,
    params: Parameters,
) -> xr.DataArray:
    smoothed = smooth_raster(
        elevation, params.region_smooth_radius, params.region_smooth_sigma
    )
    slope = compute_slope(smoothed)

    binary = slope < params.region_slope_threshold

    dilated_network = channel_network.copy(deep=True)
    dilated_network.data = isotropic_dilation(
        channel_network.data > 0, radius=params.region_dilation_radius
    )

    floor = remove_isolated_areas(binary, channel_network)
    floor = floor.astype(np.uint8)
    floor.data[np.isnan(elevation.data)] = 255
    floor = floor.rio.set_nodata(255)
    floor = floor.rio.write_nodata(255)
    return floor
