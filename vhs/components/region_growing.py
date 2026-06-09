import numpy as np
from scipy.ndimage import gaussian_filter
from xrspatial import slope as compute_slope
import xarray as xr

from vhs.utils import remove_isolated_areas


def grow_region(
    dem: xr.DataArray,
    channel_network: xr.DataArray,
    slope_threshold: float,
    smooth_sigma: float,
) -> xr.DataArray:
    smoothed = _smooth_raster(dem, smooth_sigma)
    slope = compute_slope(smoothed)

    binary = slope < slope_threshold
    floor = remove_isolated_areas(binary, channel_network)
    floor = floor.astype(np.uint8)
    floor.data[np.isnan(dem.data)] = 255
    floor = floor.rio.set_nodata(255)
    floor = floor.rio.write_nodata(255)
    return floor


def _smooth_raster(raster: xr.DataArray, sigma_meters: float) -> xr.DataArray:
    resolution = abs(raster.rio.resolution()[0])
    sigma_pixels = sigma_meters / resolution

    nan_mask = np.isnan(raster.values)
    data = raster.values.copy()

    loss = np.zeros_like(data)
    loss[nan_mask] = 1.0
    loss = gaussian_filter(loss, sigma=sigma_pixels, mode="constant", cval=1.0)

    data[nan_mask] = 0.0
    smoothed = gaussian_filter(data, sigma=sigma_pixels, mode="constant", cval=0.0)
    smoothed[nan_mask] = np.nan
    smoothed += loss * raster.values

    return raster.copy(data=smoothed)
