"""

Use slope threshold and channel network to delineate valley floor as being all low slope areas connected to the channel network.

Likely want to preprocess the channel network to remove starts as they can connect to flat hill tops.
Can do this by thresholding upstream channel length or using a minimum stream order.

Likely want to preprocess the slope raster by smoothing it to remove noise. See `valley_floor.utils.filter_nan_gaussian_conserving`.

inputs:
    - slope raster (degrees)
    - channel network (binary raster, 1 for channel, 0 for non-channel)
    - slope threshold (optional, default 10 degrees)

outputs:
    - valley floor raster (boolean, 1 for valley floor, 0 for non-valley floor)
"""

import numpy as np
from skimage.morphology import isotropic_dilation
from streamkit.terrain import gaussian_smooth_raster
from xrspatial import slope as compute_slope
import xarray as xr

from valley_floor.postprocess import remove_isolated_areas
from valley_floor.config import RegionParameters


def grow_region(
    elevation: xr.DataArray,
    channel_network: xr.DataArray,
    region_params: RegionParameters,
) -> xr.DataArray:
    smoothed = gaussian_smooth_raster(
        elevation,
        spatial_radius=region_params.smooth_radius,
        sigma=region_params.smooth_sigma,
    )
    slope = compute_slope(smoothed)

    floor = slope.copy(deep=True)
    floor.data = np.zeros_like(slope.data, dtype=bool)

    binary = slope < region_params.slope_threshold

    dilated_network = channel_network.copy(deep=True)
    dilated = isotropic_dilation(
        channel_network.data > 0, radius=region_params.dilation_radius
    )
    dilated_network.data = dilated

    floor = remove_isolated_areas(binary, channel_network)

    floor = floor.astype(np.uint8)
    floor = floor.rio.set_nodata(255)
    floor = floor.rio.write_nodata(255)
    floor.data[np.isnan(elevation.data)] = 255
    return floor
