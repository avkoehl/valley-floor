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
from valley_floor.postprocess import remove_isolated_areas


def low_slope_region(
    slope,
    channel_network,
    slope_threshold=3.0,  # degrees
    dilation_radius=3.0,  # pixels
):
    floor = slope.copy(deep=True)
    floor.data = np.zeros_like(slope.data, dtype=bool)

    binary = slope < slope_threshold

    dilated_network = channel_network.copy(deep=True)
    dilated = isotropic_dilation(channel_network.data > 0, radius=dilation_radius)
    dilated_network.data = dilated

    floor = remove_isolated_areas(binary, channel_network)

    floor = floor.astype(np.uint8)
    return floor
