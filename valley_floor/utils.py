import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import label


def filter_nan_gaussian_conserving(arr, spatial_radius, resolution, sigma):
    """Apply a gaussian filter to an array with nans.
    modified from:
    https://stackoverflow.com/a/61481246

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    radius_pixels = int(round(spatial_radius / resolution))
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = gaussian_filter(
        loss, sigma=sigma, mode="constant", cval=1, radius=radius_pixels
    )

    gauss = arr.copy()
    gauss[nan_msk] = 0
    gauss = gaussian_filter(
        gauss, sigma=sigma, mode="constant", cval=0, radius=radius_pixels
    )
    gauss[nan_msk] = np.nan

    gauss += loss * arr

    return gauss


def connected(binary, flowpaths):
    """
    Keep only connected components in a binary raster that intersect with flowpaths.
    """
    fp = flowpaths > 0
    combined = fp + binary
    combined.data[~np.isfinite(binary)] = np.nan
    combined = combined > 0

    con = label(combined, connectivity=2)
    con = con.astype(np.float64)
    con[~np.isfinite(binary)] = np.nan

    values = np.unique(con[flowpaths > 0])
    values = values[np.isfinite(values)]

    result = flowpaths.copy()
    result.data = con
    result = result.where(np.isin(con, values))
    result = result > 0
    return result
