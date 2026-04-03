import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_holes
from skimage.morphology import label
import xarray as xr


def smooth_raster(
    raster: xr.DataArray, spatial_radius: float, sigma: float
) -> xr.DataArray:
    resolution = raster.rio.resolution()[0]
    radius_pixels = int(round(spatial_radius / resolution))
    raster_copy = raster.copy(deep=True)
    raster_copy.data = _filter_nan_gaussian_conserving(
        raster_copy.data, radius_pixels, sigma
    )
    return raster_copy


def _filter_nan_gaussian_conserving(arr, radius_pixels, sigma):
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
    filled.data = remove_small_holes(filled.data, max_size=num_cells)
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
