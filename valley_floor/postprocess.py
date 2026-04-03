import numpy as np
import xarray as xr
from xrspatial import slope as compute_slope

from .config import PostprocessingParameters
from .utils import close_holes, remove_isolated_areas


def postprocess(
    floor: xr.DataArray,
    channel_network: xr.DataArray,
    dem: xr.DataArray,
    params: PostprocessingParameters = PostprocessingParameters(),
) -> xr.DataArray:
    """Postprocess a valley floor raster.

    Applies three refinement steps in order: removes high-slope pixels,
    fills small holes, and removes areas not connected to the channel network.
    Channel network pixels are always included in the output.

    Args:
        floor: Binary valley floor raster to refine (uint8, nodata=255).
            Typically the valley_floor output from delineate().
        channel_network: Full labeled stream network raster including headwaters,
            used for connectivity analysis.
        dem: Digital elevation model, used to compute slope.
        params: Postprocessing parameters. Defaults to PostprocessingParameters().

    Returns:
        Refined valley floor raster (uint8, nodata=255).
    """
    slope = compute_slope(dem)

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
