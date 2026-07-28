import time
from contextlib import contextmanager

import numpy as np
import xarray as xr
from loguru import logger

from vhs.config import Parameters, ValleyFloorDetailed
from vhs.headwaters import filter_headwaters
from vhs.postprocess import postprocess
from vhs.components.region_growing import grow_region
from vhs.components.reach_flooding import flood_reaches


@contextmanager
def _stage(name):
    t0 = time.perf_counter()
    yield
    logger.info("{} done in {:.2f}s", name, time.perf_counter() - t0)


def map_valley_floor(
    dem: xr.DataArray,
    hand: xr.DataArray,
    channel_network: xr.DataArray,
    subbasins: xr.DataArray,
    params: Parameters | None = None,
) -> xr.DataArray:
    """Delineate the valley floor; see `map_valley_floor_detailed`."""
    return map_valley_floor_detailed(
        dem, hand, channel_network, subbasins, params
    ).valley_floor


def map_valley_floor_detailed(
    dem: xr.DataArray,
    hand: xr.DataArray,
    channel_network: xr.DataArray,
    subbasins: xr.DataArray,
    params: Parameters | None = None,
) -> ValleyFloorDetailed:
    """Delineate the valley floor from a DEM, HAND, channel network, and
    subbasins raster, returning the intermediate rasters/tables alongside the
    final valley floor (see `ValleyFloorDetailed`).

    Runs, in order: headwater filtering, region growing, reach flooding,
    headwater reattachment, and postprocessing. Emits an INFO-level log line
    via loguru after each stage with its elapsed time - disabled by default
    (loguru's default level), enable with `logger.enable("vhs")` or by
    configuring a sink at INFO or below.
    """
    if params is None:
        params = Parameters()

    _validate_inputs(dem, hand, channel_network, subbasins)

    with _stage("filter_headwaters"):
        filtered_network, headwater_ids = filter_headwaters(channel_network, dem, params)

    with _stage("grow_region"):
        region_floor = grow_region(
            dem,
            filtered_network,
            params.region_slope_threshold,
            params.region_smooth_sigma,
        )

    with _stage("flood_reaches"):
        flood_floor, slope_break_pts, reach_thresholds = flood_reaches(
            filtered_network,
            dem,
            hand,
            subbasins,
            params,
        )

    floor = (region_floor == 1) | (flood_floor == 1)
    floor = floor.astype(np.uint8)
    floor.data[np.isnan(dem.data)] = 255
    floor = floor.rio.set_nodata(255)
    floor = floor.rio.write_nodata(255)

    with _stage("reattach_headwaters"):
        floor = _reattach_headwaters(floor, channel_network, headwater_ids)

    with _stage("postprocess"):
        floor = postprocess(floor, channel_network, dem, params)

    return ValleyFloorDetailed(
        valley_floor=floor,
        region_floor=region_floor,
        flood_floor=flood_floor,
        slope_break_pts=slope_break_pts,
        reach_thresholds=reach_thresholds,
    )


def _validate_inputs(dem, hand, channel_network, subbasins):
    shapes = {
        "dem": dem.shape,
        "hand": hand.shape,
        "channel_network": channel_network.shape,
        "subbasins": subbasins.shape,
    }
    if len(set(shapes.values())) > 1:
        raise ValueError(f"All inputs must have the same shape. Got: {shapes}")

    network_ids = set(np.unique(channel_network.values)) - {
        0,
        channel_network.rio.nodata,
    }
    subbasin_ids = set(np.unique(subbasins.values)) - {0, subbasins.rio.nodata}
    network_ids = {i for i in network_ids if not np.isnan(i)}
    subbasin_ids = {i for i in subbasin_ids if not np.isnan(i)}
    missing = network_ids - subbasin_ids
    if missing:
        raise ValueError(
            f"channel_network contains reach IDs with no matching subbasin: {missing}"
        )


def _reattach_headwaters(
    floor: xr.DataArray,
    channel_network: xr.DataArray,
    headwater_ids: list[int],
) -> xr.DataArray:
    floor = floor.copy(deep=True)
    # np.isin does one pass over the raster testing membership in
    # headwater_ids, instead of one `== sid` full-raster scan per id
    floor.data[np.isin(channel_network.data, headwater_ids)] = 1
    return floor
