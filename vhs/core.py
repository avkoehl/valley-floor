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
from vhs.utils.routing import compute_flow_directions


@contextmanager
def _stage(name):
    t0 = time.perf_counter()
    yield
    logger.info("{} done in {:.2f}s", name, time.perf_counter() - t0)


def map_valley_floor(
    dem: xr.DataArray,
    channel_network: xr.DataArray,
    params: Parameters | None = None,
) -> xr.DataArray:
    """Delineate the valley floor; see `map_valley_floor_detailed`."""
    return map_valley_floor_detailed(dem, channel_network, params).valley_floor


def map_valley_floor_detailed(
    dem: xr.DataArray,
    channel_network: xr.DataArray,
    params: Parameters | None = None,
) -> ValleyFloorDetailed:
    """Delineate the valley floor from a conditioned DEM and a reach-labeled
    channel network, returning the intermediate rasters/tables alongside the
    final valley floor (see `ValleyFloorDetailed`).

    `dem` must be hydrologically conditioned - D8 flow directions are
    computed from it internally (`compute_flow_directions`), so that pairing
    is always consistent. `channel_network` must be derived using the same
    flat tie-break convention `compute_flow_directions` uses (see its
    docstring) - reach flooding routes along the computed flow directions,
    so a channel network resolved differently in flats (disproportionately
    valley bottoms and floodplains) produces meaningless results there. This
    isn't validated (not verifiable from the rasters alone); it's on the
    caller to ensure it holds.

    Runs, in order: flow direction computation, headwater filtering, region
    growing, reach flooding, headwater reattachment, and postprocessing.
    Emits an INFO-level log line via loguru after each stage with its
    elapsed time - disabled by default (loguru's default level), enable
    with `logger.enable("vhs")` or by configuring a sink at INFO or below.
    """
    if params is None:
        params = Parameters()

    _validate_inputs(dem, channel_network)
    (channel_network,) = _align_coords(dem, channel_network)

    with _stage("compute_flow_directions"):
        flow_directions = compute_flow_directions(dem)

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
            flow_directions,
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


def _align_coords(dem, *others):
    """Snap x/y coordinates onto dem's.

    Rasters produced by separate pipeline runs can carry transforms that
    agree to float32 precision but differ in the last bit or two of their
    float64 pixel size, so their x/y coordinate arrays aren't bit-identical
    even when `_validate_inputs` confirms the shapes match. Left alone, that
    causes xarray's automatic coordinate alignment on binary ops (e.g.
    `a + b` in remove_isolated_areas) to silently intersect to a smaller
    grid instead of raising. Since shapes are already validated as equal,
    the rasters are meant to be pixel-for-pixel co-registered, so it's safe
    to force them onto a single coordinate system.
    """
    return tuple(o.assign_coords(x=dem["x"], y=dem["y"]) for o in others)


def _validate_inputs(dem, channel_network):
    shapes = {"dem": dem.shape, "channel_network": channel_network.shape}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"All inputs must have the same shape. Got: {shapes}")


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
