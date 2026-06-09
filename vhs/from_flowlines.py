"""
Optional streamkit-based workflow for preparing inputs to map_valley_floor.

This module demonstrates one way to produce the four raster inputs required
by the core algorithm (dem, hand, channel_network, subbasins) from a DEM and
a vector flowlines dataset. It requires streamkit as an optional dependency:

    pip install "valley-floor[streamkit] @ git+https://github.com/avkoehl/valley-floor.git"

Users with their own hydrological preprocessing workflows do not need this
module — they can pass their own dem, hand, channel_network, and subbasins
directly to map_valley_floor.
"""

import geopandas as gpd
import xarray as xr


def prepare_inputs(
    dem: xr.DataArray,
    flowlines: gpd.GeoDataFrame,
    reach_penalty: int = 5,
    reach_min_length: int = 1_000,
    reach_smooth_window: int = 5,
    reach_threshold_degrees: float = 1.0,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Prepare inputs for map_valley_floor using streamkit.

    Produces dem, hand, channel_network, and subbasins from a raw DEM and
    vector flowlines. The returned dem is the original input — it is not
    hydrologically conditioned. HAND and subbasins are derived from a
    conditioned DEM internally.

    Args:
        dem: Raw digital elevation model.
        flowlines: Vector stream network as a GeoDataFrame of LineStrings.
        reach_penalty: PELT penalty for reach segmentation. Higher values
            produce fewer, longer reaches.
        reach_min_length: Minimum reach length in meters.
        reach_smooth_window: Window size for smoothing slope before segmentation.
        reach_threshold_degrees: Merge adjacent reaches if slope difference
            is below this threshold in degrees.

    Returns:
        Tuple of (dem, hand, channel_network, subbasins), ready to pass
        directly to map_valley_floor.
    """
    try:
        from streamkit.extraction import (
            channel_heads_from_flowlines,
            extract_channel_network,
            delineate_subbasins,
        )
        from streamkit.segmentation import delineate_reaches
        from streamkit.terrain import compute_hand, flow_accumulation_workflow
    except ImportError:
        raise ImportError(
            "prepare_inputs requires streamkit. "
            'Install it with: pip install "valley-floor[streamkit] @ git+https://github.com/avkoehl/valley-floor.git"'
        )

    channel_heads = channel_heads_from_flowlines(flowlines, dem)
    conditioned, flow_dir, flow_acc = flow_accumulation_workflow(dem)
    channel_network = extract_channel_network(channel_heads, flow_dir)
    channel_network = delineate_reaches(
        channel_network,
        dem,
        flow_dir,
        flow_acc,
        reach_penalty,
        reach_min_length,
        reach_smooth_window,
        reach_threshold_degrees,
    )

    subbasins = delineate_subbasins(channel_network, flow_dir, flow_acc)
    hand = compute_hand(conditioned, channel_network)

    return dem, hand, channel_network, subbasins
