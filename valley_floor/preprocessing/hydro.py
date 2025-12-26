from streamkit.extraction import (
    extract_channel_network,
    delineate_reaches,
    delineate_subbasins,
)
from streamkit.network import analyze_stream_network
from streamkit.terrain import compute_hand_wbt, flow_accumulation_workflow
import xarray as xr
import geopandas as gpd

from ..config import ReachParameters, HeadwaterFilterParameters


def preprocess_hydro(
    dem: xr.DataArray,
    channel_heads: xr.DataArray,
    reach_params: ReachParameters,
    headwater_filter_params: HeadwaterFilterParameters,
) -> tuple[
    xr.DataArray,
    gpd.GeoDataFrame,
    xr.DataArray,
    gpd.GeoDataFrame,
    xr.DataArray,
    xr.DataArray,
]:
    conditioned, flow_dir, flow_acc, channel_network = hydro_base(dem, channel_heads)

    channel_network = delineate_reaches(
        channel_network,
        dem,
        flow_dir,
        flow_acc,
        reach_params.penalty,
        reach_params.min_length,
        reach_params.smooth_window,
        reach_params.threshold_degrees,
    )
    channel_network_gdf = analyze_stream_network(
        channel_network,
        dem,
        flow_dir,
        flow_acc,
    )

    trunk_network_gdf, trunk_network = prune_headwaters(
        channel_network,
        channel_network_gdf,
        headwater_filter_params.min_stream_order,
        headwater_filter_params.max_mean_slope,
    )

    subbasins = delineate_subbasins(trunk_network, flow_dir, flow_acc)
    hand = compute_hand_wbt(conditioned, trunk_network)
    return (
        channel_network,
        channel_network_gdf,
        trunk_network,
        trunk_network_gdf,
        subbasins,
        hand,
    )


def hydro_base(dem, channel_heads):
    conditioned_dem, flow_dir, flow_acc = flow_accumulation_workflow(dem)
    channel_network = extract_channel_network(channel_heads, flow_dir)
    return conditioned_dem, flow_dir, flow_acc, channel_network


def prune_headwaters(channel_network, network_gdf, min_stream_order, max_mean_slope):
    # Identify headwater streams to remove
    # headwater reaches are those at the upstream part of the stream network
    # with high mean slope
    # iterate through each first order segment, iterate down the reaches recording
    # any segments with mean slope > max_mean_slope until reaching a segment that
    # doesn't exceed that slope threshold
    to_remove_ids = []
    network = network_gdf.copy()
    network["segment_id"] = (network["stream_id"] // 1000).astype(int)
    for _, segment in network.groupby("segment_id"):
        if segment["strahler"].min() >= min_stream_order:
            continue
        segment = segment.sort_values("contributing_area")
        for _, reach in segment.iterrows():
            if reach["mean_slope"] > max_mean_slope:
                to_remove_ids.append(reach["stream_id"])
            else:
                break
    trunk_gdf = network_gdf[~network_gdf["stream_id"].isin(to_remove_ids)].copy()

    trunk_network = channel_network.copy(deep=True)
    for val in to_remove_ids:
        trunk_network.data[channel_network.data == val] = 0
    return trunk_gdf, trunk_network
