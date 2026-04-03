import numpy as np
import xarray as xr
import geopandas as gpd

from streamkit.extraction import (
    channel_heads_from_flowlines,
    extract_channel_network,
    delineate_subbasins,
)
from streamkit.network import (
    analyze_network,
    create_cross_sections,
    cross_sections_to_points,
)
from streamkit.segmentation import delineate_reaches
from streamkit.terrain import compute_hand, flow_accumulation_workflow

from .config import PreprocessingParameters


def preprocess(
    dem: xr.DataArray,
    flowlines: gpd.GeoDataFrame,
    params: PreprocessingParameters = PreprocessingParameters(),
) -> dict:
    """Prepare hydrological inputs for valley floor delineation using streamkit.

    Runs the full preprocessing pipeline: channel head detection, flow routing,
    network extraction, reach segmentation, headwater pruning, subbasin
    delineation, HAND computation, and cross section generation.

    Args:
        dem: Digital elevation model.
        flowlines: Stream network as a GeoDataFrame of directed LineStrings,
            from upstream to downstream.
        params: Preprocessing parameters. Defaults to PreprocessingParameters().

    Returns:
        Dictionary with keys:
            - channel_network: Full labeled stream network raster including headwaters.
            - channel_network_gdf: Full stream network as a GeoDataFrame.
            - trunk_network: Stream network with headwaters removed (raster).
            - trunk_network_gdf: Stream network with headwaters removed (GeoDataFrame).
            - subbasins: Raster of subbasin labels, one per trunk reach.
            - hand: Height Above Nearest Drainage raster.
            - xs_coords: Cross section sample points as a GeoDataFrame.
    """
    channel_heads = channel_heads_from_flowlines(flowlines, dem)
    conditioned, flow_dir, flow_acc = flow_accumulation_workflow(dem)
    channel_network = extract_channel_network(channel_heads, flow_dir)

    channel_network = delineate_reaches(
        channel_network,
        dem,
        flow_dir,
        flow_acc,
        params.reach_penalty,
        params.reach_min_length,
        params.reach_smooth_window,
        params.reach_threshold_degrees,
    )
    channel_network_gdf = analyze_network(channel_network, dem, flow_dir, flow_acc)

    trunk_network_gdf, trunk_network = _prune_headwaters(
        channel_network,
        channel_network_gdf,
        params.headwater_min_catchment_area,
        params.headwater_max_mean_slope,
    )

    subbasins = delineate_subbasins(trunk_network, flow_dir, flow_acc)
    hand = compute_hand(conditioned, trunk_network)
    xs_coords = _generate_cross_section_points(
        trunk_network_gdf,
        dem,
        params.xs_interval_distance,
        params.xs_length,
        params.xs_point_spacing,
    )

    return {
        "channel_network": channel_network,
        "channel_network_gdf": channel_network_gdf,
        "trunk_network": trunk_network,
        "trunk_network_gdf": trunk_network_gdf,
        "subbasins": subbasins,
        "hand": hand,
        "xs_coords": xs_coords,
    }


def _generate_cross_section_points(
    network_gdf, elevation, interval_distance, length, point_spacing
):
    xs_lines = create_cross_sections(
        network_gdf.geometry, interval_distance, length, smoothed=True
    )
    xs_coords = cross_sections_to_points(xs_lines, point_spacing)
    xs_coords["point_id"] = np.arange(len(xs_coords))
    xs_coords["interp_elevation"] = elevation.interp(
        x=xr.DataArray(xs_coords["geometry"].x.values),
        y=xr.DataArray(xs_coords["geometry"].y.values),
        method="linear",
    ).values
    return xs_coords


def _prune_headwaters(channel_network, network_gdf, min_catchment_area, max_mean_slope):
    to_remove_ids = []
    network = network_gdf.copy()
    network["segment_id"] = (network["stream_id"] // 1000).astype(int)
    for _, segment in network.groupby("segment_id"):
        if segment["strahler"].min() >= 2:
            continue
        segment = segment.sort_values("contributing_area")
        for _, reach in segment.iterrows():
            if (
                reach["contributing_area"] < min_catchment_area
                or reach["mean_slope"] > max_mean_slope
            ):
                to_remove_ids.append(reach["stream_id"])
            else:
                break

    trunk_gdf = network_gdf[~network_gdf["stream_id"].isin(to_remove_ids)].copy()
    trunk_network = channel_network.copy(deep=True)
    for val in to_remove_ids:
        trunk_network.data[channel_network.data == val] = 0
    return trunk_gdf, trunk_network
