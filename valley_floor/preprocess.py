from streamkit.terrain import (
    gaussian_smooth_raster,
    compute_hand_wbt,
)
from streamkit.extraction import (
    delineate_reaches,
    delineate_subbasins,
)
from streamkit.network import (
    assign_strahler_order,
    vectorize_streams,
    network_cross_sections,
    sample_cross_sections,
    stream_segment_slope,
)

from xrspatial import slope as calculate_slope


def prune_headwaters(channel_network, flow_dir, flow_acc, dem, slope_threshold):
    # remove first order streams and segments with slope above threshold
    segment_slope = stream_segment_slope(
        channel_network, flow_dir, flow_acc, dem, return_dict=False
    )
    strahler = assign_strahler_order(channel_network, flow_dir, flow_acc)

    mask = (strahler >= 2) & (segment_slope <= slope_threshold)
    pruned_channels = channel_network.where(mask, 0)
    return pruned_channels


def prepare_region_inputs(
    dem,
    channel_network,
    flow_dir,
    flow_acc,
    smooth_radius=90,
    smooth_sigma=30,
    slope_threshold=5.0,
):
    coarse_dem = gaussian_smooth_raster(dem, smooth_radius, smooth_sigma)
    slope = calculate_slope(coarse_dem)
    trimmed_channels = prune_headwaters(
        channel_network, flow_dir, flow_acc, dem, slope_threshold
    )
    return {
        "smoothed_slope": slope,
        "trimmed_channels": trimmed_channels,
    }


def prepare_flood_inputs(
    dem,
    conditioned_dem,
    channel_network,
    flow_dir,
    flow_acc,
    smooth_radius=30,
    smooth_sigma=10,
    penalty=5,
    min_reach_length=500,
    smooth_window=5,
    threshold_degrees=1,
    interval_distance=100,
    width=1000,
    smoothed=True,
    point_spacing=10,
):
    reaches = delineate_reaches(
        channel_network,
        dem,
        flow_dir,
        flow_acc,
        penalty=penalty,
        threshold_degrees=threshold_degrees,
        min_length=min_reach_length,
        smooth_window=smooth_window,
    )

    vector_streams = vectorize_streams(reaches, flow_dir, flow_acc)
    xsections = network_cross_sections(
        vector_streams.geometry, interval_distance, width, smoothed=smoothed
    )
    xscoords = sample_cross_sections(xsections, point_spacing)
    hand = compute_hand_wbt(conditioned_dem, channel_network)
    subbasins = delineate_subbasins(reaches, flow_dir, flow_acc)
    smoothed_dem = gaussian_smooth_raster(dem, smooth_radius, smooth_sigma)
    smoothed_slope = calculate_slope(smoothed_dem)
    return {
        "reaches": reaches,
        "cross_section_coords": xscoords,
        "hand": hand,
        "subbasins": subbasins,
        "smoothed_slope": smoothed_slope,
        "smoothed_dem": smoothed_dem,
    }
