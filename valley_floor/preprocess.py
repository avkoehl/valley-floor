from streamkit import (
    gaussian_smooth_raster,
    upstream_length_raster,
    flow_accumulation_workflow,
    vectorize_streams,
    delineate_reaches,
    delineate_subbasins,
    network_cross_sections,
    sample_cross_sections,
    compute_hand,
)

from xrspatial import slope as calculate_slope


def prepare_region_inputs(
    dem,
    channel_network,
    smooth_radius=90,
    smooth_sigma=30,
    upstream_length_threshold=1000,
):
    _, flow_dir, _ = flow_accumulation_workflow(dem)
    coarse_dem = gaussian_smooth_raster(dem, smooth_radius, smooth_sigma)
    slope = calculate_slope(coarse_dem)
    upstream_length = upstream_length_raster(channel_network, flow_dir)
    mask = upstream_length >= upstream_length_threshold
    trimmed_channels = channel_network.where(mask, 0)
    return {
        "smoothed_slope": slope,
        "trimmed_channels": trimmed_channels,
    }


def prepare_flood_inputs(
    dem,
    channel_network,
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
    _, flow_dir, flow_acc = flow_accumulation_workflow(dem)
    reaches = delineate_reaches(
        channel_network,
        dem,
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
    hand = compute_hand(dem, channel_network, flow_dir)
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
