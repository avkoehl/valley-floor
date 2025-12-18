from streamkit import (
    flow_accumulation_workflow,
    trace_streams,
    link_streams,
)

from valley_floor.flood import flood_extent
from valley_floor.region import low_slope_region
from valley_floor.dynamic_flood import dynamic_flood_thresholds
from valley_floor.preprocess import prepare_region_inputs, prepare_flood_inputs
from valley_floor.postprocess import process_floor
from valley_floor.config import Config


def delineate_valley_floor(
    dem,
    channel_heads,
    config: Config = Config(),
    debug_returns: bool = False,
):
    if dem.rio.crs != channel_heads.rio.crs:
        raise ValueError("DEM and channel heads must have the same CRS.")
    if channel_heads.sum() == 0:
        raise ValueError("No channel heads found in the provided raster.")
    valid_channel_heads = channel_heads.where(dem.notnull() & (channel_heads > 0))
    if valid_channel_heads.sum() == 0:
        raise ValueError("Channel heads do not intersect the DEM extent.")

    cdem, flow_dir, flow_acc = flow_accumulation_workflow(dem)
    streams = trace_streams(channel_heads, flow_dir)
    channel_network = link_streams(streams, flow_dir)

    # preprocess
    region_inputs = prepare_region_inputs(
        dem,
        channel_network,
        flow_dir,
        **config.preprocess_region,
    )
    flood_inputs = prepare_flood_inputs(
        dem,
        cdem,
        channel_network,
        flow_dir,
        flow_acc,
        **config.preprocess_flood,
    )

    # delineate valley floor using both methods
    region_floor = low_slope_region(
        region_inputs["smoothed_slope"],
        region_inputs["trimmed_channels"],
        **config.region_delineation,
    )

    if config.flood_delineation.get("dynamic"):
        thresholds, wallpoints = dynamic_flood_thresholds(
            flood_inputs["cross_section_coords"],
            flood_inputs["smoothed_dem"],
            flood_inputs["smoothed_slope"],
            flood_inputs["hand"],
            flood_inputs["subbasins"],
            return_wallpoints=True,
            **config.thresholds,
        )
    else:
        thresholds = config.thresholds["default_threshold"]

    flood_floor = flood_extent(
        flood_inputs["hand"],
        flood_inputs["smoothed_slope"],
        channel_network,
        slope_threshold=config.flood_delineation.get("slope_threshold", 10.0),
        elevation_threshold=thresholds,
        subbasin=flood_inputs["subbasins"],
    )

    # combine and postprocess
    processed_floor = process_floor(
        flood_floor,
        region_floor,
        channel_network,
        min_size=config.postprocess["min_size"],
        subbasins=flood_inputs["subbasins"],
    )

    if debug_returns:
        from streamkit import vectorize_streams

        vec_reaches = vectorize_streams(flood_inputs["reaches"], flow_dir, flow_acc)
        result = {
            "region_floor": region_floor,
            "flood_floor": flood_floor,
            "processed_floor": processed_floor,
            "vector_reaches": vec_reaches,
            **region_inputs,
            **flood_inputs,
        }
        if config.flood_delineation.get("dynamic"):
            thresholds = {str(k): v for k, v in thresholds.items()}
            vec_reaches["hand_threshold"] = (
                vec_reaches["stream_id"].astype(str).map(thresholds)
            )
            result["vector_reaches"] = vec_reaches
            result["dynamic_thresholds"] = thresholds
            result["wallpoints"] = wallpoints
        return result
    return processed_floor
