import geopandas as gdf
from streamkit import rasterize_nhd

from valley_floor.flood import flood_extent
from valley_floor.region import low_slope_region
from valley_floor.dynamic_flood import dynamic_flood_thresholds
from valley_floor.preprocess import prepare_region_inputs, prepare_flood_inputs
from valley_floor.postprocess import process_floor
from valley_floor.config import Config


def delineate_valley_floor(
    dem,
    channel_network,
    config: Config = Config(),
    debug_returns: bool = False,
):
    if isinstance(channel_network, gdf.GeoDataFrame):
        channel_network = rasterize_nhd(
            channel_network,
            dem,
        )

    # preprocess
    region_inputs = prepare_region_inputs(
        dem,
        channel_network,
        **config.preprocess_region,
    )
    flood_inputs = prepare_flood_inputs(
        dem,
        channel_network,
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
        result = {
            "region_floor": region_floor,
            "flood_floor": flood_floor,
            "processed_floor": processed_floor,
            **region_inputs,
            **flood_inputs,
        }
        if config.flood_delineation.get("dynamic"):
            result["dynamic_thresholds"] = thresholds
            result["wallpoints"] = wallpoints
        return result
    return processed_floor
