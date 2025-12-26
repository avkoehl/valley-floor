from .config import Config
from .components import grow_region, flood_reaches
from .postprocess import process_floor
from .preprocess import process_hydro

from xrspatial import slope as compute_slope


def delineate_valley_floor(
    dem,
    channel_heads,
    config: Config = Config(),
):
    (
        channel_network,
        channel_network_gdf,
        trunk_network,
        trunk_network_gdf,
        subbasins,
        hand,
    ) = process_hydro(
        dem,
        channel_heads,
        config.reach,
        config.headwater_filter,
    )

    region_floor = grow_region(dem, trunk_network, config.region)
    flood_floor, break_pts, reach_thresholds = flood_reaches(
        trunk_network_gdf,
        dem,
        hand,
        subbasins,
        config.cross_section,
        config.slope_break,
        config.threshold,
    )
    trunk_network_gdf["threshold"] = trunk_network_gdf["stream_id"].map(
        reach_thresholds
    )
    valley_floor = process_floor(
        region_floor,
        flood_floor,
        channel_network,
        compute_slope(dem),
        config.post_process,
    )
    return {
        "channel_network": channel_network,
        "channel_network_gdf": channel_network_gdf,
        "valley_floor": valley_floor,
        "trunk_network": trunk_network,
        "trunk_network_gdf": trunk_network_gdf,
        "flood_floor": flood_floor,
        "region_floor": region_floor,
        "break_points": break_pts,
    }
