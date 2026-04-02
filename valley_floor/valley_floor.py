import xarray as xr
import geopandas as gpd

from .config import Parameters, PreprocessingParameters
from .components import grow_region, flood_reaches


def delineate(
    dem: xr.DataArray,
    hand: xr.DataArray,
    subbasins: xr.DataArray,
    xs_coords: gpd.GeoDataFrame,
    channel_network: xr.DataArray,
    params: Parameters = Parameters(),
) -> dict:
    region_floor = grow_region(dem, channel_network, params)
    flood_floor, slope_break_pts, reach_thresholds = flood_reaches(
        xs_coords,
        hand,
        subbasins,
        params,
    )
    return {
        "region_floor": region_floor,
        "flood_floor": flood_floor,
        "slope_break_pts": slope_break_pts,
        "reach_thresholds": reach_thresholds,
    }


def delineate_from_dem_and_flowlines(
    dem: xr.DataArray,
    flowlines: gpd.GeoDataFrame,
    params: Parameters = Parameters(),
    preprocessing_params: PreprocessingParameters = PreprocessingParameters(),
) -> dict:
    from .preprocess import preprocess

    pre = preprocess(dem, flowlines, preprocessing_params)
    result = delineate(
        dem,
        pre["hand"],
        pre["subbasins"],
        pre["xs_coords"],
        pre["trunk_network"],
        params,
    )
    result.update(
        {
            "channel_network": pre["channel_network"],
            "channel_network_gdf": pre["channel_network_gdf"],
            "trunk_network": pre["trunk_network"],
            "trunk_network_gdf": pre["trunk_network_gdf"],
        }
    )
    return result
