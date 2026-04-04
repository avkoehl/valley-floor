import geopandas as gpd
import numpy as np
import xarray as xr

from .config import Parameters, PreprocessingParameters, PostprocessingParameters
from .postprocess import postprocess
from .components import grow_region, flood_reaches


def delineate(
    dem: xr.DataArray,
    hand: xr.DataArray,
    subbasins: xr.DataArray,
    xs_coords: gpd.GeoDataFrame,
    channel_network: xr.DataArray,
    params: Parameters = Parameters(),
) -> dict:
    """Delineate valley floors from precomputed hydrological inputs.

    Combines two components: region growing from low-slope areas connected to
    the channel network, and reach flooding based on HAND thresholds derived
    from cross section analysis. Results from both components are unioned into
    a single valley floor raster.

    Args:
        dem: Digital elevation model.
        hand: Height Above Nearest Drainage raster, aligned to dem.
        subbasins: Raster of subbasin labels, one per stream reach.
        xs_coords: Cross section sample points as a GeoDataFrame with columns
            xs_id, distance, point_id, geometry, and interp_elevation.
        channel_network: Labeled stream network raster used for region growing
            connectivity. Typically the trunk network with headwaters removed.
        params: Algorithm parameters. Defaults to Parameters().

    Returns:
        Dictionary with keys:
            - valley_floor: Union of region and flood components (uint8, nodata=255).
            - region_floor: Output of the region growing component.
            - flood_floor: Output of the reach flooding component.
            - slope_break_pts: GeoDataFrame of detected valley wall points.
            - reach_thresholds: Dict mapping subbasin ID to HAND threshold.
    """
    region_floor = grow_region(dem, channel_network, params)
    flood_floor, slope_break_pts, reach_thresholds = flood_reaches(
        xs_coords,
        hand,
        subbasins,
        params,
    )
    floor = (region_floor == 1) | (flood_floor == 1)
    floor = floor.astype(np.uint8)
    floor.data[np.isnan(dem.data)] = 255
    floor = floor.rio.set_nodata(255)
    floor = floor.rio.write_nodata(255)
    return {
        "valley_floor": floor,
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
    postprocessing_params: PostprocessingParameters | None = None,
) -> dict:
    """Delineate valley floors from a DEM and flowlines.

    Runs the full pipeline: preprocessing with streamkit to derive hydrological
    inputs, core delineation, and optionally postprocessing. This is the
    recommended entry point for users who do not have precomputed HAND,
    subbasins, or cross sections.

    Requires the streamkit optional dependency:
        pip install "valley-floor[streamkit]"

    Args:
        dem: Digital elevation model.
        flowlines: Stream network as a GeoDataFrame of directed LineStrings,
            from upstream to downstream.
        params: Core algorithm parameters. Defaults to Parameters().
        preprocessing_params: Preprocessing pipeline parameters.
            Defaults to PreprocessingParameters().
        postprocessing_params: If provided, postprocessing is applied to the
            valley floor raster. If None, the raw union of region and flood
            components is returned. Defaults to None.

    Returns:
        Dictionary with keys:
            - valley_floor: Final valley floor raster (uint8, nodata=255).
              If postprocessing_params is provided this is the postprocessed
              result, otherwise the raw union of the two components.
            - region_floor: Output of the region growing component.
            - flood_floor: Output of the reach flooding component.
            - slope_break_pts: GeoDataFrame of detected valley wall points.
            - reach_thresholds: Dict mapping subbasin ID to HAND threshold.
            - channel_network: Full labeled stream network raster.
            - channel_network_gdf: Full stream network as a GeoDataFrame.
            - trunk_network: Stream network with headwaters removed (raster).
            - trunk_network_gdf: Stream network with headwaters removed (GeoDataFrame).
    """
    from .inputs import prepare_inputs

    pre = prepare_inputs(dem, flowlines, preprocessing_params)
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
    if postprocessing_params is not None:
        result["valley_floor"] = postprocess(
            result["valley_floor"],
            pre["channel_network"],
            dem,
            postprocessing_params,
        )
    return result
