import numpy as np
import pytest

from vhs import (
    Parameters,
    ValleyFloorDetailed,
    map_valley_floor,
    map_valley_floor_detailed,
    prepare_inputs,
)
from vhs.data import load_sample_data


@pytest.fixture(scope="session")
def inputs():
    dem, flowlines = load_sample_data()
    dem, hand, channel_network, subbasins = prepare_inputs(dem, flowlines)
    return dem, hand, channel_network, subbasins


@pytest.fixture(scope="session")
def result(inputs):
    dem, hand, channel_network, subbasins = inputs
    return map_valley_floor_detailed(dem, hand, channel_network, subbasins)


# --- return types ---


def test_map_valley_floor_returns_dataarray(inputs):
    dem, hand, channel_network, subbasins = inputs
    import xarray as xr

    floor = map_valley_floor(dem, hand, channel_network, subbasins)
    assert isinstance(floor, xr.DataArray)


def test_map_valley_floor_detailed_returns_detailed(result):
    assert isinstance(result, ValleyFloorDetailed)


# --- shape and crs ---


def test_output_shapes(inputs, result):
    dem = inputs[0]
    assert result.valley_floor.shape == dem.shape
    assert result.region_floor.shape == dem.shape
    assert result.flood_floor.shape == dem.shape


def test_output_crs(inputs, result):
    dem = inputs[0]
    assert result.valley_floor.rio.crs == dem.rio.crs
    assert result.region_floor.rio.crs == dem.rio.crs
    assert result.flood_floor.rio.crs == dem.rio.crs


# --- output values ---


def test_binary_raster_values(result):
    for raster in [result.valley_floor, result.region_floor, result.flood_floor]:
        valid = raster.values[raster.values != 255]
        assert set(np.unique(valid)).issubset({0, 1})


def test_valley_floor_nodata(result):
    assert result.valley_floor.rio.nodata == 255


# --- floor is non-trivial ---


def test_valley_floor_has_floor_pixels(result):
    assert (result.valley_floor.values == 1).sum() > 0


def test_both_components_contribute(result):
    assert (result.region_floor.values == 1).sum() > 0
    assert (result.flood_floor.values == 1).sum() > 0


# --- postprocessing is applied (valley_floor <= union of components) ---


def test_postprocessing_applied(result):
    raw_union = (
        (result.region_floor.values == 1) | (result.flood_floor.values == 1)
    ).sum()
    final = (result.valley_floor.values == 1).sum()
    assert final <= raw_union


# --- diagnostics ---


def test_slope_break_pts_nonempty(result):
    assert len(result.slope_break_pts) > 0


def test_reach_thresholds_nonempty(result):
    assert len(result.reach_thresholds) > 0


def test_reach_thresholds_positive(result):
    assert all(v > 0 for v in result.reach_thresholds.values())


# --- validation errors ---


def test_shape_mismatch_raises(inputs):
    import xarray as xr

    dem, hand, channel_network, subbasins = inputs
    bad_hand = hand.isel(x=slice(0, hand.sizes["x"] - 1))
    with pytest.raises(ValueError, match="shape"):
        map_valley_floor(dem, bad_hand, channel_network, subbasins)


def test_missing_subbasin_raises(inputs):
    dem, hand, channel_network, subbasins = inputs
    bad_subbasins = subbasins.copy(deep=True)
    bad_subbasins.data[:] = 0
    with pytest.raises(ValueError, match="subbasin"):
        map_valley_floor(dem, hand, channel_network, bad_subbasins)
