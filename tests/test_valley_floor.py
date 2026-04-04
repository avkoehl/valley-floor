import pytest
import numpy as np
from valley_floor import (
    Parameters,
    PreprocessingParameters,
    PostprocessingParameters,
    delineate,
    postprocess,
)
from valley_floor.inputs import prepare_inputs
from valley_floor.data import load_sample_data


@pytest.fixture(scope="session")
def preprocessed():
    dem, flowlines = load_sample_data()
    return dem, prepare_inputs(dem, flowlines, PreprocessingParameters())


@pytest.fixture(scope="session")
def result(preprocessed):
    dem, pre = preprocessed
    res = delineate(
        dem,
        pre["hand"],
        pre["subbasins"],
        pre["xs_coords"],
        pre["trunk_network"],
        Parameters(),
    )
    return dem, pre, res


def test_delineate_runs(result):
    dem, pre, res = result
    assert res is not None


def test_delineate_output_keys(result):
    dem, pre, res = result
    assert "region_floor" in res
    assert "flood_floor" in res


def test_delineate_output_shape(result):
    dem, pre, res = result
    assert res["region_floor"].shape == dem.shape
    assert res["flood_floor"].shape == dem.shape


def test_delineate_output_crs(result):
    dem, pre, res = result
    assert res["region_floor"].rio.crs == dem.rio.crs
    assert res["flood_floor"].rio.crs == dem.rio.crs


def test_delineate_output_values(result):
    dem, pre, res = result
    for key in ["region_floor", "flood_floor"]:
        data = res[key].values
        valid = data[data != 255]
        assert set(np.unique(valid)).issubset({0, 1})


def test_postprocess_runs(result):
    dem, pre, res = result
    floor = postprocess(
        res["valley_floor"],
        pre["channel_network"],
        dem,
        PostprocessingParameters(),
    )
    assert floor is not None


def test_postprocess_changes_output(result):
    dem, pre, res = result
    raw_ones = int(
        (res["region_floor"].values == 1).sum() + (res["flood_floor"].values == 1).sum()
    )
    floor = postprocess(
        res["valley_floor"],
        pre["channel_network"],
        dem,
        PostprocessingParameters(),
    )
    post_ones = int((floor.values == 1).sum())
    assert post_ones != raw_ones


def test_postprocess_output_values(result):
    dem, pre, res = result
    floor = postprocess(
        res["valley_floor"],
        pre["channel_network"],
        dem,
        PostprocessingParameters(),
    )
    valid = floor.values[floor.values != 255]
    assert set(np.unique(valid)).issubset({0, 1})
