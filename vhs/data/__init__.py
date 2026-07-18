"""Bundled sample dataset.

A small example watershed (EPSG:2991) used in the README figures, tests, and for
trying out the package without supplying your own data.
"""

from importlib.resources import files

import rioxarray
import xarray as xr

_FILES = {
    "dem": "dem.tif",
    "hand": "hand.tif",
    "channel_network": "channel_network.tif",
    "subbasins": "subbasins.tif",
    "valley_floor": "valley_floor.tif",
}


def sample_data_path(name: str) -> str:
    """Return the filesystem path to a bundled sample raster.

    Parameters
    ----------
    name : str
        One of ``dem``, ``hand``, ``channel_network``, ``subbasins``,
        ``valley_floor``.
    """
    if name not in _FILES:
        raise KeyError(f"Unknown sample raster {name!r}. Options: {list(_FILES)}")
    return str(files(__package__).joinpath(_FILES[name]))


def load_sample() -> dict[str, xr.DataArray]:
    """Load the bundled sample dataset.

    Returns a dict with the four model inputs plus a reference ``valley_floor``:

    - ``dem``             : float32 elevation (m)
    - ``hand``            : float32 height above nearest drainage (m)
    - ``channel_network`` : int32 reach-labeled raster (IDs match subbasins)
    - ``subbasins``       : int32 subbasin raster
    - ``valley_floor``    : uint8 reference output from :func:`vhs.map_valley_floor`
    """
    return {
        name: rioxarray.open_rasterio(sample_data_path(name), masked=True).squeeze()
        for name in _FILES
    }
