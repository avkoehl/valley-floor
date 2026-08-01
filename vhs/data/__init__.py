"""Bundled sample dataset.

A small example watershed (EPSG:2991) used in the README figures, tests, and for
trying out the package without supplying your own data.
"""

from importlib.resources import files

import rioxarray
import xarray as xr

_FILES = {
    "dem": "dem.tif",
    "channel_network": "channel_network.tif",
    "valley_floor": "valley_floor.tif",
}


def sample_data_path(name: str) -> str:
    """Return the filesystem path to a bundled sample raster.

    Parameters
    ----------
    name : str
        One of ``dem``, ``channel_network``, ``valley_floor``.
    """
    if name not in _FILES:
        raise KeyError(f"Unknown sample raster {name!r}. Options: {list(_FILES)}")
    return str(files(__package__).joinpath(_FILES[name]))


def load_sample() -> dict[str, xr.DataArray]:
    """Load the bundled sample dataset.

    Returns a dict with the two model inputs plus a reference ``valley_floor``:

    - ``dem``             : float32 elevation (m), hydrologically conditioned
    - ``channel_network`` : float64 reach-labeled raster, derived using the same
      flat tie-break convention as :func:`vhs.utils.routing.compute_flow_directions`
    - ``valley_floor``    : float32 reference output from :func:`vhs.map_valley_floor`
    """
    return {
        name: rioxarray.open_rasterio(sample_data_path(name), masked=True).squeeze()
        for name in _FILES
    }
