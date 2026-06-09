import numpy as np
import geopandas as gpd
import xarray as xr
from rasterio.transform import xy
from shapely.geometry import LineString
from skimage.morphology import label


def remove_isolated_areas(binary: xr.DataArray, flowpaths: xr.DataArray) -> xr.DataArray:
    fp = flowpaths > 0
    combined = (fp + binary) > 0

    con = label(combined, connectivity=2).astype(np.float64)
    values = np.unique(con[flowpaths.values > 0])
    values = values[np.isfinite(values)]

    result = flowpaths.copy(data=con)
    result = result.where(np.isin(con, values))
    return result > 0


def raster_network_to_vector(channel_network: xr.DataArray) -> gpd.GeoDataFrame:
    """Vectorize a labeled channel network raster into per-reach LineStrings.

    Naive geometric vectorization — traces the pixel skeleton of each reach into
    an ordered LineString but does not establish network topology, connectivity
    between reaches, or flow direction. Each reach is treated independently.
    Suitable for generating cross sections but not for network analysis.

    Args:
        channel_network: Labeled raster where each reach has a unique integer ID
            and non-channel pixels are 0 or nodata. Expected to be a 1-pixel
            skeleton (e.g. output of a stream link operation).

    Returns:
        GeoDataFrame with one row per reach, columns: stream_id, geometry.
    """
    data = channel_network.values
    transform = channel_network.rio.transform()
    crs = channel_network.rio.crs

    rows, cols = np.where((data != 0) & ~np.isnan(data))
    ids = data[rows, cols]

    pixel_lookup: dict[int, set] = {}
    for sid, r, c in zip(ids, rows, cols):
        sid = int(sid)
        if sid not in pixel_lookup:
            pixel_lookup[sid] = set()
        pixel_lookup[sid].add((int(r), int(c)))

    flowlines = []
    for sid, pixels in pixel_lookup.items():
        endpoint = _find_endpoint(pixels)
        path = _walk_chain(pixels, endpoint)
        if len(path) < 2:
            continue
        path_rows, path_cols = zip(*path)
        xs, ys = xy(transform, path_rows, path_cols, offset="center")
        flowlines.append({"geometry": LineString(zip(xs, ys)), "stream_id": sid})

    return gpd.GeoDataFrame(flowlines, crs=crs)


def _find_endpoint(pixels: set) -> tuple[int, int]:
    for pixel in pixels:
        neighbors = [n for n in _get_8_neighbors(pixel) if n in pixels]
        if len(neighbors) <= 1:
            return pixel
    return next(iter(pixels))


def _walk_chain(pixels: set, start: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    visited = {start}
    current = start
    while True:
        neighbors = [
            n for n in _get_8_neighbors(current)
            if n in pixels and n not in visited
        ]
        if not neighbors:
            break
        current = neighbors[0]
        visited.add(current)
        path.append(current)
    return path


def _get_8_neighbors(pixel: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pixel
    return [
        (r + dr, c + dc)
        for dr in [-1, 0, 1]
        for dc in [-1, 0, 1]
        if (dr, dc) != (0, 0)
    ]
