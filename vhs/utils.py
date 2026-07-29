import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray
from rasterio.transform import xy
from shapely.geometry import LineString
from skimage.morphology import label


# Florinsky's 5x5 third-order polynomial fit: p (dz/dx) and q (dz/dy) are each
# a fixed linear combination of the 25 cells in the window, with the center
# cell (0, 0) carrying coefficient 0 in both. Coefficients derived by
# expanding the original closed-form p/q equations term by term.
_FLORINSKY_COEFS = (
    ((-2, -2), 31, -31), ((-2, -1), -44, 5), ((-2, 0), 0, 17), ((-2, 1), 44, 5), ((-2, 2), -31, -31),
    ((-1, -2), -5, 44), ((-1, -1), -62, 62), ((-1, 0), 0, 68), ((-1, 1), 62, 62), ((-1, 2), 5, 44),
    ((0, -2), -17, 0), ((0, -1), -68, 0), ((0, 1), 68, 0), ((0, 2), 17, 0),
    ((1, -2), -5, -44), ((1, -1), -62, -62), ((1, 0), 0, -68), ((1, 1), 62, -62), ((1, 2), 5, -44),
    ((2, -2), 31, 31), ((2, -1), -44, -5), ((2, 0), 0, -17), ((2, 1), 44, -5), ((2, 2), -31, 31),
)


def calculate_slope(dem, resolution=None, z_factor=1.0):
    """
    Slope via Florinsky's 5x5 third-order polynomial fit, ported from
    WhiteboxTools' Slope tool (whitebox_tools, Lindsay, MIT license), which
    implements equations from:
    Florinsky, I. (2016). Digital Terrain Analysis in Soil Science and
    Geology, 2nd ed., Chapter 4, p. 117. Academic Press.
    """
    is_xarray = hasattr(dem, "rio") and hasattr(dem, "values")

    if is_xarray:
        if resolution is None:
            res_x, _ = dem.rio.resolution()
            resolution = abs(res_x)
        arr = dem.values.astype(np.float64)
    else:
        arr = dem
        if resolution is None:
            raise ValueError("resolution must be provided for plain array input")

    rows, cols = arr.shape
    center = arr * z_factor
    p_accum = np.zeros_like(arr)
    q_accum = np.zeros_like(arr)

    # accumulate p/q one shifted window at a time instead of materializing
    # all 25 shifted grids simultaneously (previously ~25x the DEM's memory
    # footprint at once; this keeps only a handful of full-size arrays alive)
    for (dy, dx), pc, qc in _FLORINSKY_COEFS:
        shifted = np.full_like(arr, np.nan)
        r0s, r1s = max(0, -dy), rows - max(0, dy)
        c0s, c1s = max(0, -dx), cols - max(0, dx)
        r0d, r1d = max(0, dy), rows - max(0, -dy)
        c0d, c1d = max(0, dx), cols - max(0, -dx)
        shifted[r0d:r1d, c0d:c1d] = arr[r0s:r1s, c0s:c1s]
        shifted *= z_factor
        z_i = np.where(np.isnan(shifted), center, shifted)
        del shifted

        if pc:
            p_accum += pc * z_i
        if qc:
            q_accum += qc * z_i
        del z_i

    scale = 1.0 / (420.0 * resolution)
    p_accum *= scale
    q_accum *= scale

    slope_arr = np.degrees(np.arctan(np.hypot(p_accum, q_accum)))
    slope_arr[np.isnan(arr)] = np.nan

    if is_xarray:
        return dem.copy(data=slope_arr)
    return slope_arr


def remove_isolated_areas(
    binary: xr.DataArray, flowpaths: xr.DataArray
) -> xr.DataArray:
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
            n for n in _get_8_neighbors(current) if n in pixels and n not in visited
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
        (r + dr, c + dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)
    ]
