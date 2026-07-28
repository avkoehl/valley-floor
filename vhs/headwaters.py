import numpy as np
import xarray as xr

from vhs.utils import raster_network_to_vector
from vhs.config import Parameters


def _index_pixels_by_id(id_raster: np.ndarray) -> dict:
    """Group pixel (row, col) coordinates by integer raster value.

    Single pass over the raster instead of one `== id` scan per id, so this
    is O(pixels log pixels) rather than O(pixels * n_ids). Groups preserve
    row-major (np.where) order within each id, matching the per-id scans it
    replaces.
    """
    valid = np.isfinite(id_raster) & (id_raster != 0)
    rows, cols = np.where(valid)
    ids = id_raster[rows, cols].astype(np.int64)

    order = np.argsort(ids, kind="stable")
    ids_sorted = ids[order]
    rows_sorted = rows[order]
    cols_sorted = cols[order]

    unique_ids, start_idx = np.unique(ids_sorted, return_index=True)
    end_idx = np.append(start_idx[1:], len(ids_sorted))

    return {
        int(uid): (rows_sorted[s:e], cols_sorted[s:e])
        for uid, s, e in zip(unique_ids, start_idx, end_idx)
    }


def filter_headwaters(
    channel_network: xr.DataArray,
    dem: xr.DataArray,
    params: Parameters,
) -> tuple[xr.DataArray, list[int]]:
    network_gdf = raster_network_to_vector(channel_network)
    pixel_index = _index_pixels_by_id(channel_network.values)
    tip_ids = _identify_tips(channel_network, dem, network_gdf, pixel_index)

    headwater_ids = []
    for stream_id in tip_ids:
        reach = network_gdf[network_gdf["stream_id"] == stream_id].iloc[0]
        length = reach.geometry.length
        if length < params.headwater_min_length:
            headwater_ids.append(stream_id)
            continue
        slope = _mean_slope(dem, stream_id, length, pixel_index)
        if slope > params.headwater_max_mean_slope:
            headwater_ids.append(stream_id)

    filtered = channel_network.copy(deep=True)
    for sid in headwater_ids:
        filtered.data[channel_network.data == sid] = 0

    return filtered, headwater_ids


def _identify_tips(
    channel_network: xr.DataArray,
    dem: xr.DataArray,
    network_gdf,
    pixel_index: dict,
) -> list[int]:
    dem_data = dem.values
    net_data = channel_network.values
    nrows, ncols = net_data.shape

    tips = []

    for stream_id in network_gdf["stream_id"]:
        rows, cols = pixel_index.get(int(stream_id), (np.empty(0, dtype=int), np.empty(0, dtype=int)))

        if len(rows) == 0:
            continue

        # single pixel reach is always a tip
        if len(rows) == 1:
            tips.append(stream_id)
            continue

        elevations = dem_data[rows, cols]
        valid_mask = np.isfinite(elevations)
        if not valid_mask.any():
            continue

        # check headwater end: max elevation pixels
        # if no neighbor from a different reach exists → no upstream connection → tip
        max_elev = elevations[valid_mask].max()
        max_pixels = [
            (r, c)
            for r, c, v in zip(rows, cols, elevations)
            if np.isfinite(v) and v == max_elev
        ]

        upstream_connected = False
        for r, c in max_pixels:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < nrows and 0 <= nc < ncols:
                        neighbor = net_data[nr, nc]
                        if (
                            neighbor != 0
                            and neighbor != stream_id
                            and not np.isnan(neighbor)
                        ):
                            upstream_connected = True
                            break
                if upstream_connected:
                    break
            if upstream_connected:
                break

        if not upstream_connected:
            tips.append(stream_id)

    return tips


def _mean_slope(
    dem: xr.DataArray,
    stream_id: int,
    length: float,
    pixel_index: dict,
) -> float:
    rows, cols = pixel_index.get(int(stream_id), (np.empty(0, dtype=int), np.empty(0, dtype=int)))
    elevation_values = dem.values[rows, cols]
    elevation_values = elevation_values[np.isfinite(elevation_values)]
    if len(elevation_values) == 0 or length == 0:
        return 0.0
    elevation_change = float(np.max(elevation_values) - np.min(elevation_values))
    slope = elevation_change / length
    return np.degrees(np.arctan(slope))
