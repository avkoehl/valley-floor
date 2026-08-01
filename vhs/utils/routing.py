"""D8 flow-direction routing, used to derive per-reach flood thresholds and
apply per-reach flooding without a precomputed subbasins/HAND raster.

Both `route_points_to_reach` and `flood_from_reaches` walk the same D8 graph
(built once per call via `_downstream_index`) and share one invariant: a
pixel's "gain" relative to a reach is its own elevation minus the elevation
of the specific channel cell the path connects to - carried forward
unchanged along the path rather than accumulated hop by hop, so it isn't
inflated by any local non-monotonic wobble in the DEM. Both also allow a
path to pass freely through other reaches' cells; only the caller's chosen
target/seed reach ends a walk, so a downstream reach can flood upstream
through a tributary's territory (and vice versa isn't possible, since a
walk only ever moves downstream).

This relies on `flow_directions` coming from a DEM conditioned to be
monotonically non-increasing along the flow path. `flow_directions` is
computed internally (`compute_flow_directions`, below) from `dem` itself, so
that pairing is consistent by construction - the caller only needs to
ensure `channel_network` was derived using the same tie-break convention
`compute_flow_directions` uses (see its docstring), since a mismatch there
would still produce meaningless routing. A locally non-monotonic path could
also get pruned early during flooding (see
`_bounded_multiclaim_flood_numba`); given a properly conditioned DEM this is
not expected to matter in practice.
"""

import numba
import numpy as np
import xarray as xr
from numba import types
from numba.typed import Dict

_DIR_CODES = np.array([64, 128, 1, 2, 4, 8, 16, 32], dtype=np.int16)
_DIR_DROW = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
_DIR_DCOL = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)


def compute_flow_directions(dem: xr.DataArray) -> xr.DataArray:
    """Compute D8 flow direction from a conditioned DEM (ESRI encoding).

    Steepest descent, with a tie-break rule chosen specifically to make the
    result acyclic by construction: exact elevation ties (flats) only ever
    flow toward a strictly larger row-major index, never a smaller one.
    Since every step is either strictly-decreasing elevation or
    flat-with-increasing-index, no sequence of steps can return to its
    starting cell.

    A `channel_network` derived externally (rather than from this same
    function) must resolve flats the same way to stay aligned with the flow
    directions this computes - flats are disproportionately valley bottoms
    and floodplains, the terrain this package cares about most, so a
    tie-break mismatch is more likely to matter here than it would
    elsewhere.
    """
    res_x, res_y = dem.rio.resolution()
    dx, dy = abs(res_x), abs(res_y)
    flow_dir_arr = _flow_direction_numba(
        dem.values.astype(np.float64), dx, dy, _DIR_CODES, _DIR_DROW, _DIR_DCOL
    )
    flow_dir = dem.copy(data=flow_dir_arr)
    flow_dir.encoding = {}
    flow_dir = flow_dir.rio.write_nodata(0)
    return flow_dir


@numba.njit(cache=True)
def _flow_direction_numba(dem_arr, dx, dy, dir_codes, dir_drow, dir_dcol):
    nrows, ncols = dem_arr.shape
    flow_dir = np.zeros((nrows, ncols), dtype=np.int16)

    diag = np.sqrt(dx * dx + dy * dy)
    dists = np.array([dy, diag, dx, diag, dy, diag, dx, diag])

    for row in range(nrows):
        for col in range(ncols):
            elev = dem_arr[row, col]
            if np.isnan(elev):
                continue

            self_index = row * ncols + col
            best_slope = 0.0
            best_code = 0
            best_flat_code = 0
            best_flat_index = -1

            for k in range(8):
                nr = row + dir_drow[k]
                nc = col + dir_dcol[k]
                if not (0 <= nr < nrows and 0 <= nc < ncols):
                    continue
                nelev = dem_arr[nr, nc]
                if np.isnan(nelev):
                    continue

                if nelev < elev:
                    slope = (elev - nelev) / dists[k]
                    if slope > best_slope:
                        best_slope = slope
                        best_code = dir_codes[k]
                elif nelev == elev:
                    n_index = nr * ncols + nc
                    if n_index > self_index and n_index > best_flat_index:
                        best_flat_index = n_index
                        best_flat_code = dir_codes[k]

            if best_code != 0:
                flow_dir[row, col] = best_code
            elif best_flat_code != 0:
                flow_dir[row, col] = best_flat_code
            # else: leave as 0 (terminal - pit/edge/no valid direction)

    return flow_dir


def _esri_dirmap():
    dirmap = {
        64: (-1, 0),  # North
        128: (-1, 1),  # Northeast
        1: (0, 1),  # East
        2: (1, 1),  # Southeast
        4: (1, 0),  # South
        8: (1, -1),  # Southwest
        16: (0, -1),  # West
        32: (-1, -1),  # Northwest
        -1: (0, 0),  # outlet/terminal
        -2: (0, 0),
        0: (0, 0),
    }
    dirmap_numba = Dict.empty(
        key_type=types.int64, value_type=types.UniTuple(types.int64, 2)
    )
    for k, v in dirmap.items():
        dirmap_numba[k] = v
    return dirmap_numba


@numba.njit(cache=True)
def _downstream_index(flow_dir_arr, dirmap):
    nrows, ncols = flow_dir_arr.shape
    downstream = np.full(nrows * ncols, -1, dtype=np.int64)
    for row in range(nrows):
        for col in range(ncols):
            code = flow_dir_arr[row, col]
            if code == 0 or code == -1 or code == -2:
                continue
            dr, dc = dirmap[code]
            nr, nc = row + dr, col + dc
            if 0 <= nr < nrows and 0 <= nc < ncols:
                downstream[row * ncols + col] = nr * ncols + nc
    return downstream


def _build_downstream_index(flow_directions: xr.DataArray) -> np.ndarray:
    dirmap = _esri_dirmap()
    values = flow_directions.values
    if np.issubdtype(values.dtype, np.floating):
        # a float flow_directions raster (e.g. loaded with masked=True, so
        # its nodata sentinel became NaN) would otherwise cast to undefined
        # int64 garbage below; treat NaN as terminal, same as 0/-1/-2.
        values = np.nan_to_num(values, nan=0.0)
    return _downstream_index(values.astype(np.int64), dirmap)


def _reach_label_int(channel_network: xr.DataArray) -> np.ndarray:
    """Labeled reach raster as int64, with -1 for non-channel/nodata pixels
    (rather than 0, so a reach can never be legitimately labeled 0)."""
    values = channel_network.values
    if np.issubdtype(values.dtype, np.floating):
        is_channel = np.isfinite(values) & (values != 0)
    else:
        is_channel = values != 0
    return np.where(is_channel, values, -1).astype(np.int64)


# --- point -> reach routing (cross-section breakpoint thresholds) ---


@numba.njit(cache=True)
def _route_points_numba(start_idx, target_ids, dem_flat, label_flat, downstream, max_steps):
    n = len(start_idx)
    gains = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)
    visited = np.empty(max_steps, dtype=np.int64)

    for i in range(n):
        idx = start_idx[i]
        target = target_ids[i]
        start_elev = dem_flat[idx]

        if label_flat[idx] == target:
            gains[i] = 0.0
            valid[i] = True
            continue

        count = 0
        visited[count] = idx
        count += 1
        current = idx
        while count < max_steps:
            nxt = downstream[current]
            if nxt == -1:
                break
            already = False
            for k in range(count):
                if visited[k] == nxt:
                    already = True
                    break
            if already:
                break
            visited[count] = nxt
            count += 1
            current = nxt
            if label_flat[current] == target:
                gains[i] = start_elev - dem_flat[current]
                valid[i] = True
                break

    return gains, valid


def route_points_to_reach(
    rows: np.ndarray,
    cols: np.ndarray,
    target_reach_ids: np.ndarray,
    dem: xr.DataArray,
    channel_network: xr.DataArray,
    flow_directions: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Route each point (row/col pixel coords) downhill along
    `flow_directions` until it enters its own `target_reach_ids[i]`'s cells
    in `channel_network`, passing freely through any other reach's cells
    encountered along the way.

    Returns (gains, valid): `gains[i]` is the point's own elevation minus
    the elevation of the specific channel cell where it entered its target
    reach (NaN where invalid). `valid[i]` is False if the point's flow path
    never reaches its target reach - it exits the domain or terminates
    (pit/flat) first.
    """
    downstream = _build_downstream_index(flow_directions)

    nrows, ncols = dem.shape
    dem_flat = dem.values.astype(np.float64).reshape(-1)
    label_flat = _reach_label_int(channel_network).reshape(-1)

    start_idx = (
        np.asarray(rows, dtype=np.int64) * ncols + np.asarray(cols, dtype=np.int64)
    )
    target_ids = np.asarray(target_reach_ids, dtype=np.int64)

    return _route_points_numba(
        start_idx, target_ids, dem_flat, label_flat, downstream, nrows * ncols
    )


# --- bounded, multi-claim reach flooding ---


@numba.njit(cache=True)
def _bounded_multiclaim_flood_numba(
    seed_idx_sorted, seed_elev_sorted, seed_boundaries, thresholds, elevation, downstream, nrows, ncols
):
    n = nrows * ncols
    n_reaches = thresholds.shape[0]
    flooded = np.zeros(n, dtype=np.bool_)
    # which reach's *current* traversal last claimed a pixel - a same-reach
    # dedup guard only (prevents reprocessing/cycles within one reach's own
    # run); a different reach revisiting an already-flooded pixel is exactly
    # the multi-claim behavior this is meant to allow, not something to guard
    # against, so this is never checked across reaches.
    last_claimed_by = np.full(n, -1, dtype=np.int64)
    carried_elev = np.zeros(n, dtype=np.float64)
    queue = np.empty(n, dtype=np.int64)

    dir_drow = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
    dir_dcol = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int64)

    for r in range(n_reaches):
        threshold = thresholds[r]
        head = 0
        tail = 0
        for s in range(seed_boundaries[r], seed_boundaries[r + 1]):
            idx = seed_idx_sorted[s]
            flooded[idx] = True
            carried_elev[idx] = seed_elev_sorted[s]
            last_claimed_by[idx] = r
            queue[tail] = idx
            tail += 1

        while head < tail:
            idx = queue[head]
            head += 1
            row = idx // ncols
            col = idx % ncols
            base_elev = carried_elev[idx]

            for d in range(8):
                nr = row + dir_drow[d]
                nc = col + dir_dcol[d]
                if not (0 <= nr < nrows and 0 <= nc < ncols):
                    continue
                n_idx = nr * ncols + nc
                if downstream[n_idx] != idx:
                    continue
                if last_claimed_by[n_idx] == r:
                    continue
                gain = elevation[n_idx] - base_elev
                if gain > threshold:
                    continue
                flooded[n_idx] = True
                carried_elev[n_idx] = base_elev
                last_claimed_by[n_idx] = r
                queue[tail] = n_idx
                tail += 1

    return flooded


def flood_from_reaches(
    channel_network: xr.DataArray,
    flow_directions: xr.DataArray,
    dem: xr.DataArray,
    reach_thresholds: dict,
) -> xr.DataArray:
    """Flood every reach in `channel_network` up to its own threshold in
    `reach_thresholds`, growing upstream along `flow_directions` from each
    reach's own channel cells, unconstrained by watershed/subbasin
    boundaries. A pixel can be flooded by more than one reach.

    Reach ids in `channel_network` with no entry in `reach_thresholds` are
    not seeded (not flooded by that reach).
    """
    labels = _reach_label_int(channel_network)
    rows, cols = np.where(labels >= 0)
    reach_ids = labels[rows, cols]

    unique_reaches = np.array(sorted(reach_thresholds.keys()), dtype=np.int64)
    mask = np.isin(reach_ids, unique_reaches)
    rows, cols, reach_ids = rows[mask], cols[mask], reach_ids[mask]

    nrows, ncols = dem.shape
    seed_idx = (rows.astype(np.int64) * ncols + cols.astype(np.int64))
    seed_elev = dem.values[rows, cols].astype(np.float64)
    compact_idx = np.searchsorted(unique_reaches, reach_ids)
    thresholds = np.array(
        [reach_thresholds[int(rid)] for rid in unique_reaches], dtype=np.float64
    )

    order = np.argsort(compact_idx, kind="stable")
    seed_idx_sorted = seed_idx[order]
    seed_elev_sorted = seed_elev[order]
    compact_sorted = compact_idx[order]
    boundaries = np.searchsorted(
        compact_sorted, np.arange(len(unique_reaches) + 1)
    ).astype(np.int64)

    downstream = _build_downstream_index(flow_directions)
    elevation = dem.values.astype(np.float64).reshape(-1)

    flooded_flat = _bounded_multiclaim_flood_numba(
        seed_idx_sorted,
        seed_elev_sorted,
        boundaries,
        thresholds,
        elevation,
        downstream,
        nrows,
        ncols,
    )
    flooded = flooded_flat.reshape(nrows, ncols)

    floor = dem.copy(data=flooded.astype(np.uint8))
    floor.data[np.isnan(dem.data)] = 255
    floor = floor.rio.set_nodata(255)
    floor = floor.rio.write_nodata(255, encoded=True)
    return floor
