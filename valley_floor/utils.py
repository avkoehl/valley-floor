import numpy as np
from skimage.morphology import label


def connected(binary, flowpaths):
    """
    Keep only connected components in a binary raster that intersect with flowpaths.
    """
    fp = flowpaths > 0
    combined = fp + binary
    combined.data[~np.isfinite(binary)] = np.nan
    combined = combined > 0

    con = label(combined, connectivity=2)
    con = con.astype(np.float64)
    con[~np.isfinite(binary)] = np.nan

    values = np.unique(con[flowpaths > 0])
    values = values[np.isfinite(values)]

    result = flowpaths.copy()
    result.data = con
    result = result.where(np.isin(con, values))
    result = result > 0
    return result
