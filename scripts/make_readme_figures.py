"""Generate the README figures from the bundled sample dataset.

Run from the repo root:

    uv run python scripts/make_readme_figures.py

Writes PNGs into ``assets/``. These are committed and referenced by the README;
they are not part of the installed package.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from scipy.ndimage import maximum_filter

import vfmap

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _extent(da):
    x, y = da.x.values, da.y.values
    return [x.min(), x.max(), y.min(), y.max()]


def _hillshade(dem):
    ls = LightSource(azdeg=315, altdeg=45)
    valid = np.isfinite(dem.values)
    z = np.nan_to_num(dem.values, nan=np.nanmin(dem.values))
    hs = ls.hillshade(z, vert_exag=2)
    hs[~valid] = np.nan  # nodata stays transparent instead of flat mid-grey
    return hs


def _panel(ax, title):
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_contours(ax, dem, ext, interval=5):
    z = dem.values
    lo = np.floor(np.nanmin(z) / interval) * interval
    hi = np.nanmax(z)
    levels = np.arange(lo, hi + interval, interval)
    ax.contour(
        z, levels=levels, extent=ext, origin="upper",
        colors="black", linewidths=0.35, alpha=0.3,
    )


def graphical_abstract(data, out_path):
    dem = data["dem"]
    channel = data["channel_network"]
    floor = data["valley_floor"]

    ext = _extent(dem)
    hs = _hillshade(dem)

    fig = plt.figure(figsize=(10.4, 4.2), dpi=150)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.18, 1.15], wspace=0.08)

    # DEM
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(hs, extent=ext, cmap="gray", alpha=1.0)
    ax.imshow(dem.values, extent=ext, cmap="terrain", alpha=0.6)
    _add_contours(ax, dem, ext)
    _panel(ax, "Conditioned DEM")

    # Channel network, colored by reach id, over hillshade
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(hs, extent=ext, cmap="gray", alpha=1.0)
    _add_contours(ax, dem, ext)
    rng = np.random.default_rng(0)
    colors = rng.random((256, 3)) * 0.7 + 0.15
    net_ids = channel.values.copy()
    valid = np.isfinite(net_ids) & (net_ids != 0)
    ids = np.unique(net_ids[valid])
    remap = np.full(net_ids.shape, -1.0)
    for i, v in enumerate(ids):
        remap[net_ids == v] = i % 256
    remap = maximum_filter(remap, size=3)  # thicken 1px lines for legibility
    remap = np.where(remap >= 0, remap, np.nan)
    ax.imshow(remap, extent=ext, cmap=ListedColormap(colors), interpolation="nearest")
    _panel(ax, "Channel network")

    # arrow
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#333333"))
    ax.text(0.5, 0.62, "vfmap", ha="center", fontsize=11, style="italic")

    # Valley floor over hillshade
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(hs, extent=ext, cmap="gray")
    _add_contours(ax, dem, ext)
    vf = np.where(floor.values == 1, 1.0, np.nan)
    ax.imshow(vf, extent=ext, cmap=ListedColormap(["#2b8cbe"]), alpha=0.75,
              interpolation="nearest")
    _panel(ax, "Valley floor")

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out_path)


def main():
    ASSETS.mkdir(exist_ok=True)
    data = vfmap.load_sample()
    graphical_abstract(data, ASSETS / "graphical_abstract.png")


if __name__ == "__main__":
    main()
