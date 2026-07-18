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

import vhs

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def _extent(da):
    x, y = da.x.values, da.y.values
    return [x.min(), x.max(), y.min(), y.max()]


def _hillshade(dem):
    ls = LightSource(azdeg=315, altdeg=45)
    z = np.nan_to_num(dem.values, nan=np.nanmin(dem.values))
    return ls.hillshade(z, vert_exag=2)


def _panel(ax, title):
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])


def graphical_abstract(data, out_path):
    dem = data["dem"]
    hand = data["hand"]
    subbasins = data["subbasins"]
    channel = data["channel_network"]
    floor = data["valley_floor"]

    ext = _extent(dem)
    hs = _hillshade(dem)

    fig = plt.figure(figsize=(13, 4.2), dpi=150)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.18, 1.15], wspace=0.08)

    # DEM
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(hs, extent=ext, cmap="gray", alpha=1.0)
    ax.imshow(dem.values, extent=ext, cmap="terrain", alpha=0.6)
    _panel(ax, "DEM")

    # HAND
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(hand.values, extent=ext, cmap="Blues_r", vmax=np.nanpercentile(hand, 95))
    _panel(ax, "HAND")

    # Subbasins + channel network
    ax = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(0)
    colors = rng.random((256, 3)) * 0.6 + 0.3
    sub = subbasins.values.copy()
    ids = np.unique(sub[np.isfinite(sub)])
    remap = np.full(sub.shape, np.nan)
    for i, v in enumerate(ids):
        remap[sub == v] = i % 256
    ax.imshow(remap, extent=ext, cmap=ListedColormap(colors), interpolation="nearest")
    net = np.where(np.isfinite(channel.values) & (channel.values != 0), 1.0, np.nan)
    ax.imshow(net, extent=ext, cmap=ListedColormap(["#111111"]), interpolation="nearest")
    _panel(ax, "Subbasins + channel network")

    # arrow
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#333333"))
    ax.text(0.5, 0.62, "vhs", ha="center", fontsize=11, style="italic")

    # Valley floor over hillshade
    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(hs, extent=ext, cmap="gray")
    vf = np.where(floor.values == 1, 1.0, np.nan)
    ax.imshow(vf, extent=ext, cmap=ListedColormap(["#2b8cbe"]), alpha=0.75,
              interpolation="nearest")
    _panel(ax, "Valley floor")

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out_path)


def main():
    ASSETS.mkdir(exist_ok=True)
    data = vhs.load_sample()
    graphical_abstract(data, ASSETS / "graphical_abstract.png")


if __name__ == "__main__":
    main()
