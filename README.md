# valley-floor

A Python package for delineating valley floors from digital elevation models (DEMs).

Valley floors are the topographic region between valley walls shaped mainly by fluvial
processes, composed of floodplains, terraces, alluvial fans, and channels.

The method combines two components:

1. **Region growing** — low-slope pixels connected to the channel network. Captures
   wider, unconfined valley floors with multiple channels and floodplains.
2. **Reach flooding** — reach-specific relative elevation thresholds derived from
   cross-section analysis. Captures narrow, confined valley floors bounded by steep
   hillslopes.

## Installation

Core package (no preprocessing):
```bash
pip install "valley-floor @ git+https://github.com/avkoehl/valley-floor.git"
```

With the optional preprocessing pipeline (requires streamkit):
```bash
pip install "valley-floor[streamkit] @ git+https://github.com/avkoehl/valley-floor.git"
```

For development:
```bash
git clone git@github.com:avkoehl/valley-floor.git
cd valley-floor
uv sync                        # core only
uv sync --extra streamkit --group dev  # include streamkit and development dependencies
```

tests:
```bash
uv run pytest -v
```

## Contact

Arthur Koehl — avkoehl at ucdavis.edu
