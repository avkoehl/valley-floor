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

Core package: 
```bash
pip install "valley-floor @ git+https://github.com/avkoehl/valley-floor.git"
```

With the optional preprocessing pipeline and example dataset (requires streamkit):
```bash
pip install "valley-floor[streamkit] @ git+https://github.com/avkoehl/valley-floor.git"
```

For development:
```bash
git clone git@github.com:avkoehl/valley-floor.git
cd valley-floor
uv sync                        # core only
uv sync --extra streamkit --group dev  # include streamkit and development dependencies
uv run ipykernal install --user --name valley-floor # register the package's virtual environment as a Jupyter kernel
```

## Tests
```bash
uv run pytest -v
```

## Documentation

```
uv run quartodoc build
quarto preview
```

## Contact

Arthur Koehl — avkoehl at ucdavis.edu
