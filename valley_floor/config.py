from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    preprocess_region: dict = None
    preprocess_flood: dict = None
    postprocess: dict = None
    region_delineation: dict = None
    thresholds: dict = None
    flood_delineation: dict = None

    def __post_init__(self):
        if self.preprocess_region is None:
            self.preprocess_region = {
                "smooth_radius": 90,
                "smooth_sigma": 30,
                "slope_threshold": 5.0,
            }
        if self.preprocess_flood is None:
            self.preprocess_flood = {
                "smooth_radius": 30,
                "smooth_sigma": 10,
                "penalty": 5,
                "min_reach_length": 1000,
                "smooth_window": 5,
                "threshold_degrees": 1,
                "interval_distance": 100,
                "width": 2000,
                "smoothed": True,
                "point_spacing": 10,
            }
        if self.postprocess is None:
            self.postprocess = {"min_size": 40000}
        if self.region_delineation is None:
            self.region_delineation = {"slope_threshold": 5.0, "dilation_radius": 3.0}
        if self.thresholds is None:
            self.thresholds = {
                "min_slope": 10.0,
                "elevation_threshold": 10.0,
                "min_points": 10,
                "percentile": 85,
                "buffer": 1.0,
                "default_threshold": 10,
            }
        if self.flood_delineation is None:
            self.flood_delineation = {"slope_threshold": 15.0, "dynamic": True}

    def to_json(self, filepath):
        """Save configuration to JSON file"""
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, filepath):
        """Load configuration from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)


# Usage
if __name__ == "__main__":
    config = Config()

    # Save to JSON
    config.to_json("config.json")

    # Load from JSON
    config_from_json = Config.from_json("config.json")
    print("Configuration loaded successfully!")
