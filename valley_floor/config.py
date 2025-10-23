from dataclasses import dataclass


@dataclass
class Config:
    preprocess_region = {
        "smooth_radius": 90,
        "smooth_sigma": 30,
        "upstream_length_threshold": 1000,
    }
    preprocess_flood = {
        "smooth_radius": 30,
        "smooth_sigma": 10,
        "penalty": 5,
        "min_reach_length": 500,
        "smooth_window": 5,
        "threshold_degrees": 1,
        "interval_distance": 100,
        "width": 2000,
        "smoothed": True,
        "point_spacing": 10,
    }
    postprocess = {
        "min_size": 40000,
    }
    region_delineation = {"slope_threshold": 3.0, "dilation_radius": 3.0}
    thresholds = {
        "min_slope": 10.0,
        "elevation_threshold": 10.0,
        "min_points": 10,
        "percentile": 80,
        "buffer": 0.0,
        "default_threshold": 10,
    }
    flood_delineation = {"slope_threshold": 10.0, "dynamic": True}
