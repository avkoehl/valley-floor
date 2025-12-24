import numpy as np


def find_slope_breaks(gdf, min_slope_degrees, min_elevation_gain):
    """
    Identifies the start of the first sustained steep segment on both sides
    of a cross section.
    """
    # 1. Convert slope degrees to ratio
    min_slope_ratio = np.tan(np.radians(min_slope_degrees))

    # 2. Split into Left (<= 0) and Right (>= 0) based on distance
    # We include 0 in both so the center point can be a start candidate
    left_bank = gdf[gdf["distance"] <= 0].copy()
    right_bank = gdf[gdf["distance"] >= 0].copy()

    results = {}

    for side_name, df in [("left", left_bank), ("right", right_bank)]:
        if df.empty:
            results[side_name] = None
            continue

        # 3. Sort by ABSOLUTE distance (walking outwards from center)
        # This ensures we calculate slope moving *up* the bank regardless of side
        df["abs_dist"] = df["distance"].abs()
        df = df.sort_values("abs_dist")

        # 4. Calculate Forward Slope (Slope to the NEXT point)
        # Shift(-1) lets us compare Row i with Row i+1
        delta_z = df["interp_elevation"].shift(-1) - df["interp_elevation"]
        delta_x = df["abs_dist"].shift(-1) - df["abs_dist"]

        # Calculate slope and handle potential division by zero
        slopes = (delta_z / delta_x).fillna(0)

        # 5. Create Boolean Mask for Steep Segments
        is_steep = slopes >= min_slope_ratio

        # 6. Group Consecutive Steep Segments
        # This is the vectorized equivalent of your while loop.
        # It assigns a unique ID to every block of consecutive True/False values.
        segment_ids = (is_steep != is_steep.shift()).cumsum()

        # 7. Aggregate Elevation Gain per Segment
        # We only care about segments where is_steep is True
        steep_segments = df[is_steep].groupby(segment_ids)

        found_point = None

        # Iterate through the groups (in order of distance from center)
        for _, group in steep_segments:
            # Get the point AFTER the last point in the group to calc full gain
            # Because 'group' contains points [i, i+1...], the slope exists BETWEEN them.
            # The total gain is (Elevation of Last Point's Next Neighbor) - (Elevation of First Point)

            start_idx = group.index[0]
            last_idx = group.index[-1]

            # Look up the actual end elevation (the point after the steep segment ends)
            # We use the original dataframe to grab the next point
            try:
                # Find the integer location of the last point in the sorted df
                loc_end = df.index.get_loc(last_idx)
                end_elev = df.iloc[loc_end + 1]["interp_elevation"]
                start_elev = group.iloc[0]["interp_elevation"]

                if (end_elev - start_elev) > min_elevation_gain:
                    found_point = group.iloc[0]["point_id"]
                    break  # Stop at the first valid break found
            except IndexError:
                # Reached end of array
                continue

        results[side_name] = found_point

    return results
