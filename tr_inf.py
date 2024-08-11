import numpy as np

def sph2cart(az, el, r):
    az = np.radians(az)
    el = np.radians(el)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def doppler_correlation(doppler_1, doppler_2, doppler_threshold):
    return abs(doppler_1 - doppler_2) < doppler_threshold

def range_gate(distance, range_threshold):
    return distance < range_threshold

def initialize_tracks(measurements, doppler_threshold, range_threshold, firm_threshold):
    tracks = []
    track_ids = {}
    miss_counts = {}
    hit_counts = {}
    tentative_ids = {}
    firm_ids = set()

    for measurement in measurements:
        measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
        measurement_doppler = measurement[3]

        # Flag to determine if measurement was assigned
        assigned = False

        for track_id, track in enumerate(tracks):
            last_measurement = track[-1]
            last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
            last_doppler = last_measurement[3]

            # Calculate distance and check conditions
            distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))
            doppler_correlated = doppler_correlation(measurement_doppler, last_doppler, doppler_threshold)
            range_satisfied = range_gate(distance, range_threshold)

            if doppler_correlated and range_satisfied:
                if track_id not in firm_ids:
                    if track_id in tentative_ids:
                        hit_counts[track_id] += 1
                        miss_counts[track_id] = 0
                        if hit_counts[track_id] >= firm_threshold:
                            firm_ids.add(track_id)
                            print(f"Track ID {track_id} is now firm.")
                    else:
                        tentative_ids[track_id] = True
                        hit_counts[track_id] = 1
                        miss_counts[track_id] = 0
                tracks[track_id].append(measurement)
                print(f"Measurement {measurement} assigned to Track ID {track_id}: Both Doppler and Range conditions satisfied.")
                assigned = True
                break
            elif doppler_correlated or range_satisfied:
                # Prefer Doppler correlation if both conditions are not met
                if doppler_correlated:
                    if track_id not in firm_ids:
                        if track_id in tentative_ids:
                            hit_counts[track_id] += 1
                            miss_counts[track_id] = 0
                            if hit_counts[track_id] >= firm_threshold:
                                firm_ids.add(track_id)
                                print(f"Track ID {track_id} is now firm.")
                        else:
                            tentative_ids[track_id] = True
                            hit_counts[track_id] = 1
                            miss_counts[track_id] = 0
                    tracks[track_id].append(measurement)
                    print(f"Measurement {measurement} assigned to Track ID {track_id}: Doppler condition satisfied.")
                    assigned = True
                    break
                elif range_satisfied:
                    if track_id not in firm_ids:
                        if track_id in tentative_ids:
                            hit_counts[track_id] += 1
                            miss_counts[track_id] = 0
                            if hit_counts[track_id] >= firm_threshold:
                                firm_ids.add(track_id)
                                print(f"Track ID {track_id} is now firm.")
                        else:
                            tentative_ids[track_id] = True
                            hit_counts[track_id] = 1
                            miss_counts[track_id] = 0
                    tracks[track_id].append(measurement)
                    print(f"Measurement {measurement} assigned to Track ID {track_id}: Range condition satisfied.")
                    assigned = True
                    break

        if not assigned:
            # Create a new track
            track_id = len(tracks)
            tracks.append([measurement])
            track_ids[track_id] = track_id
            miss_counts[track_id] = 0
            hit_counts[track_id] = 1
            tentative_ids[track_id] = True
            print(f"Measurement {measurement} initiated a new Track ID {track_id}.")

        # Increment miss count for all tracks that were not assigned this measurement
        for track_id in range(len(tracks)):
            if track_id not in firm_ids and not assigned:
                if track_id in miss_counts:
                    miss_counts[track_id] += 1
                    if miss_counts[track_id] > firm_threshold:
                        print(f"Track ID {track_id} has too many misses and will be removed.")
                        # Optionally, remove track if too many misses
                        tracks[track_id] = []

    return tracks, track_ids, miss_counts, hit_counts, firm_ids

# Sample measurements (azimuth, elevation, range, doppler)
sample_measurements = [
    (10, 5, 100, 5),
    (12, 6, 105, 6),
    (9, 4, 98, 4.5),
    (50, 20, 500, 50),
    (52, 22, 505, 52),
    (11, 5, 102, 5.5),
    (53, 23, 510, 53),
    (55, 25, 515, 55),
    (14, 7, 110, 6.2),
    (51, 21, 495, 49),
    (13, 6, 107, 6.1),
    (54, 24, 512, 54),
    (56, 26, 520, 56),
    (15, 8, 115, 6.5),
    (48, 19, 480, 47),
    (16, 9, 120, 7),
    (57, 27, 525, 57),
    (17, 10, 125, 7.2),
    (49, 20, 485, 48),
    (18, 11, 130, 7.5),
    (58, 28, 530, 58),
    (19, 12, 135, 8),
    (60, 29, 540, 60),
    (20, 13, 140, 8.2),
    (61, 30, 550, 61),
    (21, 14, 145, 8.5),
    (62, 31, 560, 62),
    (22, 15, 150, 9),
    (63, 32, 570, 63),
    (23, 16, 155, 9.2),
    (64, 33, 580, 64),
    (24, 17, 160, 9.5),
    (65, 34, 590, 65),
    (25, 18, 165, 10),
    (66, 35, 600, 66),
]

# Parameters for gating
doppler_threshold = 2.0  # Doppler gate threshold
range_threshold = 10.0   # Range gate threshold in Cartesian distance
firm_threshold = 3       # Number of continuous hits needed to firm a track

# Initialize tracks
tracks, track_ids, miss_counts, hit_counts, firm_ids = initialize_tracks(
    sample_measurements, doppler_threshold, range_threshold, firm_threshold
)

# Output the tracks and their associated measurements
for track_id, track in enumerate(tracks):
    print(f"Track ID {track_id}:")
    for measurement in track:
        print(f"  Measurement: {measurement}")
    print(f"  Hits: {hit_counts.get(track_id, 0)}, Misses: {miss_counts.get(track_id, 0)}")
    if track_id in firm_ids:
        print(f"  Track ID {track_id} is firm.")
    else:
        print(f"  Track ID {track_id} is tentative.")
