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

def initialize_tracks(measurements, doppler_threshold, range_threshold):
    tracks = []
    track_ids = []
    miss_counts = []
    hit_counts = []

    for measurement in measurements:
        assigned = False
        measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
        for track_id, track in enumerate(tracks):
            last_measurement = track[-1]
            last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
            
            # Calculate Doppler correlation
            doppler_correlated = doppler_correlation(measurement[3], last_measurement[3], doppler_threshold)
            
            # Calculate Range gate
            distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))
            range_satisfied = range_gate(distance, range_threshold)
            
            if doppler_correlated and range_satisfied:
                tracks[track_id].append(measurement)
                hit_counts[track_id] += 1
                miss_counts[track_id] = 0  # Reset miss count on a hit
                print(f"Measurement {measurement} assigned to Track ID {track_id}: Both Doppler and Range conditions satisfied.")
                assigned = True
                break
            elif doppler_correlated:
                tracks[track_id].append(measurement)
                hit_counts[track_id] += 1
                miss_counts[track_id] = 0
                print(f"Measurement {measurement} assigned to Track ID {track_id}: Doppler condition satisfied, but Range condition not satisfied.")
                assigned = True
                break
            elif range_satisfied:
                tracks[track_id].append(measurement)
                hit_counts[track_id] += 1
                miss_counts[track_id] = 0
                print(f"Measurement {measurement} assigned to Track ID {track_id}: Range condition satisfied, but Doppler condition not satisfied.")
                assigned = True
                break

        if not assigned:
            track_id = len(tracks)
            tracks.append([measurement])
            track_ids.append(track_id)
            miss_counts.append(0)
            hit_counts.append(1)  # Initiate with 1 hit count since it's a new track
            print(f"Measurement {measurement} initiated a new Track ID {track_id}.")

    return tracks, track_ids, miss_counts, hit_counts

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
]

# Parameters for gating
doppler_threshold = 2.0  # Doppler gate threshold
range_threshold = 10.0   # Range gate threshold in Cartesian distance

# Initialize tracks
tracks, track_ids, miss_counts, hit_counts = initialize_tracks(sample_measurements, doppler_threshold, range_threshold)

# Output the tracks and their associated measurements
for track_id, track in enumerate(tracks):
    print(f"Track ID {track_id}:")
    for measurement in track:
        print(f"  Measurement: {measurement}")
    print(f"  Hits: {hit_counts[track_id]}, Misses: {miss_counts[track_id]}")
