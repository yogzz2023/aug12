import numpy as np
from scipy.stats import chi2

def sph2cart(az, el, r):
    az = np.radians(az)
    el = np.radians(el)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def mahalanobis_distance(x, y, cov_inv):
    delta = y - x
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

def chi_squared_test(measurement, track, cov_inv, chi2_threshold):
    distances = [mahalanobis_distance(np.array(track_measurement[:3]), np.array(measurement[:3]), cov_inv) for track_measurement in track]
    min_distance = min(distances)
    return min_distance < chi2_threshold

def doppler_gate_check(measurement, track, doppler_threshold):
    doppler_diffs = [abs(track_measurement[3] - measurement[3]) for track_measurement in track]
    min_doppler_diff = min(doppler_diffs)
    return min_doppler_diff < doppler_threshold

def range_gate_check(measurement, track, range_threshold):
    range_diffs = [np.linalg.norm(np.array(track_measurement[:3]) - np.array(measurement[:3])) for track_measurement in track]
    min_range_diff = min(range_diffs)
    return min_range_diff < range_threshold

def initialize_tracks(measurements, cov_inv, chi2_threshold, doppler_threshold, range_threshold):
    tracks = []
    track_ids = []
    hit_count = []
    miss_count = []

    for measurement in measurements:
        assigned = False
        for track_id, track in enumerate(tracks):
            doppler_passed = doppler_gate_check(measurement, track, doppler_threshold)
            range_passed = range_gate_check(measurement, track, range_threshold)
            chi_squared_passed = chi_squared_test(measurement, track, cov_inv, chi2_threshold)
            
            if doppler_passed and range_passed:
                tracks[track_id].append(measurement)
                hit_count[track_id] += 1
                assigned = True
                break
            elif doppler_passed and not range_passed and chi_squared_passed:
                tracks[track_id].append(measurement)
                hit_count[track_id] += 1
                assigned = True
                break
            else:
                miss_count[track_id] += 1

        if not assigned:
            track_id = len(tracks)
            tracks.append([measurement])
            track_ids.append(track_id)
            hit_count.append(1)
            miss_count.append(0)
    
    return tracks, track_ids, hit_count, miss_count

# Sample measurements (azimuth, elevation, range, Doppler velocity, time)
sample_measurements = [
    (10, 5, 100, 10, 0.1),
    (12, 6, 105, 11, 0.2),
    (9, 4, 98, 10.5, 0.3),
    (50, 20, 500, 20, 0.4),
    (52, 22, 505, 21, 0.5),
    (11, 5, 102, 10.2, 0.6),
    (53, 23, 510, 21.5, 0.7),
    (55, 25, 515, 22, 0.8),
]

# Convert sample measurements to Cartesian coordinates
converted_measurements = [sph2cart(az, el, r) + (doppler,) for az, el, r, doppler, _ in sample_measurements]

# Covariance matrix (assuming identity for simplicity)
cov_matrix = np.eye(3)
cov_inv = np.linalg.inv(cov_matrix)

# Chi-squared threshold
state_dim = 3  # 3D state (e.g., x, y, z)
chi2_threshold = chi2.ppf(0.95, df=state_dim)

# Doppler and range thresholds
doppler_threshold = 2  # Arbitrary threshold for Doppler gate
range_threshold = 15  # Arbitrary threshold for range gate

# Initialize tracks
tracks, track_ids, hit_count, miss_count = initialize_tracks(converted_measurements, cov_inv, chi2_threshold, doppler_threshold, range_threshold)

# Output the tracks and their associated measurements
for track_id, track in enumerate(tracks):
    print(f"Track ID {track_id}:")
    for measurement in track:
        print(f"  Measurement: {measurement}")
    print(f"  Hits: {hit_count[track_id]}, Misses: {miss_count[track_id]}")
