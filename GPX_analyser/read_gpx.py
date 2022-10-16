import gpxpy
import matplotlib.pyplot as plt
import math
import numpy as np

gpx_file = open('/Users/andrewsneddon/Downloads/COURSE_104175732.gpx', 'r')
gpx = gpxpy.parse(gpx_file)


def longitude_conversion_factor(latitude):
    return 40075*math.cos(2*math.pi*latitude/360)/360


def individual_points_distances(lat_pt1, lat_pt2, long_pt1, long_pt2):
    radius = 6371
    dlat = math.radians(lat_pt2 - lat_pt1)
    dlon = math.radians(long_pt2 - long_pt1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat_pt1)) * math.cos(math.radians(lat_pt2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d*1000


def correct_distance_for_elevation(distance, elevation_1, elevation_2):
    elevation_change = elevation_2 - elevation_1
    distance_corrected = (distance**2+elevation_change**2)**0.5
    return elevation_change, distance_corrected


def estimate_distance_of_gpx_file(latitudes, longitudes, elevations):
    distances = []
    elv_changes = []
    gradients = []
    for i in range(len(latitudes)-1):
        distance = individual_points_distances(
            latitudes[i], latitudes[i+1],
            longitudes[i], longitudes[i+1]
        )
        elv_change, distance_corr = correct_distance_for_elevation(
            distance, elevations[i], elevations[i+1])
        elv_changes.append(elv_change)
        distances.append(distance_corr)
        gradients.append(elv_change/distance_corr*100)

    return np.array(distances), np.array(elv_changes), np.array(gradients)

lats = []  # list to store latitudes
longs = []  # list to store longitudes
elv = []  # list to store elevations
for point in gpx.tracks[0].segments[0].points:
    print('Point at ({0},{1}) -> {2}'.format(point.latitude, point.longitude, point.elevation))
    lats.append(point.latitude)
    longs.append(point.longitude)
    elv.append(point.elevation)

distances, elevation_change, grads = estimate_distance_of_gpx_file(lats, longs, elv)

print(f'Total distance is {round(np.array(distances).sum()/1000, 2)} km')
print(f'Total climbing is {round(elevation_change[elevation_change>0].sum(), 2)} m')
