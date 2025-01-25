#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def graham_scan(point_set):
    def calculate_orientation(point_a, point_b, point_c):
        return (point_b[0] - point_a[0]) * (point_c[1] - point_a[1]) - (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])

    point_set = sorted(point_set, key=lambda coord: (coord[0], coord[1]))
    lower_hull = []
    for point in point_set:
        while len(lower_hull) >= 2 and calculate_orientation(lower_hull[-2], lower_hull[-1], point) <= 0:
            lower_hull.pop()
        lower_hull.append(point)

    upper_hull = []
    for point in reversed(point_set):
        while len(upper_hull) >= 2 and calculate_orientation(upper_hull[-2], upper_hull[-1], point) <= 0:
            upper_hull.pop()
        upper_hull.append(point)

    return np.array(lower_hull[:-1] + upper_hull[:-1])

def jarvis_march(point_set):
    total_points = len(point_set)
    if total_points < 3:
        return point_set

    convex_hull = []
    leftmost_index = np.argmin(point_set[:, 0])
    current_index = leftmost_index
    while True:
        convex_hull.append(point_set[current_index])
        next_index = (current_index + 1) % total_points
        for candidate_index in range(total_points):
            if (np.cross(point_set[candidate_index] - point_set[current_index], point_set[next_index] - point_set[current_index]) > 0):
                next_index = candidate_index
        current_index = next_index
        if current_index == leftmost_index:
            break

    return np.array(convex_hull)

def quickhull(point_set):
    def build_hull(subset, start_point, end_point):
        if not len(subset):
            return []

        distances = np.cross(subset - start_point, end_point - start_point)
        farthest_index = np.argmax(distances)
        farthest_point = subset[farthest_index]

        left_subset_1 = subset[np.cross(subset - start_point, farthest_point - start_point) > 0]
        left_subset_2 = subset[np.cross(subset - farthest_point, end_point - farthest_point) > 0]

        return build_hull(left_subset_1, start_point, farthest_point) + [farthest_point] + build_hull(left_subset_2, farthest_point, end_point)

    point_set = np.unique(point_set, axis=0)
    if len(point_set) < 3:
        return point_set

    leftmost_point = point_set[np.argmin(point_set[:, 0])]
    rightmost_point = point_set[np.argmax(point_set[:, 0])]

    upper_set = point_set[np.cross(point_set - leftmost_point, rightmost_point - leftmost_point) > 0]
    lower_set = point_set[np.cross(point_set - leftmost_point, rightmost_point - leftmost_point) < 0]

    upper_hull = build_hull(upper_set, leftmost_point, rightmost_point)
    lower_hull = build_hull(lower_set, rightmost_point, leftmost_point)

    return np.array([leftmost_point] + upper_hull + [rightmost_point] + lower_hull)

def monotone_chain(point_set):
    point_set = sorted(point_set, key=lambda coord: (coord[0], coord[1]))

    lower_hull = []
    for point in point_set:
        while len(lower_hull) >= 2 and np.cross(lower_hull[-1] - lower_hull[-2], point - lower_hull[-1]) <= 0:
            lower_hull.pop()
        lower_hull.append(point)

    upper_hull = []
    for point in reversed(point_set):
        while len(upper_hull) >= 2 and np.cross(upper_hull[-1] - upper_hull[-2], point - upper_hull[-1]) <= 0:
            upper_hull.pop()
        upper_hull.append(point)

    return np.array(lower_hull[:-1] + upper_hull[:-1])

def visualize_results(point_set, convex_hull_function, plot_title):
    plt.figure()
    plt.title(plot_title)
    plt.scatter(point_set[:, 0], point_set[:, 1], color='blue')
    hull_points = convex_hull_function(point_set)
    hull_points = np.vstack((hull_points, hull_points[0]))
    plt.plot(hull_points[:, 0], hull_points[:, 1], color='blue')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()

if __name__ == '__main__':
    data_points = np.loadtxt('/root/Desktop/host/section_2_129L/mesh.dat', skiprows=1)
    visualize_results(data_points, graham_scan, 'Graham Scan')
    visualize_results(data_points, jarvis_march, 'Jarvis March')
    visualize_results(data_points, quickhull, 'Quickhull')
    visualize_results(data_points, monotone_chain, 'Monotone Chain')
