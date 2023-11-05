from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib as mpl
import math
from itertools import combinations
from sklearn.decomposition import PCA

# mpl.use('macosx')

LINE_COLORS = ["red", "green", "blue", "orange", "purple", "brown", "pink", "black",
               "royalblue", "mediumblue", "darkblue",
               "navy", "steelblue"]

def converge_points_poly(points: np.ndarray, degree: int = 2):
    points = np.array(points)
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    z = points[:, 2]
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)
    reg_y = LinearRegression().fit(x_poly, y)
    reg_z = LinearRegression().fit(x_poly, z)

    return reg_y, reg_z, poly


def draw_interpolation_line(ax, points_cluster, color, degree=1):
    reg_y, reg_z, poly_transform = converge_points_poly(points_cluster, degree=degree)
    points_cluster = np.array(points_cluster)
    x_vals = np.linspace(min(points_cluster[:,0]), max(points_cluster[:,0]), 100).reshape(-1,1)
    x_vals_poly = poly_transform.transform(x_vals)
    y_vals = reg_y.predict(x_vals_poly)
    z_vals = reg_z.predict(x_vals_poly)
    ax.scatter(
        points_cluster[:, 0],
        points_cluster[:, 1],
        points_cluster[:, 2],
        color=color,
        s=5,
        label='Original Points',
    )
    ax.plot(x_vals.flatten(), y_vals, z_vals, color=color, label='Regression Curve')

    return ax


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)


def max_distance(points):
    max_dist = 0
    # Generate all possible pairs of points
    for point1, point2 in combinations(points, 2):
        # Calculate the distance between the two points
        dist = distance(point1, point2)
        # Update max_dist if the current distance is greater
        if dist > max_dist:
            max_dist = dist
    return max_dist


def draw_interpolation_line_v2(ALL_POSITIONS, ax):
    LINES = {}
    for i in ALL_POSITIONS:
        points = ALL_POSITIONS[i]
        points_array = np.array(points)
        pca = PCA(n_components=1)
        pca.fit(points_array)
        line_direction = pca.components_[0]
        line_point = pca.mean_
        d = max_distance(points)
        t_min = -d / 2
        t_max = d / 2
        line_start = line_point + line_direction * t_min
        line_end = line_point + line_direction * t_max
        ax.plot(
            [line_start[0], line_end[0]],
            [line_start[1], line_end[1]],
            [line_start[2], line_end[2]],
            color=LINE_COLORS[i],
        )
        LINES[i] = [line_start, line_end]
    return ax, LINES


def draw_interpolation_line_v3(positions):
    points_array = np.array(positions)
    pca = PCA(n_components=1)
    pca.fit(points_array)
    line_direction = pca.components_[0]
    line_point = pca.mean_
    d = max_distance(positions)
    t_min = -d / 2
    t_max = d / 2
    line_start = line_point + line_direction * t_min
    line_end = line_point + line_direction * t_max

    return line_start, line_end


def find_closest_point(point_list, ref_point):
    points_array = np.array(point_list)
    ref_array = np.array(ref_point)
    distances = np.sum((points_array - ref_array)**2, axis=1)
    min_index = np.argmin(distances)
    return min_index
