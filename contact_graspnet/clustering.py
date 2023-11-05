import csv
import ast
from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib as mpl

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform

from scipy.spatial.transform import Rotation
import numpy as np


def quaternions_to_euler(quaternions):
    quaternions_array = np.array(quaternions)
    rotation = Rotation.from_quat(quaternions_array)
    euler_angles = rotation.as_euler("xyz", degrees=True)
    euler_list = euler_angles.tolist()
    return euler_list


def euler_to_quaternion(yaw, pitch, roll) -> Tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return w, x, y, z


def quaternion_rotation_matrix(Q) -> np.ndarray:
    w, x, y, z = Q
    R = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )
    return R


def quaternion_angular_distance(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    dot = np.clip(dot, -1.0, 1.0)
    angle = 2 * np.arccos(np.abs(dot))
    return angle


def cluster_quaternions(orientations, global_indices, eps_degrees=40, min_samples=5):
    distance_matrix = squareform(
        pdist(orientations, metric=quaternion_angular_distance)
    )
    db = DBSCAN(
        eps=np.radians(eps_degrees), min_samples=min_samples, metric="precomputed"
    ).fit(distance_matrix)
    labels = db.labels_
    # Calculate the number of clusters, ignoring noise if present
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Estimated number of clusters of orientation:", n_clusters)

    cluster_indices = []
    for index, label in enumerate(labels):
        if label != -1:  # Check to make sure you're not looking at noise
            cluster_indices.append((global_indices[index], label))
        else:
            cluster_indices.append((global_indices[index], None))
    return cluster_indices


def grasping_points_csv_to_dict(path="grasping_points.csv"):
    data = []

    with open("grasping_points_new.csv", mode="r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row if it exists

        for row in csv_reader:
            # Use ast.literal_eval to safely convert the string representation of a list into an actual list
            pred_grasps_cam = ast.literal_eval(row[0])
            scores = float(row[1])
            contact_pts = ast.literal_eval(row[2])

            # Now you can work with the data as lists and floats
            data.append(
                {
                    "pred_grasps_cam": pred_grasps_cam,
                    "scores": scores,
                    "contact_pts": contact_pts,
                }
            )
            # Create a 3D scatter plot
    return data


def grasping_points_to_position_and_orientation(data) -> Tuple[np.ndarray, np.ndarray]:
    positions = []
    orientations = []

    for item in data:
        contact_pts = item["contact_pts"]
        pred_grasps_cam = item["pred_grasps_cam"]
        x, y, z = contact_pts
        x = float(x)
        y = float(y)
        z = float(z)
        # ax.scatter(x, y, z, c="b", marker="o")
        position = np.array([x, y, z])
        pitch = math.asin(-float(pred_grasps_cam[2][0]))
        yaw = math.atan2(float(pred_grasps_cam[1][0]), float(pred_grasps_cam[0][0]))
        roll = math.atan2(float(pred_grasps_cam[2][1]), float(pred_grasps_cam[2][2]))
        # orientation = np.array([roll, pitch, yaw])
        quaternion = euler_to_quaternion(yaw, pitch, roll)
        positions.append(position)
        orientations.append(quaternion)

    return positions, orientations


def grasping_points_to_position_and_orientation_v2(pred_grasps_cam):
    positions = []
    orientations = []
    matrices = []

    for i in pred_grasps_cam[-1]:
        x = i[0][3]
        y = i[1][3]
        z = i[2][3]
        # ax.scatter(x, y, z, c="b", marker="o")
        position = np.array([x, y, z])
        pitch = math.asin(-float(i[2][0]))
        yaw = math.atan2(float(i[1][0]), float(i[0][0]))
        roll = math.atan2(float(i[2][1]), float(i[2][2]))
        # orientation = np.array([roll, pitch, yaw])
        quaternion = euler_to_quaternion(yaw, pitch, roll)
        positions.append(position)
        orientations.append(quaternion)
        matrices.append(i)

    return positions, orientations, matrices
