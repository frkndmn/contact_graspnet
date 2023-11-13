import math
import numpy as np
from typing import Tuple, Dict

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from agr_helper import (
    quaternion_angular_distance,
    euler_to_quaternion
)


def cluster_by_position(positions: np.ndarray) -> np.ndarray:
    db = DBSCAN(eps=0.05, min_samples=5).fit(positions)
    labels = db.labels_
    return np.array(labels)


def cluster_by_orientation(orientations: np.ndarray, global_indices: np.ndarray, eps_degrees: int = 40, min_samples: int = 5):
    distance_matrix = squareform(pdist(orientations, metric=quaternion_angular_distance))
    db = DBSCAN(
        eps=np.radians(eps_degrees), min_samples=min_samples, metric="precomputed"
    ).fit(distance_matrix)
    labels = db.labels_
    print(f"Estimated number of clusters of orientation: {len(set(labels)) - (1 if -1 in labels else 0)}")
    cluster_indices = []
    for index, label in enumerate(labels):
        if label != -1:
            cluster_indices.append((global_indices[index], label))
        else:
            cluster_indices.append((global_indices[index], None))
    return cluster_indices


def grasping_poses_to_position_and_orientation(pred_grasps_cam: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    orientations = []
    matrices = []

    for i in pred_grasps_cam[-1]:
        x = i[0][3]
        y = i[1][3]
        z = i[2][3]
        position = np.array([x, y, z])
        pitch = math.asin(-float(i[2][0]))
        yaw = math.atan2(float(i[1][0]), float(i[0][0]))
        roll = math.atan2(float(i[2][1]), float(i[2][2]))
        quaternion = euler_to_quaternion(yaw, pitch, roll)
        positions.append(position)
        orientations.append(quaternion)
        matrices.append(np.array(i))

    return np.array(positions), np.array(orientations), np.array(matrices)


def generate_indices_array(labels: np.ndarray, n_clusters: int, positions: np.ndarray, orientations: np.ndarray) -> Tuple[np.ndarray, Dict]:
    global_labels = {}
    indices = []
    last_index = 0
    for parent_cluster in range(n_clusters):
        cluster_orientations_indices = []
        cluster_positions = []
        cluster_orientations = []
        for index, l in enumerate(labels):
            if l == parent_cluster:
                cluster_orientations.append(orientations[index])
                cluster_orientations_indices.append(index)
                cluster_positions.append(positions[index])

        if not np.any(cluster_orientations):
            continue

        sub_cluster_indices = cluster_by_orientation(
            np.array(cluster_orientations),
            np.array(cluster_orientations_indices),
        )

        for j, (matrices_index, sub_cluster) in enumerate(sub_cluster_indices):
            if sub_cluster is None:
                global_label = None
            else:
                global_label = global_labels.get(parent_cluster, {}).get(sub_cluster, None)
                if global_label is None:
                    global_label = last_index
                    if parent_cluster not in global_labels:
                        global_labels[parent_cluster] = {}
                    global_labels[parent_cluster][sub_cluster] = global_label
                    last_index += 1
            indices.append((matrices_index, global_label))

    return indices, global_labels
