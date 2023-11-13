import math
from typing import Tuple

import numpy as np

from agr_helper import (
    n_closest_points_indices,
    quaternion_angular_distance
)


def find_most_different_orientation_couples(oris1: np.ndarray, oris1_idx: np.ndarray, oris2: np.ndarray, oris2_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_difference = 0
    most_different_couple_idx = (None, None)
    most_different_couple = (None, None)

    for q1_ix, q1 in enumerate(oris1):
        for q2_ix, q2 in enumerate(oris2):
            difference = quaternion_angular_distance(q1, q2)
            if difference > max_difference:
                max_difference = difference
                most_different_couple_idx = (oris1_idx[q1_ix], oris2_idx[q2_ix])
                most_different_couple = (q1, q2)

    return np.array(most_different_couple_idx), np.array(most_different_couple)


def generate_spline_edges(line: np.ndarray, positions: np.ndarray, orientations: np.ndarray, n_close: int = 3) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    edge_1 = line[0]
    edge_2 = line[1]
    edge_1_position_indices = n_closest_points_indices(positions, edge_1, n=n_close)
    edge_2_position_indices = n_closest_points_indices(positions, edge_2, n=n_close)

    edge_1_orientations = np.array([orientations[i] for i in edge_1_position_indices])
    edge_2_orientations = np.array([orientations[i] for i in edge_2_position_indices])

    (edge_1_ori_idx, edge_2_ori_idx), (edge_1_ori, edge_2_ori) = find_most_different_orientation_couples(
        edge_1_orientations, edge_1_position_indices,
        edge_2_orientations, edge_2_position_indices,
    )

    return (positions[edge_1_ori_idx], edge_1_ori), (positions[edge_2_ori_idx], edge_2_ori)


def lerp(position1: np.ndarray, position2: np.ndarray, t: float) -> np.ndarray:
    return (1 - t) * np.array(position1) + t * np.array(position2)


def slerp(quat1: np.ndarray, quat2: np.ndarray, t: float) -> np.ndarray:
    cos_theta = np.dot(quat1, quat2)
    if cos_theta < 0.0:
        quat1 = -quat1
        cos_theta = -cos_theta
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    if cos_theta > 0.95:
        return (1 - t) * np.array(quat1) + t * np.array(quat2)
    theta = math.acos(cos_theta)
    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
    factor0 = math.sin((1 - t) * theta) / sin_theta
    factor1 = math.sin(t * theta) / sin_theta
    return np.add(np.multiply(quat1, factor0), np.multiply(quat2, factor1))


def generate_intermediate_poses(positions1: np.ndarray, quaternions1: np.ndarray, positions2: np.ndarray, quaternions2: np.ndarray, n: int = 50) -> np.ndarray:
    t_values = np.linspace(0, 1, n + 2)[1:-1]
    new_poses = []
    for t in t_values:
        new_position = lerp(positions1, positions2, t)
        new_orientation = slerp(quaternions1, quaternions2, t)
        new_poses.append(
            generate_matrix(new_position, new_orientation)
        )
    return np.array(new_poses)


def generate_matrix(pos: np.ndarray, ori: np.ndarray) -> np.ndarray:
    w, x, y, z = ori
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rotation_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy), pos[0]],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx), pos[1]],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy), pos[2]],
        [0, 0, 0, 1]
    ])
    return rotation_matrix
