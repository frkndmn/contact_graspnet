import heapq
import math
from math import sqrt
import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def n_closest_points_indices(points, target_point, n=3):
    # Calculate the distance from each point to the target point along with the index
    distances = [(calculate_distance(point, target_point), index) for index, point in enumerate(points)]

    # Find the n smallest distances (no need to take the square root as we're only comparing distances)
    closest_indices = heapq.nsmallest(n, distances)

    # Return only the indices
    return [index for (_, index) in closest_indices]


def quaternion_angular_difference(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return 2 * np.arccos(dot_product)


def find_most_different_orientation_couples(oris1, oris1_idx, oris2, oris2_idx):
    max_difference = 0
    most_different_couple_idx = (None, None)
    most_different_couple = (None, None)

    for q1_ix, q1 in enumerate(oris1):
        for q2_ix, q2 in enumerate(oris2):
            difference = quaternion_angular_difference(q1, q2)
            if difference > max_difference:
                max_difference = difference
                most_different_couple_idx = (oris1_idx[q1_ix], oris2_idx[q2_ix])
                most_different_couple = (q1, q2)

    return most_different_couple_idx, most_different_couple


def generate_spline_edges(line, positions, orientations, n_close=3):
    edge_1 = line[0]
    edge_2 = line[1]
    edge_1_position_indices = n_closest_points_indices(positions, edge_1, n=n_close)
    edge_2_position_indices = n_closest_points_indices(positions, edge_2, n=n_close)

    edge_1_orientations = [orientations[i] for i in edge_1_position_indices]
    edge_2_orientations = [orientations[i] for i in edge_2_position_indices]

    (edge_1_ori_idx, edge_2_ori_idx), (edge_1_ori, edge_2_ori) = find_most_different_orientation_couples(
        edge_1_orientations, edge_1_position_indices,
        edge_2_orientations, edge_2_position_indices,
    )

    return (
        (positions[edge_1_ori_idx], edge_1_ori),
        (positions[edge_2_ori_idx], edge_2_ori),
    )


# Define a function to interpolate between two positions
def lerp(position1, position2, t):
    return (1 - t) * np.array(position1) + t * np.array(position2)


# Define a function to interpolate between two quaternions using slerp
def slerp(quat1, quat2, t):
    cos_theta = np.dot(quat1, quat2)
    # If the dot product is negative, the quaternions have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if cos_theta < 0.0:
        quat1 = -quat1
        cos_theta = -cos_theta
    # Clamp the value to avoid numerical issues
    cos_theta = max(min(cos_theta, 1.0), -1.0)

    # If the quaternions are very close, lerp instead to avoid division by 0
    if cos_theta > 0.95:
        return (1 - t) * np.array(quat1) + t * np.array(quat2)

    # Compute the sin of the theta using trig identity
    theta = math.acos(cos_theta)
    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)

    # Compute the two factors to blend the input quaternions
    factor0 = math.sin((1 - t) * theta) / sin_theta
    factor1 = math.sin(t * theta) / sin_theta

    return np.add(np.multiply(quat1, factor0), np.multiply(quat2, factor1))


# Define a function to generate N new poses between two given poses
def generate_intermediate_poses(positions1, quaternions1, positions2, quaternions2, N):
    t_values = np.linspace(0, 1, N + 2)[1:-1]

    # Generate the new poses
    new_poses = []
    for t in t_values:
        new_position = lerp(positions1, positions2, t)
        new_orientation = slerp(quaternions1, quaternions2, t)
        new_poses.append(
            generate_matrix(new_position, new_orientation)
        )

    return new_poses


def generate_matrix(pos, ori):
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
