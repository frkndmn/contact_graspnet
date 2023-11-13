import mayavi.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

import mesh_utils

from agr_clustering import (
    grasping_poses_to_position_and_orientation,
    cluster_by_position,
    generate_indices_array,
)
from agr_spline import generate_intermediate_poses
from agr_interpolation import generate_interpolation_edges
from agr_helper import (
    COLORS,
    closest_position_index,
    save_interpolation_data,
    quaternions_to_euler,
)


def plot_mesh(mesh, cam_trafo=np.eye(4), mesh_pose=np.eye(4)):
    """
    Plots mesh in mesh_pose from

    Arguments:
        mesh {trimesh.base.Trimesh} -- input mesh, e.g. gripper

    Keyword Arguments:
        cam_trafo {np.ndarray} -- 4x4 transformation from world to camera coords (default: {np.eye(4)})
        mesh_pose {np.ndarray} -- 4x4 transformation from mesh to world coords (default: {np.eye(4)})
    """

    homog_mesh_vert = np.pad(mesh.vertices, (0, 1), "constant", constant_values=(0, 1))
    mesh_cam = homog_mesh_vert.dot(mesh_pose.T).dot(cam_trafo.T)[:, :3]
    mlab.triangular_mesh(
        mesh_cam[:, 0],
        mesh_cam[:, 1],
        mesh_cam[:, 2],
        mesh.faces,
        colormap="Blues",
        opacity=0.5,
    )


def plot_coordinates(t, r, tube_radius=0.005):
    """
    plots coordinate frame

    Arguments:
        t {np.ndarray} -- translation vector
        r {np.ndarray} -- rotation matrix

    Keyword Arguments:
        tube_radius {float} -- radius of the plotted tubes (default: {0.005})
    """
    mlab.plot3d(
        [t[0], t[0] + 0.2 * r[0, 0]],
        [t[1], t[1] + 0.2 * r[1, 0]],
        [t[2], t[2] + 0.2 * r[2, 0]],
        color=(1, 0, 0),
        tube_radius=tube_radius,
        opacity=1,
    )
    mlab.plot3d(
        [t[0], t[0] + 0.2 * r[0, 1]],
        [t[1], t[1] + 0.2 * r[1, 1]],
        [t[2], t[2] + 0.2 * r[2, 1]],
        color=(0, 1, 0),
        tube_radius=tube_radius,
        opacity=1,
    )
    mlab.plot3d(
        [t[0], t[0] + 0.2 * r[0, 2]],
        [t[1], t[1] + 0.2 * r[1, 2]],
        [t[2], t[2] + 0.2 * r[2, 2]],
        color=(0, 0, 1),
        tube_radius=tube_radius,
        opacity=1,
    )


def show_image(rgb, segmap):
    """
    Overlay rgb image with segmentation and imshow segment

    Arguments:
        rgb {np.ndarray} -- color image
        segmap {np.ndarray} -- integer segmap of same size as rgb
    """
    plt.figure()
    figManager = plt.get_current_fig_manager()

    plt.ion()
    plt.show()

    if rgb is not None:
        plt.imshow(rgb)
    if segmap is not None:
        cmap = plt.get_cmap("rainbow")
        cmap.set_under(alpha=0.0)
        plt.imshow(segmap, cmap=cmap, alpha=0.5, vmin=0.0001)
    plt.draw()
    plt.pause(0.001)


def visualize_grasps(
    full_pc,
    pred_grasps_cam,
    scores,
    plot_opencv_cam=False,
    pc_colors=None,
    gripper_openings=None,
    gripper_width=0.08,
):
    """Visualizes colored point cloud and predicted grasps. If given, colors grasps by segmap regions.
    Thick grasp is most confident per segment. For scene point cloud predictions, colors grasps according to confidence.

    Arguments:
        full_pc {np.ndarray} -- Nx3 point cloud of the scene
        pred_grasps_cam {dict[int:np.ndarray]} -- Predicted 4x4 grasp trafos per segment or for whole point cloud
        scores {dict[int:np.ndarray]} -- Confidence scores for grasps

    Keyword Arguments:
        plot_opencv_cam {bool} -- plot camera coordinate frame (default: {False})
        pc_colors {np.ndarray} -- Nx3 point cloud colors (default: {None})
        gripper_openings {dict[int:np.ndarray]} -- Predicted grasp widths (default: {None})
        gripper_width {float} -- If gripper_openings is None, plot grasp widths (default: {0.008})
    """

    print("Visualizing process started. It takes time.")

    fig = mlab.figure("Predicted Grasping Poses")
    mlab.view(azimuth=180, elevation=180, distance=0.2)
    draw_pc_with_colors(full_pc, pc_colors)

    if plot_opencv_cam:
        plot_coordinates(
            np.zeros(3,),
            np.eye(3, 3),
        )

    #
    # Adaptive Goal Region
    #

    positions, orientations, matrices = grasping_poses_to_position_and_orientation(pred_grasps_cam)

    # Cluster by Position
    labels = cluster_by_position(positions)
    n_clusters = len(set(labels))
    print(f"Estimated number of clusters: {n_clusters}")

    # Cluster by Orientation and Generate Unique Labels
    indices, global_labels = generate_indices_array(labels, n_clusters, positions, orientations)

    # Draw Spline
    adaptive_goal_region_data = []
    gripper_openings_k = np.ones(1) * gripper_width  # constant
    for parent_cluster in global_labels:
        for sub_cluster in global_labels[parent_cluster]:
            label = global_labels[parent_cluster][sub_cluster]
            sub_cluster_positions = np.array([
                positions[matrices_index]
                for matrices_index, global_label in indices
                if global_label == label
            ])
            if np.any(sub_cluster_positions):
                line_start, line_end = generate_interpolation_edges(sub_cluster_positions)
                ind_start = closest_position_index(positions, line_start)
                ind_end = closest_position_index(positions, line_end)

                line_start_pos = positions[ind_start]
                line_start_ori = orientations[ind_start]
                line_end_pos = positions[ind_end]
                line_end_ori = orientations[ind_end]

                adaptive_goal_region_data.append(
                    np.concatenate([
                        line_start_pos,
                        quaternions_to_euler(line_start_ori),
                        line_end_pos,
                        quaternions_to_euler(line_end_ori),
                    ])
                )

                interpolated = generate_intermediate_poses(
                    line_start_pos,
                    line_start_ori,
                    line_end_pos,
                    line_end_ori,
                    50,
                )
                for interpolated_matrix in interpolated:
                    draw_grasps(
                        [interpolated_matrix],
                        np.eye(4),
                        color=COLORS[label],
                        gripper_openings=gripper_openings_k,
                    )

    # Draw Raw Grasping Poses (Default: False)
    scatter = False
    if scatter:
        for global_index, color_index in indices:
            if color_index is None:
                continue
            color = COLORS[color_index]
            draw_grasps(
                [matrices[global_index]],
                np.eye(4),
                color=color,
                gripper_openings=gripper_openings_k,
            )

    save_interpolation_data(adaptive_goal_region_data)
    mlab.show()


def draw_pc_with_colors(
    pc,
    pc_colors=None,
    single_color=(0.3, 0.3, 0.3),
    mode="2dsquare",
    scale_factor=0.0018,
):
    """
    Draws colored point clouds

    Arguments:
        pc {np.ndarray} -- Nx3 point cloud
        pc_colors {np.ndarray} -- Nx3 point cloud colors

    Keyword Arguments:
        single_color {tuple} -- single color for point cloud (default: {(0.3,0.3,0.3)})
        mode {str} -- primitive type to plot (default: {'point'})
        scale_factor {float} -- Scale of primitives. Does not work for points. (default: {0.002})

    """

    if pc_colors is None:
        mlab.points3d(
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            color=single_color,
            scale_factor=scale_factor,
            mode=mode,
        )
    else:
        # create direct grid as 256**3 x 4 array
        def create_8bit_rgb_lut():
            xl = np.mgrid[0:256, 0:256, 0:256]
            lut = np.vstack(
                (
                    xl[0].reshape(1, 256**3),
                    xl[1].reshape(1, 256**3),
                    xl[2].reshape(1, 256**3),
                    255 * np.ones((1, 256**3)),
                )
            ).T
            return lut.astype("int32")

        scalars = pc_colors[:, 0] * 256**2 + pc_colors[:, 1] * 256 + pc_colors[:, 2]
        rgb_lut = create_8bit_rgb_lut()
        points_mlab = mlab.points3d(
            pc[:, 0], pc[:, 1], pc[:, 2], scalars, mode=mode, scale_factor=0.0018
        )
        points_mlab.glyph.scale_mode = "scale_by_vector"
        points_mlab.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(
            0, rgb_lut.shape[0]
        )
        points_mlab.module_manager.scalar_lut_manager.lut.number_of_colors = (
            rgb_lut.shape[0]
        )
        points_mlab.module_manager.scalar_lut_manager.lut.table = rgb_lut


def draw_grasps(
    grasps,
    cam_pose,
    gripper_openings,
    color=(0, 1.0, 0),
    colors=None,
    show_gripper_mesh=False,
    tube_radius=0.0008,
):
    """
    Draws wireframe grasps from given camera pose and with given gripper openings

    Arguments:
        grasps {np.ndarray} -- Nx4x4 grasp pose transformations
        cam_pose {np.ndarray} -- 4x4 camera pose transformation
        gripper_openings {np.ndarray} -- Nx1 gripper openings

    Keyword Arguments:
        color {tuple} -- color of all grasps (default: {(0,1.,0)})
        colors {np.ndarray} -- Nx3 color of each grasp (default: {None})
        tube_radius {float} -- Radius of the grasp wireframes (default: {0.0008})
        show_gripper_mesh {bool} -- Renders the gripper mesh for one of the grasp poses (default: {False})
    """

    gripper = mesh_utils.create_gripper("panda")
    gripper_control_points = gripper.get_control_point_tensor(
        1, False, convex_hull=False
    ).squeeze()
    mid_point = 0.5 * (gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array(
        [
            np.zeros((3,)),
            mid_point,
            gripper_control_points[1],
            gripper_control_points[3],
            gripper_control_points[1],
            gripper_control_points[2],
            gripper_control_points[4],
        ]
    )

    if show_gripper_mesh and len(grasps) > 0:
        plot_mesh(gripper.hand, cam_pose, grasps[0])

    all_pts = []
    connections = []
    index = 0
    N = 7
    for i, (g, g_opening) in enumerate(zip(grasps, gripper_openings)):
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:, 0] = (
            np.sign(grasp_line_plot[2:, 0]) * g_opening / 2
        )

        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((7, 1))), axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:, :3]

        # color = color if colors is None else colors[i]

        all_pts.append(pts)
        connections.append(
            np.vstack(
                [
                    np.arange(index, index + N - 1.5),
                    np.arange(index + 1, index + N - 0.5),
                ]
            ).T
        )
        index += N
        # mlab.plot3d(pts[:, 0], pts[:, 1], pts[:, 2], color=color, tube_radius=tube_radius, opacity=1.0)

    # speeds up plot3d because only one vtk object
    all_pts = np.vstack(all_pts)
    connections = np.vstack(connections)
    src = mlab.pipeline.scalar_scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2])
    src.mlab_source.dataset.lines = connections
    src.update()
    lines = mlab.pipeline.tube(src, tube_radius=tube_radius, tube_sides=12)
    mlab.pipeline.surface(lines, color=color, opacity=1.0)


def draw_line(line_start, line_end, color=(0, 0, 0)):
    x = [line_start[0], line_end[0]]
    y = [line_start[1], line_end[1]]
    z = [line_start[2], line_end[2]]
    mlab.plot3d(x, y, z, tube_radius=0.01, color=color)  # tube_radius=None for a 1D line
