import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import glob
import open3d as o3d


def build_matrix_of_indices(height, width):
    """Builds a [height, width, 2] numpy array containing coordinates.

    @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)


def compute_xyz(depth_img, camera_params):
    """Compute ordered point cloud from depth image and camera parameters.
    Assumes camera uses left-handed coordinate system, with
        x-axis pointing right
        y-axis pointing up
        z-axis pointing "forward"

    @param depth_img: a [H x W] numpy array of depth values in meters
    @param camera_params: a dictionary with parameters of the camera used

    @return: a [H x W x 3] numpy array
    """

    # Compute focal length from camera parameters

    fx = camera_params[0, 0]
    fy = camera_params[1, 1]

    x_offset = 640 / 2
    y_offset = 480 / 2

    indices = build_matrix_of_indices(480, 640)
    indices[..., 0] = np.flipud(
        indices[..., 0]
    )  # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]

    return xyz_img


example_images_dir = os.path.abspath(".") + "/test_data/"

OSD_image_files = sorted(glob.glob(example_images_dir + "/franka_real.npy"))
# Load the entire dictionary from the .npy file
N = len(OSD_image_files)
rgb_imgs = np.zeros((N, 720, 1280, 3), dtype=np.float32)
xyz_imgs = np.zeros((N, 720, 1280, 3), dtype=np.float32)
label_imgs = np.zeros((N, 720, 1280), dtype=np.uint8)
for i, img_file in enumerate(OSD_image_files):
    loaded_data_dict = np.load(
        img_file,
        allow_pickle=True,
        encoding="bytes",
    ).item()
    arr1 = loaded_data_dict["rgb"]
    dict2 = loaded_data_dict["depth"]

    # plt.imshow(arr1)

    # plt.imshow(dict2)
    # dict2 = loaded_data_dict["xyz"]
    # K = loaded_data_dict["K"]
    # dict2 = np.asarray(dict2)
    # asd = compute_xyz(dict2, K)

    img_rgb = o3d.geometry.Image((arr1 * 255).astype(np.uint8))
    # asd = o3d.geometry.Image((asd * 255).astype(np.uint8))

    # data_dict = {
    #    "rgb": np.array(img_rgb),
    #   "xyz": np.array(asd),
    #    "label": label_imgs,
    #    "seg": segmentation_map,
    # }
# np.save("/home/furkan/uois/example_images/franka_xyz.npy", data_dict)
# o3d.visualization.draw_geometries([asd])

# Convert point cloud to numpy array
# xyz = np.asarray(asd)

#    Reverse y values
# xyz[:, 1] *= -1

# Create XYZ image
# xyz_image = o3d.geometry.Image(xyz)

# Convert to point cloud and visualize
# pcd = o3d.geometry.PointCloud.create_from_depth_image(xyz_image, intrinsics)
# o3d.visualization.draw_geometries([pcd])


print("hello")
