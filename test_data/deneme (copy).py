import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import glob
import open3d as o3d

from PIL import Image
import numpy as np

loaded_data_dict = np.load(
    "predictions_franka_gazebo.npz",
    allow_pickle=True,
    encoding="bytes",
)
N = len(loaded_data_dict)
for i in range(N):
    rgb_img = loaded_data_dict["rgb"]
    depth_img = loaded_data_dict["depth"]
    cm_pr = loaded_data_dict["K"]
    plt.figure("Welcome to figure 1")
    plt.imshow(depth_img)
    plt.figure("Welcome to figure 2")
    plt.imshow(rgb_img)
    print("d")


# example_images_dir = os.path.abspath(".") + "/test_data/"

# OSD_image_files = sorted(glob.glob(example_images_dir + "franka_real.npy"))
# # Load the entire dictionary from the .npy file
# N = len(OSD_image_files)
# for i, img_file in enumerate(OSD_image_files):
#     loaded_data_dict = np.load(
#         img_file,
#         allow_pickle=True,
#         encoding="bytes",
#     ).item()
#     arr1 = loaded_data_dict["rgb"]
#     plt.imshow(arr1)
#     arr22 = loaded_data_dict["depth"]
#     plt.imshow(arr22)
#     # dict2 = loaded_data_dict["xyz"]
#     arr3 = loaded_data_dict["K"]


# print("hello")
