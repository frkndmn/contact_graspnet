import json
import os
import sys
import numpy as np

from contact_grasp_estimator import GraspEstimator
from data import load_available_input_data
from flask import Flask, request, send_file
import tensorflow.compat.v1 as tf


tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

current_dir = os.path.dirname(__file__)
global_config_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'global_config.json'), 'r+')
upload_save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)))
global_config = json.loads(global_config_file.read())

checkpoint_dir = 'checkpoints/scene_test_2048_bs3_hor_sigma_001'
z_range = [0.2, 1.1]
local_regions = False
filter_grasps = False
skip_border_objects = False
forward_passes = 5
app = Flask(__name__)

# Build the model
grasp_estimator = GraspEstimator(global_config)
grasp_estimator.build_network()
saver = tf.train.Saver(save_relative_paths=True)

# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

# Load weights
grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')


@app.route('/run')
def read_file():
    path = request.args.get('path')
    file = request.files['file']
    if path:
        if os.path.exists(path) and os.path.isfile(path):
            print('Loading ', path)
        else:
            return f"File not found at path: {path}"
    elif file:
        path = upload_save_dir + file.filename
        file.save(path)
        print("Sent file saved as: ", path)
    else:
        return "Please provide a valid 'path' parameter."

    pc_segments = {}
    segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(path, K=None)

    if segmap is None and (local_regions or filter_grasps):
        raise ValueError('Need segmentation map to extract local regions or filter grasps')

    if pc_full is None:
        print('Converting depth to point cloud(s)...')
        pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
            depth,
            cam_K,
            segmap=segmap,
            rgb=rgb,
            skip_border_objects=skip_border_objects,
            z_range=z_range,
        )

    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
        sess,
        pc_full,
        pc_segments=pc_segments,
        local_regions=local_regions,
        filter_grasps=filter_grasps,
        forward_passes=forward_passes,
    )
    file_name = 'predictions_{}'.format(
        os.path.basename(path.replace('png', 'npz').replace('npy', 'npz'))
    )
    save_path = os.path.abspath(
        os.path.join(current_dir, '..', '..', '..', 'assets', 'grasping_pose_results', file_name)
    )

    np.savez(
        save_path,
        pred_grasps_cam=pred_grasps_cam,
        scores=scores,
        contact_pts=contact_pts,
    )

    # Visualize results
    # show_image(rgb, segmap)
    # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
    if file:
        print(f"File sent, temporarily saved on: {save_path}")
        return send_file(save_path, as_attachment=True)
    else:
        print(f"Saved on: {save_path}")
        return save_path


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6161)
