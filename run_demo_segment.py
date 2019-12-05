import cv2
import random
import numpy as np
random.seed(6)
np.random.seed(6)

from segment_brain import SegmentBrain
from duo_sensor import DuoSensor
from segment_visualizer import SegmentVisualizer
from sim_folder_manager import SimFolderManager
import time
from math import pi


MAX_HISTORY_LENGTH = 10600000 + 1
USERNAME = 'intec'

TABLE_ENTRIES = 8000  # 8000
TABLE_LEARN_TIME = TABLE_ENTRIES * 16
PREDICTION_LEARN_TIME = TABLE_ENTRIES * 16

IM_DIM = 32  # assume square image, this is width & height
IM_PIXELS = IM_DIM * IM_DIM  # assume grayscale


def get_sensors_params():
    params = {
        'image_dim': IM_DIM  # sensor class has to figure out subset & scale to achieve this dim
    }
    return params


def get_sim_folder_manager_params():
    params = {
        'sim_prefix': 'test',
        'sim_folders_path': '/srv/projects/NL-sim/',
        'scripts_folder_path': '/srv/projects/NL/'
    }
    return params


def get_brain_params():
    params = {
        # ************ general ************
        'max_history_length': MAX_HISTORY_LENGTH,
        'enable_learning': True,

        # ************ MultiLayerSharedTiles ************
        'use_multi_layer': True,
        'tile_entries_per_layer': [TABLE_ENTRIES],
        'table_learn_time_per_layer': [TABLE_LEARN_TIME],
        'prediction_learn_time_per_layer': [PREDICTION_LEARN_TIME],
        'tiles_per_layer_NxN': [4],  # N where tiled NxN
        'input_dim_per_tile': IM_PIXELS / (4 * 4),  # (32 * 32 * 3.) / (4 * 4.) = 192.0
            }

    return params


def get_visualizer_params():
    params = {
        'fps_display_interval': 3,
        'image_display_secs_fast': 5,
        'waitKey_time_fast': 1,  # 1, 100, 5000
        'image_display_secs_slow': 0,  # 0: every frame
        'waitKey_time_slow': 1,  # 1, 100, 5000  #
        'scale_camera_factor': 20,
        'auto_switch_to_slow_disp_time': None,
        'init_fast': True  # start with "fast" display
    }
    return params


def init_demo():
    return {
        'robot_brain': SegmentBrain(get_brain_params()),
        'robot_sensors': DuoSensor(get_sensors_params()),
        'visualizer': SegmentVisualizer(get_visualizer_params()),
        'sim_folder_manager': SimFolderManager(get_sim_folder_manager_params())
    }


def run_demo(demo_components):
    robot_brain = demo_components['robot_brain']
    robot_sensors = demo_components['robot_sensors']
    visualizer = demo_components['visualizer']

    random.seed(1233)

    while True:

        im = robot_sensors.read_input()

        robot_brain.process_input(input_im=im)

        visualizer.visualize(input_im=im,
                             segment_brain=robot_brain)  # So it can call .get_table_ims() only sometimes

    robot_brain.save_model(models_save_folder=sim_folder_manager.get_models_save_folder())

    print ('Finished Evaluation.')

    while True:
        cv2.waitKey(1)


def demo():
    demo_components = init_demo()
    run_demo(demo_components)


if __name__ == '__main__':
    demo()
