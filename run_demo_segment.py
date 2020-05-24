import cv2
import random
import numpy as np
random.seed(6)
np.random.seed(6)

from robot_brain_classes.segment_brain import SegmentBrain
from robot_sensor_classes.video_playback import VideoPlaybackSensor
from robot_sensor_classes.physics_2d import Physics2DSensor
from visualizer_classes.segment_visualizer import SegmentVisualizer
from sim_folder_manager import SimFolderManager
import time
from math import pi


MAX_HISTORY_LENGTH = 10600000 + 1
USERNAME = 'intec'

IM_DIM = 32  #64  # 128, # assume square image, this is width & height
TILES_LAYER_0 = 1  # N where tiled NxN
IM_PIXELS = IM_DIM * IM_DIM  # assume grayscale

TABLE_ENTRIES = 10 * 10  # 4 * 4  # 10 * 10  # 20 * 20
TABLE_LEARN_TIME = 3000  # 3000; TABLE_ENTRIES
PREDICTION_LEARN_TIME = TABLE_ENTRIES * 160

COLOR_ENABLED = False

ENABLE_WEIGHT_BIAS = True

USE_VIDEO_IN = False


def get_sensors_params():
    params_video_playback = {
        'image_dim': IM_DIM,  # sensor class has to figure out subset & scale to achieve this dim
        'video_dir': '/srv/projects/NL-data/',
        'video_filename': 'videoplayback',  # 3840x2160
        # 'video_filename': 'P1033727.mp4',  # 3840x2160
        'stop_preload_at_frames': None,  # None: use whole video
        'use_full_frame': False,  # use the whole image
        'partial_frame_factor': 4,  # from center, what factor to use - larger factor ~ smaller part of image
        'return_type': np.float32  # 32 or 64
    }

    params_physics_2d = {
        'image_dim': IM_DIM,  # sensor class has to figure out subset & scale to achieve this dim
        'return_type': np.float32  # 32 or 64
    }

    if USE_VIDEO_IN:
        params = params_video_playback
    else:
        params = params_physics_2d

    return params


def get_sim_folder_manager_params():
    params = {
        'sim_prefix': 'test',
        'sim_folders_path': '/srv/projects/NL-sim/',
        'scripts_folder_path': '/srv/projects/NL/'
    }
    return params


def get_brain_params():

    if COLOR_ENABLED:
        input_dim = IM_PIXELS * 3 / (TILES_LAYER_0 * TILES_LAYER_0)  # (32 * 32 * 3.) / (4 * 4.) = 192.0
    else:
        input_dim = IM_PIXELS / (TILES_LAYER_0 * TILES_LAYER_0)

    input_dim = int(input_dim)

    params = {
        # ************ general ************
        'max_history_length': MAX_HISTORY_LENGTH,
        'enable_learning': True,

        # ************ SimpleWeightTiles ************
        'color_enabled': COLOR_ENABLED,
        'enable_weight_bias': ENABLE_WEIGHT_BIAS,
        'use_multi_layer': True,
        'tile_entries_per_layer': [TABLE_ENTRIES],
        'table_learn_time_per_layer': [TABLE_LEARN_TIME],
        'prediction_learn_time_per_layer': [PREDICTION_LEARN_TIME],
        'tiles_per_layer_NxN': [TILES_LAYER_0],  # N where tiled NxN
        'input_dim': input_dim,
        'table_ims_scale_pixels': 1200  # 2000 for 4k monitor, 1600 for laptop
            }

    return params


def get_visualizer_params():
    params = {
        'color_enabled': COLOR_ENABLED,
        'fps_display_interval': 6,
        'image_display_secs_fast': 1,
        'waitKey_time_fast': 1,  # 1, 100, 5000
        'image_display_secs_slow': 0,  # 0: every frame
        'waitKey_time_slow': 100,  # 1, 100, 5000  #
        'scale_camera_factor': 1,
        'auto_switch_to_slow_disp_time': None,
        'init_fast': True  # start with "fast" display
    }
    return params


def init_demo():

    if USE_VIDEO_IN:
        robot_sensors = VideoPlaybackSensor(get_sensors_params())
    else:
        robot_sensors = Physics2DSensor(get_sensors_params())

    return {
        'robot_brain': SegmentBrain(get_brain_params()),
        'robot_sensors': robot_sensors,
        'visualizer': SegmentVisualizer(get_visualizer_params()),
        'sim_folder_manager': SimFolderManager(get_sim_folder_manager_params())
    }

def run_demo(demo_components):
    robot_brain = demo_components['robot_brain']
    robot_sensors = demo_components['robot_sensors']
    visualizer = demo_components['visualizer']
    sim_folder_manager = demo_components['sim_folder_manager']

    random.seed(1233)

    while True:

        # reference
        # a = np.dot(np.random.random((200, 200)), np.random.random((200, 200)))

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
