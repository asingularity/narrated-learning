import cv2
import random
import numpy as np
random.seed(6)
np.random.seed(6)

from robot_brain_classes.exp_brain import ExpBrain
from robot_sensor_classes.video_playback import VideoPlaybackSensor
from robot_sensor_classes.duo_playback import DuoPlaybackSensor
from robot_sensor_classes.physics_2d import Physics2DSensor
from visualizer_classes.segment_visualizer import SegmentVisualizer
from sim_folder_manager import SimFolderManager
import time
from math import pi


MAX_HISTORY_LENGTH = 10600000 + 1
USERNAME = 'intec'

COLOR_ENABLED = False

USE_VIDEO_IN = False

if USE_VIDEO_IN:
    IM_DIM = 128  # pixels, width & height
else:
    IM_DIM = 16  # pixels, width & height


# uses: IM_DIM
def get_sensors_params():
    # params_video_playback = {
    #     'image_dim': IM_DIM,  # sensor class has to figure out subset & scale to achieve this dim
    #     'video_dir': '/srv/projects/NL-data/',
    #     'video_filename': 'videoplayback',  # 3840x2160
    #     # 'video_filename': 'P1033727.mp4',  # 3840x2160
    #     'stop_preload_at_frames': None,  # None: use whole video
    #     'use_full_frame': False,  # use the whole image
    #     'partial_frame_factor': 4,  # from center, what factor to use - larger factor ~ smaller part of image
    #     'return_type': np.float32,  # 32 or 64
    #     'skip_frame_count': 1,  # Normal is 1 (not 0!); += K frames from video on each step; so we can see prediction better for high frame rate videos
    #     'prop_use_for_holdout': 0.0,
    #     'switch_to_holdout_frame': None  # TODO re-introduce later for when training is done
    # }

    params_video_playback = {
        'image_dim': IM_DIM,  # sensor class has to figure out subset & scale to achieve this dim
        'video_dir': '/srv/projects/NL-data/',
        'video_filename': 'DSC_0446.MOV',  # 32300 frames
        # 'video_filename': 'P1033727.mp4',  # 3840x2160
        'stop_preload_at_frames': None,  # None: use whole video
        'use_full_frame': True,  # use the whole image
        'partial_frame_factor': 4,  # from center, what factor to use - larger factor ~ smaller part of image
        'return_type': np.float32,  # 32 or 64
        'skip_frame_count': 1,  # Normal is 1 (not 0!); += K frames from video on each step; so we can see prediction better for high frame rate videos
        'prop_use_for_holdout': 0.0,
        'switch_to_holdout_frame': None  # TODO re-introduce later for when training is done
    }

    params_physics_2d = {
        'image_dim': IM_DIM,  # sensor class has to figure out subset & scale to achieve this dim
        'return_type': np.float32  # 32 or 64
    }

    params_duo_playback = {
        'image_dim': IM_DIM,  # sensor class has to figure out subset & scale to achieve this dim
        'video_dir': '/srv/projects/NL-data/',
        'video_filename': 'DUOCapture-19-07-2020-14-26-57-330_20k_frames.avi',  #
        'partial_frame_factor': 0.4,  # this prop of inside of frame
        'return_type': np.float32,  # 32 or 64
        'skip_frame_count': 1,  # Normal is 1 (not 0!); += K frames from video on each step; so we can see prediction better for high frame rate videos
        'prop_use_for_holdout': 0.3,
        'switch_to_holdout_frame': None
    }

    if USE_VIDEO_IN:
        #params = params_duo_playback
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

    params = {
        'image_dim_NxN_pixels': IM_DIM,
        'use_full_input': not USE_VIDEO_IN
    }

    return params


def get_visualizer_params():
    params = {
        'color_enabled': COLOR_ENABLED,
        'fps_display_interval': 6,
        'image_display_secs_fast': 0.2, #0.2,
        'waitKey_time_fast': 1,  # 1, 100, 5000
        'image_display_secs_slow': 0.01,  # 0: every frame
        'waitKey_time_slow': 1,  # 1, 100, 5000  #
        'scale_camera_factor': 1,
        'auto_switch_to_slow_disp_time': None, #50000,  # TODO re-introduce later for when training is done
        'init_fast': True  # start with "fast" display
    }
    return params


def init_demo():

    if USE_VIDEO_IN:
        robot_sensors = VideoPlaybackSensor(get_sensors_params())
        #robot_sensors = DuoPlaybackSensor(get_sensors_params())
    else:
        robot_sensors = Physics2DSensor(get_sensors_params())

    return {
        'robot_brain': ExpBrain(get_brain_params()),
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
