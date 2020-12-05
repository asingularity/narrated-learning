import cv2
import random
import numpy as np
random.seed(6)
np.random.seed(6)

from robot_brain_classes.wta_layer_brain import WTALayerBrain
from robot_brain_classes.wta_multi_layer_brain import WTAMultiLayerBrain
from robot_brain_classes.wta_iterative_brain import WTAIterativeBrain
from robot_brain_classes.wta_determinate_brain import WTADeterminateBrain
from robot_brain_classes.wta_remainder_brain import WTARemainderBrain
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
    LEARN_RATE = 0.0001 * 10
else:
    LEARN_RATE = 0.001  # * 0.25

IM_DIM = 16  # pixels, width & height

DISABLE_BRAIN = False  # for testing input by itself; also starts slow display

# uses: IM_DIM
def get_sensors_params():
    # TODO before using, fix im_dim like below
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
        'image_dim': 128,  # sensor class has to figure out subset & scale to achieve this dim
        'output_image_dim': IM_DIM,  # output image dim
        'output_image_start_RC': (50, 64),
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
        'take_subimage_factor': None, # 4,  # (None for don't use). Use an image this factor larger for same simulation (i.e. higher res sim), and take a sub-image of that larger image. this changes the input!
        'return_type': np.float64  # 32 or 64
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

    # # WTALayerBrain: DEPRECATED
    # params = {
    #     'image_dim_display': 1500,
    #     'image_dim_NxN_pixels': IM_DIM,
    #     'learning_rate': 0.001,
    #     'bins_per_pixel': 6,
    #     'num_rf': 10,
    #     'layer_start_times': np.array([0, 1, 2, 3, 4, 5]) * 100000,
    #     'max_time': 10000000,
    #     'learning_off_time': 6 * 100000,
    #     'plot_interval_seconds': 30
    # }

    # WTAMultiLayerBrain
    multilayer_brain_params = {
        'image_dim_display': 1500,
        'image_dim_NxN_pixels': IM_DIM,
        'learning_rate': 0.001 * 0.5,
        'bins_per_pixel': 6,
        'num_rf_per_wta_layer_per_hl': np.array([20, 80]),
        'num_wta_layer_per_hl': np.array([6, 12]),
        'input_time_steps_per_hl': np.array([1, 4]),
        'layer_start_time_offset_per_hl': np.array([6000, 4 * 6000]) * 5, # np.array([300000, 300000])
        'enable_learning_off': True,
        'plot_interval_seconds': 30,
        'enable_hack_skip_first_layer': False,
        'enable_generic_weights_viz': False
    }

    # WTAIterativeBrain
    iterative_brain_params = {
        'image_dim_display': int(2200 / 2),  # 2700: full on laptop, 3000: full on desktop
        'image_dim_NxN_pixels': IM_DIM,
        'learning_rate': LEARN_RATE,
        'bins_per_pixel': 6,
        'num_rf_per_hl': np.array([2 * 240, 2 * 240]),  #, 480]),  # np.array([240, 2 * 480])
        'num_iter_per_input': [6, 6],  # only for iterative, not used in determinate
        'input_num_time_steps_per_hl': np.array([1, 3]),  #, 3]),
        'input_delta_time_steps_per_hl': np.array([1, 5]),  #, 5]),
        'predict_ahead_time': 5,
        'start_time_per_hl': np.array([0, 10000]),  #50000]),
        'learning_off_time_per_hl': [10000, 5000000],  # 200000, ...
        'delta_t_start_per_wta_group': 5000,
        'plot_interval_seconds': 30,
        'enable_generic_weights_viz': False,
        'prediction_enabled': False
    }

    #return multilayer_brain_params
    return iterative_brain_params


def get_visualizer_params():
    params = {
        'color_enabled': COLOR_ENABLED,
        'fps_display_interval': 5,
        'image_display_secs_fast': 10,  # 8 for good speed
        'waitKey_time_fast': 1,  # 1, 100, 5000
        'image_display_secs_slow': 0.02, #0.001,  # 0: every frame
        'waitKey_time_slow': 100,  # 1, 100, 5000  #
        'scale_camera_factor': 1,
        'auto_switch_to_slow_disp_time': None, #250000, #50000,  # TODO re-introduce later for when training is done
        'init_fast': not DISABLE_BRAIN  # start with "fast" display if brain is enabled
    }
    return params


def init_demo():

    if USE_VIDEO_IN:
        robot_sensors = VideoPlaybackSensor(get_sensors_params())
        #robot_sensors = DuoPlaybackSensor(get_sensors_params())
    else:
        robot_sensors = Physics2DSensor(get_sensors_params())

    return {
        # *** DEPRECATED *** 'robot_brain': WTALayerBrain(get_brain_params()),
        #'robot_brain': WTAMultiLayerBrain(get_brain_params()),
        #'robot_brain': WTAIterativeBrain(get_brain_params()),
        'robot_brain': WTARemainderBrain(get_brain_params()),
        #'robot_brain': WTADeterminateBrain(get_brain_params()),
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

        if not DISABLE_BRAIN:
            robot_brain.process_input(input_im=im)

        visualizer.visualize(input_im=im,
                             segment_brain=robot_brain,
                             disable_brain=DISABLE_BRAIN)  # So it can call .get_table_ims() only sometimes

    robot_brain.save_model(models_save_folder=sim_folder_manager.get_models_save_folder())

    print ('Finished Evaluation.')

    while True:
        cv2.waitKey(1)


def demo():
    demo_components = init_demo()
    run_demo(demo_components)


if __name__ == '__main__':
    demo()
