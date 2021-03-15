'''

this file:
    run_demo_perception.py
    started 3/6/21

goes with:
    multi_layer_seqnn_brain.py
    offline_analyses/...

this is the multi layer, optimized, sped up newer version of:
    [X]     run_demo_predict.py
    [ ]     robot_brain_classes/two_stage_seqnn_brain.py
    [ ]     offline_analyses/try_sparsify_3.py

'''

import cv2
import random
import numpy as np
random.seed(6)
np.random.seed(6)

from robot_brain_classes.multi_layer_seqnn_brain import MultiLayerSeqNNBrain, AnotherBrain
from visualizer_classes.segment_visualizer import SegmentVisualizer
from sim_folder_manager import SimFolderManager
from robot_sensor_classes.video_playback import VideoPlaybackSensor

DISABLE_BRAIN = False  # for testing input by itself; also starts slow display
RF_IM_DIM = 32  # 16, 32, 64

def get_sensors_params():

    params_video_playback = {
        'image_dim': 256,  # video is cropped to square, then resized (scaled) to this square size. changing this resets preload of pickle.
        'output_image_dim': RF_IM_DIM,  # output image dim provided from class; must be smaller than or equal to 'image_dim'. affects how much of the image we actually use
        'output_image_start_RC': (128 - int(RF_IM_DIM/2), 128 - int(RF_IM_DIM/2)),  # relative to 'image_dim'. affects which subimage of the image we actually use
        'video_dir': '/srv/projects/video-downloads/',
        'video_filename': 'sea-turtles-yLuEx-XH3Uc.mp4',
        #'video_filename': 'seattle-driving-fkps18H3SXY.mp4',
        'stop_preload_at_frames': 1 * 60 * 60 * 30,  # None: use whole video
        'use_full_frame': True,  # use the whole image
        'partial_frame_factor': 4,  # from center, what factor to use - larger factor ~ smaller part of image
        'return_type': np.float32,  # 32 or 64
        'skip_frame_count': 1,  # Normal is 1 (not 0!); += K frames from video on each step; so we can see prediction better for high frame rate videos
        'prop_use_for_holdout': 0.0,
        'switch_to_holdout_frame': None  # TODO re-introduce later for when training is done
    }

    return params_video_playback


def get_sim_folder_manager_params():
    params = {
        'sim_prefix': 'test',
        'sim_folders_path': '/srv/projects/NL-sim/',
        'scripts_folder_path': '/srv/projects/NL/'
    }
    return params


def get_brain_params():
    brain_params = {
        'input_im_dim': RF_IM_DIM
    }

    return brain_params


def get_visualizer_params():
    params = {
        'color_enabled': False,
        'fps_display_interval': 3,
        'image_display_secs_fast': 3,  # 8+ for good speed
        'waitKey_time_fast': 1,  # 1, 100, 5000
        'image_display_secs_slow': 0.0,#2,  # 0: every frame
        'waitKey_time_slow': 1,  # 1, 100, 5000
        'scale_camera_factor': 1,
        'auto_switch_to_slow_disp_time': None,
        'init_fast': not DISABLE_BRAIN  # start with "fast" display if brain is enabled
    }
    return params


def init_demo():
    return {
        #'robot_brain': MultiLayerSeqNNBrain(get_brain_params()),
        'robot_brain': AnotherBrain(get_brain_params()),
        'robot_sensors': VideoPlaybackSensor(get_sensors_params()),
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

















