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

#from robot_brain_classes.multi_layer_seqnn_brain import MultiLayerSeqNNBrain
#from robot_brain_classes.tiled_multilayer_wta import SeqNNSeqKMeansBrain
#from robot_brain_classes.batch_iter_wta import BatchIterWTABrain
from robot_brain_classes.rate_control_wta import RateControlWTABRain
from robot_brain_classes.rate_control_som_wta import RateControlSomWtaBrain
from visualizer_classes.segment_visualizer import SegmentVisualizer
from sim_folder_manager import SimFolderManager
from robot_sensor_classes.video_playback import VideoPlaybackSensor
from robot_preprocess_classes.event_pre_processor import EventPreProcessor


DISABLE_BRAIN = False  # for testing input by itself; also starts slow display
RF_IM_DIM = 8  # 8, 16, 32, 64

ROOT_DIR = '/srv'
# ROOT_DIR = '/home/csaba'


def get_sensors_params():

    # 'video_filename': 'seattle-driving-fkps18H3SXY.mp4',

    square_crop = (128 - int(RF_IM_DIM / 2), 128 - int(RF_IM_DIM / 2), RF_IM_DIM)

    params_video_playback = {
        'pickle_im_square_crop': None,
        'pickle_im_square_resize': 256,
        'force_reload_to_pkl': False,
        'gb_per_pkl_file': 1,
        'returned_im_square_crop': square_crop,
        'returned_im_dtype': np.float32,  # only np.float32 supported
        'returned_im_use_color': False,  # only False supported for now
        'video_dir': ROOT_DIR + '/projects/video-downloads',
        #'video_filename': 'sea-turtles-yLuEx-XH3Uc.mp4'
        #'video_filename': 'seattle-driving-fkps18H3SXY.mp4'
        'video_filename': 'sea-turtles-11hr-spxtEt6RaS4.mp4'
    }

    return params_video_playback


def get_pre_proc_params():
    params_pre_proc = {
        'brightness_threshold': 1.0 / 255,
        'im_dim': RF_IM_DIM
    }
    return params_pre_proc


def get_sim_folder_manager_params():

    print()
    sim_prefix = input('sim prefix? >> ')
    print()

    if sim_prefix[-1] != '_':
        sim_prefix += '_'

    if len(sim_prefix) == 0:
        sim_prefix = 'default_'

    params = {
        'sim_prefix': sim_prefix,
        'sim_folders_path': ROOT_DIR + '/projects/NL-sim2/',
        'scripts_folder_path': ROOT_DIR + '/projects/NL/'
    }
    return params


def get_brain_params():
    brain_params = {
        'input_im_dim': RF_IM_DIM,
        'num_rfs': 400,
        'lr': 1.0 / 1000,
        'rate_lr': 1.0 / 1000,
        'max_time': 5000000,
        'apply_rate_control': True,
        'do_raster_plots_every_k_im': 4  # or None
    }

    return brain_params


def get_visualizer_params():
    params = {
        'color_enabled': False,
        'fps_display_interval': 6,
        'image_display_secs_fast': 3,  # 8+ for good speed
        'waitKey_time_fast': 1,  # 1, 100, 5000
        'image_display_secs_slow': 0.0, # 0.01  # 2,  # 0: every frame
        'waitKey_time_slow': 1,  # 1, 100, 5000
        'scale_camera_factor': 1,
        'auto_switch_to_slow_disp_time': None,
        'init_fast': not DISABLE_BRAIN  # start with "fast" display if brain is enabled
    }
    return params


def init_demo():
    return {
        #'robot_brain': MultiLayerSeqNNBrain(get_brain_params()),
        #'robot_brain': SeqNNSeqKMeansBrain(get_brain_params()),
        #'robot_brain': BatchIterWTABrain(get_brain_params()),
        'robot_brain': RateControlWTABRain(get_brain_params()),
        #'robot_brain': RateControlSomWtaBrain(get_brain_params()),
        'robot_sensors': VideoPlaybackSensor(get_sensors_params()),
        'visualizer': SegmentVisualizer(get_visualizer_params()),
        'sim_folder_manager': SimFolderManager(get_sim_folder_manager_params())
    }


def run_demo(demo_components):

    robot_brain = demo_components['robot_brain']
    robot_sensors = demo_components['robot_sensors']
    visualizer = demo_components['visualizer']
    sim_folder_manager = demo_components['sim_folder_manager']

    pre_proc = EventPreProcessor(get_pre_proc_params())

    robot_brain.set_plots_folder(sim_folder_manager.get_plots_save_folder())

    random.seed(1233)

    while True:

        # reference
        # a = np.dot(np.random.random((200, 200)), np.random.random((200, 200)))

        im = robot_sensors.read_input()

        events_p, events_n = pre_proc.step(input_frame=im)

        if not DISABLE_BRAIN:
            robot_brain.process_input(input_events_p=events_p, input_events_n=events_n)

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

















