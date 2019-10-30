import cv2
import random
import numpy as np
random.seed(6)
np.random.seed(6)

from robot_brain import RobotBrain
from robot_model import RobotModel
from robot_sensors import RobotSensors
from robot_environment import RobotEnvironment
from visualizer import Visualizer
from sim_folder_manager import SimFolderManager
from task_manager import TaskManager
import time
from math import pi


MAX_HISTORY_LENGTH = 10600000 + 1
USERNAME = 'intec'
NUM_INPUT_RAYS = 32
INPUT_DIM = NUM_INPUT_RAYS * 3

SIM_LOAD_NAME = '2019-07-21T21:18:20.175737'
ENABLE_TASK_MODE = False

TABLE_ENTRIES = 8000  # * 10  # 2000 with 16; 4000 with 8
TABLE_LEARN_TIME = TABLE_ENTRIES * 16
PREDICTION_LEARN_TIME = TABLE_ENTRIES * 16

SHOW_PLAN_ONLY = False


def get_model_params():
    params = {
        'max_angular_velocity': 0.35 * 1.5,
        'max_linear_velocity': 0.4
    }
    return params


def get_sensors_params():
    params = {
        'num_rays': NUM_INPUT_RAYS,
        'fov_degrees': 100
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
        'error_average_steps': 1000,
        'input_dim': INPUT_DIM,
        'enable_learning': not ENABLE_TASK_MODE,

        # ************ load from file ************
        'predictor_ensemble_load_from_file': ENABLE_TASK_MODE,
        'predictor_ensemble_filename': '/srv/projects/NL-sim/' + SIM_LOAD_NAME + '/ensemble.pkl',
        'predictor_ensemble_save_every_k_secs': None,  # None: never save

        # ************ I-O-C predictor ensemble ************
        'table_entries': TABLE_ENTRIES,
        'table_learn_time': TABLE_LEARN_TIME,
        'prediction_learn_time': PREDICTION_LEARN_TIME,
        'predict_time_per_layer': [1],  # referenced to layer before it
        'max_num_goal_states': 9
    }
    return params


def get_environment_params():
    params = {
        'width': 10,
        'height': 10,
        'add_random_color_boundary_walls': True,
        'min_num_walls': 0,
        'max_num_walls': 0,
        'wall_min_length': 2,
        'wall_max_length': 3,
        'min_space_between_walls': 3,
        'init_robot_x': 5,
        'init_robot_y': 5,
        'init_robot_theta': 45
    }
    return params


def get_visualizer_params():
    params = {
        'fps_display_interval': 3,
        'plot_brain_error_frames': None,
        'image_display_secs_fast': 5,
        'waitKey_time_fast': 1,  # 1, 100, 5000
        'image_display_secs_slow': 0,  # 0: every frame
        'waitKey_time_slow': 1,  # 1, 100, 5000  # TODO use SHOW_PLAN_ONLY (1000)
        'scale_topdown_factor': 20,
        'scale_camera_factor': 20,
        'no_wall_ray_color': (0.3, 0.3, 0.3),
        'auto_switch_to_slow_disp_time': TABLE_LEARN_TIME + PREDICTION_LEARN_TIME,
        'show_table_ims': True,
        'init_fast': True  # start with "fast" display
    }
    return params


def get_task_manager_params():
    params = {
        'run_steps': MAX_HISTORY_LENGTH,
        'enabled': ENABLE_TASK_MODE,
        'enabled_after_t': TABLE_LEARN_TIME + PREDICTION_LEARN_TIME,  # None or a time step, additional way to enable but with delay. overridden by 'enabled' flag.
        'goal_regions': [  # c, r, w, h
            [0, 0, 3, 3],
            [0, 7, 3, 3],
            [7, 0, 3, 3],
            [7, 7, 3, 3]
        ],
        'steps_per_task_goal': 200 / 4,  # if enabled is True  # TODO use SHOW_PLAN_ONLY (1)
        'task_randomize_robot_position': SHOW_PLAN_ONLY,  # for debugging, random robot position each step in task mode
        'debug_print': False  # prints correct / incorrect goal reached
    }
    return params


def init_demo():
    return {
        'robot_environment': RobotEnvironment(get_environment_params()),
        'robot_brain': RobotBrain(get_brain_params()),
        'robot_model': RobotModel(get_model_params()),
        'robot_sensors': RobotSensors(get_sensors_params()),
        'visualizer': Visualizer(get_visualizer_params()),
        'sim_folder_manager': SimFolderManager(get_sim_folder_manager_params()),
        'task_manager': TaskManager(get_task_manager_params())
    }


def run_demo(demo_components):
    robot_environment = demo_components['robot_environment']
    robot_brain = demo_components['robot_brain']
    robot_model = demo_components['robot_model']
    robot_sensors = demo_components['robot_sensors']
    visualizer = demo_components['visualizer']
    sim_folder_manager = demo_components['sim_folder_manager']
    task_manager = demo_components['task_manager']

    use_keyboard_input = False
    random.seed(1233)  # change to make movement different, without different walls

    # TODO fix see through walls from left side of vertical wall viewing right
    # TODO is this fixed?

    t_process = 0
    t_total = 0
    last_disp_t = time.time()

    while not task_manager.finished_sim():
        t_total_0 = time.time()

        current_task_goal_index, goal_index_reached_this_step, randomize_robot_position = task_manager.do_step(topdown_info=robot_environment.get_topdown_info(),
                                                                                                               robot_environment=robot_environment,
                                                                                                               robot_sensors=robot_sensors,
                                                                                                               robot_brain=robot_brain)

        robot_sensors.read_input(nonzero_tiles=robot_environment.get_nonzero_tiles(),
                                 robot_theta=robot_environment.get_robot_theta())

        #   robot_sensors.rays updated from environment: STATE_T+1
        #   robot_model.last_motor_command: CMD_T

        t_process_0 = time.time()
        new_motor_out = robot_brain.process_input_get_motor(rays=robot_sensors.get_rays(),
                                                            raycast_image=robot_sensors.get_raycast_image(),
                                                            last_motor_command=robot_model.get_last_motor_command(),
                                                            models_save_folder=sim_folder_manager.get_models_save_folder(),
                                                            debug_topdown_info=robot_environment.get_topdown_info(),  # for storing robot position, angle for debugging planning
                                                            goal_index_reached=goal_index_reached_this_step,
                                                            goal_index_task=current_task_goal_index)
        t_process += (time.time() - t_process_0)

        robot_model.act_upon_processing(motor_command=new_motor_out)

        if use_keyboard_input:
            linear_speed, angular_speed = visualizer.get_linear_angular_speed()
        else:
            linear_speed, angular_speed = robot_model.get_delta_configuration()

        #   robot_model.last_motor_command updated to new random command CMD_T
        #   robot_environment state updated with motor cmd CMD_T: STATE_T -> STATE_T+1
        visualizer.visualize(rays=robot_sensors.get_rays(),
                             raycast_image=robot_sensors.get_raycast_image(),
                             topdown_info=robot_environment.get_topdown_info(),
                             plots_save_folder=sim_folder_manager.get_plots_save_folder(),
                             goal_regions=task_manager.get_goal_regions(),
                             current_task_goal_index=current_task_goal_index,
                             robot_brain=robot_brain)  # So it can call .get_table_ims() only sometimes

        robot_environment.step_environment(linear_speed=linear_speed,
                                           angular_speed=angular_speed,
                                           randomize_robot_position=randomize_robot_position)

        t_total += (time.time() - t_total_0)

        if time.time() - last_disp_t > 30:
            last_disp_t = time.time()
            print('t_process / t_total: ', t_process / t_total)
            t_process = 0
            t_total = 0

    robot_brain.save_model(models_save_folder=sim_folder_manager.get_models_save_folder())
    print ('Finished Simulation.')

    robot_brain.save_states_history(plots_save_folder=sim_folder_manager.get_plots_save_folder(), state_indices_list=[0])
    robot_brain.save_debug_topdown_info_history(plots_save_folder=sim_folder_manager.get_plots_save_folder())
    task_manager.evaluate(plots_save_folder=sim_folder_manager.get_plots_save_folder())
    print ('Finished Evaluation.')

    while True:
        cv2.waitKey(1)


def demo():
    demo_components = init_demo()
    run_demo(demo_components)


if __name__ == '__main__':
    demo()
