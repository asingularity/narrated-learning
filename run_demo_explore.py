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


MAX_HISTORY_LENGTH = 1000000 + 1
USERNAME = 'intec'
NUM_INPUT_RAYS = 16
INPUT_DIM = NUM_INPUT_RAYS * 3

SIM_LOAD_NAME = None


def get_model_params():
    params = {
        'max_angular_velocity': 0.25,
        'linear_velocity': 0.4
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
        'training_delay': 128,
        'input_dim': INPUT_DIM,

        # ************ I-O-C predictor ensemble ************
        # TO DO: add online training back in
        'predictor_ensemble_load_from_file': False,
        'predictor_ensemble_filename': None,
    }
    return params


def get_environment_params():
    params = {
        'width': 30,
        'height': 30,
        'add_random_color_boundary_walls': False,
        'min_num_walls': 8,
        'max_num_walls': 8,
        'wall_min_length': 8,
        'wall_max_length': 10,
        'min_space_between_walls': 2,
        'init_robot_x': 5,
        'init_robot_y': 5,
        'init_robot_theta': 45
    }
    return params


def get_visualizer_params():
    params = {
        'fps_display_interval': 3,
        'plot_brain_error_frames': None,
        'image_display_frames_fast': 1000,  # 1  # 1000
        'waitKey_time_fast': 1,  # 1, 100, 5000
        'image_display_frames_slow': 1,  # 1  # 1000
        'waitKey_time_slow': 10,  # 1, 100, 5000
        'scale_topdown_factor': 20,
        'scale_camera_factor': 20,
        'no_wall_ray_color': (0.1, 0.1, 0.1),
        'auto_switch_to_slow_disp_time': None
    }
    return params


def get_task_manager_params():
    params = {
        'run_steps_if_task_mode_disabled': MAX_HISTORY_LENGTH,
        'enabled': False,
        'num_trials_per_set': 5000,
        'sleep_every_trial': 0.5,  # to be able to see the next goal
        'constrain_to_params': True,
        'max_trial_steps': 2,
        'min_delta_theta': -pi/6.0,
        'max_delta_theta': pi/6.0,
        'min_distance': 5,
        'max_distance': 5
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

    while not task_manager.finished_sim():

        new_goal_chosen_this_step = task_manager.do_step(topdown_info=robot_environment.get_topdown_info(),
                                                         robot_environment=robot_environment,
                                                         robot_sensors=robot_sensors,
                                                         robot_brain=robot_brain)

        if new_goal_chosen_this_step:
            robot_brain.process_new_goal_states(goal_states=task_manager.get_task_goal_states(),
                                                # these parameters are needed for displaying planned paths:
                                                visualizer=visualizer,
                                                rays=robot_sensors.get_rays(),
                                                topdown_info=robot_environment.get_topdown_info(),
                                                current_goal_position_angle=task_manager.get_current_goal_position_angle())

        robot_sensors.read_input(nonzero_tiles=robot_environment.get_nonzero_tiles(),
                                 robot_theta=robot_environment.get_robot_theta())

        #   robot_sensors.rays updated from environment: STATE_T+1
        #   robot_model.last_motor_command: CMD_T

        robot_brain.process_input(rays=robot_sensors.get_rays(),
                                  last_motor_command=robot_model.get_last_motor_command(),
                                  models_save_folder=sim_folder_manager.get_models_save_folder(),
                                  debug_topdown_info=robot_environment.get_topdown_info())  # for storing robot position, angle for debugging planning

        robot_model.act_upon_processing(motor_command=robot_brain.get_motor_output())

        if use_keyboard_input:
            linear_speed, angular_speed = visualizer.get_linear_angular_speed()
        else:
            linear_speed, angular_speed = robot_model.get_delta_configuration()

        robot_environment.step_environment(linear_speed=linear_speed,
                                           angular_speed=angular_speed)

        #   robot_model.last_motor_command updated to new random command CMD_T
        #   robot_environment state updated with motor cmd CMD_T: STATE_T -> STATE_T+1

        visualizer.visualize(rays=robot_sensors.get_rays(),
                             robot_brain=robot_brain,  # get_autoenc_images, get_predictor_images, get_error_names_histories
                             topdown_info=robot_environment.get_topdown_info(),
                             plots_save_folder=sim_folder_manager.get_plots_save_folder(),
                             current_goal_position_angle=task_manager.get_current_goal_position_angle(),
                             plan_position_angle_list=robot_brain.get_plan_position_angle_list(rays=robot_sensors.get_rays(),
                                                                                               goal_states=task_manager.get_task_goal_states()))

    print ('Finished Simulation.')

    robot_brain.save_states_history(plots_save_folder=sim_folder_manager.get_plots_save_folder(), state_indices_list=[0])
    robot_brain.save_debug_topdown_info_history(plots_save_folder=sim_folder_manager.get_plots_save_folder())
    task_manager.evaluate(plots_save_folder=sim_folder_manager.get_plots_save_folder())
    print ('Finished Evaluation.')


def demo():
    demo_components = init_demo()
    run_demo(demo_components)


if __name__ == '__main__':
    demo()
