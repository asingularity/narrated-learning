
import random
import numpy as np
random.seed(0)
np.random.seed(0)

from robot_brain import RobotBrain
from robot_model import RobotModel
from robot_sensors import RobotSensors
from robot_environment import RobotEnvironment
from performance_evaluator import PerformanceEvaluator
from visualizer import Visualizer
from sim_folder_manager import SimFolderManager
from task_manager import TaskManager
import time
from math import pi


MAX_HISTORY_LENGTH = 4000000 + 1
USERNAME = 'intec'
NUM_INPUT_RAYS = 40
INPUT_DIM = NUM_INPUT_RAYS * 3


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
        'sim_folders_path': '/home/' + USERNAME + '/projects/NL/sim/',
        'scripts_folder_path': '/home/' + USERNAME + '/projects/NL/'
    }
    return params


def get_brain_params():
    params = {
        # ************ general ************
        'max_history_length': MAX_HISTORY_LENGTH,
        'error_average_steps': 1000,  # 1000
        'predictors_enable': True,
        'training_delay': 128,
        # ************ autoencoders ************
        'autoencoders': [
            {'num_inputs': INPUT_DIM, 'num_hidden': INPUT_DIM / 2, 'learning_rate': 0.01},
            {'num_inputs': INPUT_DIM / 2, 'num_hidden': INPUT_DIM / 4, 'learning_rate': 0.01},
            {'num_inputs': INPUT_DIM / 4, 'num_hidden': INPUT_DIM / 8, 'learning_rate': 0.01}
        ],
        'autoencoders_training_time_range': [0, MAX_HISTORY_LENGTH],
        'autoencoders_save_every_k_steps': None,
        'autoencoders_enable_training': False,
        'autoencoders_load_from_file': True,
        'autoencoders_load_filename': '/home/' + USERNAME + '/projects/NL/sim/2017-02-19T16:01:29.835767_autoencoders/autoencoders.pkl',
        # ************ predictors ************
        'predictors': [
            {'state_index_input': 0, 'state_index_context': 0, 'state_index_output': 0,
                                     'dt_context': 2,          'dt_output': 1}
        ],
        'predictors_training_time_range': [0, MAX_HISTORY_LENGTH],
        'predictors_test_every_k_steps': None, #500,  # 500 for training
        'predictors_save_every_k_steps': 1000000,
        'predictors_enable_training': False,
        'predictors_optimize_training': True,  # second step
        'predictors_load_from_file': True,
        'predictors_load_filename': '/home/' + USERNAME + '/projects/NL/sim/2017-03-05T07:14:47.040189_predictor_inverse_level_0/predictors.pkl',
        # ************ inverse model ************
        'inverse_models': [
            {'state_index_current': 0, 'state_index_future': 0, 'dt': 1}
        ],
        'inverse_training_time_range': [0, MAX_HISTORY_LENGTH],
        'inverse_test_every_k_steps': None, #500, #None,
        'inverse_save_every_k_steps': None, #4000000,
        'inverse_enable_training': False,
        'inverse_load_from_file': False,
        'inverse_load_filename': '/home/' + USERNAME + '/projects/NL/sim/2017-03-05T07:14:47.040189_predictor_inverse_level_0/inverse.pkl',
    }
    return params


def get_environment_params():
    params = {
        'width': 40,
        'height': 40,
        'min_num_walls': 20,
        'max_num_walls': 20,
        'wall_min_length': 1,
        'wall_max_length': 20,
        'min_space_between_walls': 2,
        'init_robot_x': 25,
        'init_robot_y': 25,
        'init_robot_theta': 45
    }
    return params


def get_evaluator_params():
    params = {
        'run_time': MAX_HISTORY_LENGTH
    }
    return params


def get_visualizer_params():
    params = {
        'fps_display_interval': 3,
        'image_display_frames': 100,  # 1000
        'plot_brain_error_frames': 500,
        'waitKey_time': 1,  # 1
        'scale_topdown_factor': 10,
        'scale_camera_factor': 20,
        'no_wall_ray_color': (0.1, 0.1, 0.1),
    }
    return params


def get_task_manager_params():
    params = {
        'enabled': False,
        'constrain_to_params': True,
        'max_task_steps': 10,  # 64,  # 64 * 2
        'min_delta_theta': -pi/4.0,
        'max_delta_theta': pi/4.0,
        'min_distance': 3,
        'max_distance': 3
    }
    return params


def init_demo():
    return {
        'robot_environment': RobotEnvironment(get_environment_params()),
        'robot_brain': RobotBrain(get_brain_params()),
        'robot_model': RobotModel(get_model_params()),
        'robot_sensors': RobotSensors(get_sensors_params()),
        'perf_eval': PerformanceEvaluator(get_evaluator_params()),
        'visualizer': Visualizer(get_visualizer_params()),
        'sim_folder_manager': SimFolderManager(get_sim_folder_manager_params()),
        'task_manager': TaskManager(get_task_manager_params())
    }


def run_demo(demo_components):
    robot_environment = demo_components['robot_environment']
    robot_brain = demo_components['robot_brain']
    robot_model = demo_components['robot_model']
    robot_sensors = demo_components['robot_sensors']
    perf_eval = demo_components['perf_eval']
    visualizer = demo_components['visualizer']
    sim_folder_manager = demo_components['sim_folder_manager']
    task_manager = demo_components['task_manager']

    use_keyboard_input = False
    task_manager_enabled = task_manager.get_enabled()
    new_task_chosen = False
    task_goal_states = None
    random.seed(1233)  # change to make movement different, without different walls
    # TODO fix see through walls from left side of vertical wall viewing right

    while not perf_eval.finished():

        if task_manager_enabled:
            time.sleep(0.3)
            task_manager.choose_new_task_goal(topdown_info=robot_environment.get_topdown_info())
            task_goal_nonzero_tiles = robot_environment.get_nonzero_tiles(robot_position_angle=task_manager.get_current_goal_position_angle())
            task_goal_rays = robot_sensors.get_rays(nonzero_tiles=task_goal_nonzero_tiles,
                                                    robot_position_angle=task_manager.get_current_goal_position_angle())
            task_goal_sensory_input = task_goal_rays['ray_colors']
            task_goal_states = robot_brain.get_autoencoder_states_for_input(net_input=task_goal_sensory_input)
            robot_brain.reset_for_new_task()
            new_task_chosen = True

        finished_task = False
        while not finished_task:

            robot_sensors.read_input(nonzero_tiles=robot_environment.get_nonzero_tiles(),
                                     robot_theta=robot_environment.get_robot_theta())

            #   robot_sensors.rays updated from environment: STATE_T+1
            #   robot_model.last_motor_command: CMD_T

            robot_brain.process_input(rays=robot_sensors.get_rays(),
                                      last_motor_command=robot_model.get_last_motor_command(),
                                      goal_states=task_goal_states,
                                      models_save_folder=sim_folder_manager.get_models_save_folder())

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
                                 current_goal_position_angle=task_manager.get_current_goal_position_angle())

            if task_manager_enabled:
                if new_task_chosen:
                    time.sleep(0.3)
                    new_task_chosen = False
                finished_task = task_manager.finished_task()
            else:
                finished_task = True

            perf_eval.step()

        perf_eval.evaluate()

    print 'Finished Simulation.'


def demo():
    demo_components = init_demo()
    run_demo(demo_components)


if __name__ == '__main__':
    demo()
