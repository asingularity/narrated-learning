

from robot_brain import RobotBrain
from robot_model import RobotModel
from robot_sensors import RobotSensors
from robot_environment import RobotEnvironment
from performance_evaluator import PerformanceEvaluator
from visualizer import Visualizer
from sim_folder_manager import SimFolderManager


MAX_HISTORY_LENGTH = 4000000 + 1
USERNAME = 'intec'


def get_model_params():
    params = {
        'max_angular_velocity': 0.25,
        'linear_velocity': 0.4
    }
    return params


def get_sensors_params():
    params = {
        'num_rays': 40,
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
        'error_average_steps': 10000,
        'predictors_enable': True,
        'training_delay': 128,
        # ************ autoencoders ************
        'autoencoders': [
            {'num_inputs': 40, 'num_hidden': 20, 'learning_rate': 0.01},
            {'num_inputs': 20, 'num_hidden': 10, 'learning_rate': 0.01},
            {'num_inputs': 10, 'num_hidden': 5, 'learning_rate': 0.01}
        ],
        'autoencoders_training_time_range': [0, MAX_HISTORY_LENGTH],
        'autoencoders_save_every_k_steps': None,
        'autoencoders_enable_training': False,
        'autoencoders_load_from_file': True,
        'autoencoders_load_filename': '/home/' + USERNAME + '/projects/NL/sim/autoencoders_2017-01-22T20:37:11.979346/autoencoders.pkl',
        # ************ predictors ************
        'predictors': [
            {'state_index_input': 3, 'state_index_context': 3, 'state_index_output': 3,
             'dt_output': 8, 'dt_context': 16},
            {'state_index_input': 3, 'state_index_context': 3, 'state_index_output': 3,
             'dt_output': 16, 'dt_context': 32},
            {'state_index_input': 3, 'state_index_context': 3, 'state_index_output': 3,
             'dt_output': 32, 'dt_context': 64}
        ],
        'predictors_training_time_range': [0, MAX_HISTORY_LENGTH],
        'predictors_test_every_k_steps': 50,  # 50 for training
        'predictors_save_every_k_steps': None,
        'predictors_enable_training': False,
        'predictors_load_from_file': True,
        'predictors_load_filename': '/home/' + USERNAME + '/projects/NL/sim/predictors_2017-01-23T08:39:18.437290/predictors.pkl',
        # ************ inverse model ************
        'inverse_models': [
            {'state_index_current': 0, 'state_index_future': 0, 'dt': 1},
            {'state_index_current': 1, 'state_index_future': 1, 'dt': 2},
            {'state_index_current': 2, 'state_index_future': 2, 'dt': 2},
            {'state_index_current': 3, 'state_index_future': 3, 'dt': 4},
            {'state_index_current': 3, 'state_index_future': 3, 'dt': 8},
        ],
        'inverse_training_time_range': [0, MAX_HISTORY_LENGTH],
        'inverse_test_every_k_steps': 50,
        'inverse_save_every_k_steps': 2000000,
        'inverse_enable_training': True,
        'inverse_load_from_file': False,
        'inverse_load_filename': '/home/' + USERNAME + '/projects/NL/sim/<none>/<none>.pkl',
    }
    return params


def get_environment_params():
    params = {
        'use_keyboard_input': False,
        'width': 40,
        'height': 40,
        'init_robot_x': 25,
        'init_robot_y': 25,
        'init_robot_theta': 45,
        'walls': [
            {'x_start': 2,
             'y_start': 11,
             'length': 21,
             'color': 0.1,
             'orientation': 'vertical'},
            {'x_start': 5,
             'y_start': 22,
             'length': 11,
             'color': 0.2,
             'orientation': 'vertical'},
            {'x_start': 10,
             'y_start': 15,
             'length': 15,
             'color': 0.3,
             'orientation': 'horizontal'},
            {'x_start': 32,
             'y_start': 32,
             'length': 7,
             'color': 0.4,
             'orientation': 'horizontal'},
            {'x_start': 22,
             'y_start': 15,
             'length': 7,
             'color': 0.5,
             'orientation': 'vertical'},
            {'x_start': 38,
             'y_start': 20,
             'length': 7,
             'color': 0.6,
             'orientation': 'vertical'},
            {'x_start': 28,
             'y_start': 14,
             'length': 12,
             'color': 0.7,
             'orientation': 'vertical'},
            {'x_start': 24,
             'y_start': 6,
             'length': 12,
             'color': 0.8,
             'orientation': 'horizontal'},
        ]
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
        'image_display_frames': 1000,
        'plot_brain_error_frames': 50000,
        'waitKey_time': 1,
        'scale_topdown_factor': 10,
        'scale_camera_factor': 20,
        'no_wall_ray_color': 0.1,
    }
    return params


def init_demo():
    return {
        'robot_brain': RobotBrain(get_brain_params()),
        'robot_model': RobotModel(get_model_params()),
        'robot_sensors': RobotSensors(get_sensors_params()),
        'robot_environment': RobotEnvironment(get_environment_params()),
        'perf_eval': PerformanceEvaluator(get_evaluator_params()),
        'visualizer': Visualizer(get_visualizer_params()),
        'sim_folder_manager': SimFolderManager(get_sim_folder_manager_params())
    }


#@profile
def run_demo(demo_components):
    robot_brain = demo_components['robot_brain']
    robot_model = demo_components['robot_model']
    robot_sensors = demo_components['robot_sensors']
    robot_environment = demo_components['robot_environment']
    perf_eval = demo_components['perf_eval']
    visualizer = demo_components['visualizer']
    sim_folder_manager = demo_components['sim_folder_manager']

    while not perf_eval.finished():
        robot_sensors.read_input(robot_environment, robot_model)
        robot_brain.process_input(robot_sensors, sim_folder_manager)
        robot_model.act_upon_processing(robot_brain)
        robot_environment.step_environment(robot_model, visualizer)
        perf_eval.evaluate(robot_sensors,
                           robot_brain,
                           robot_model,
                           robot_environment)
        visualizer.visualize(robot_sensors,
                             robot_brain,
                             robot_model,
                             robot_environment,
                             sim_folder_manager)

    print 'Finished Simulation.'


def learning_demo():
    demo_components = init_demo()
    run_demo(demo_components)


def task_demo():
    demo_components = init_demo()
    run_demo(demo_components)


if __name__ == '__main__':
    do_learning_demo = True
    do_task_demo = False

    if do_learning_demo:
        learning_demo()

    if do_task_demo:
        task_demo()
