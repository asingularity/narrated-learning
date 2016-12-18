

from robot_brain import RobotBrain
from robot_model import RobotModel
from robot_sensors import RobotSensors
from robot_environment import RobotEnvironment
from performance_evaluator import PerformanceEvaluator
from visualizer import Visualizer


def get_model_params():
    params = {
        'max_angular_velocity': 0.2,
        'linear_velocity': 0.05
    }
    return params


def get_sensors_params():
    params = {
        'num_rays': 40,
        'fov_degrees': 100
    }
    return params


def get_brain_params():
    params = {
        'autoenc_heirarchy_compression': [0.5, 0.5, 0.5],
        'autoenc_learning_rate': 0.01,
        'autoenc_learning_disable_step': 2000000,
        'predict_time_steps': [1, 2, 4, 8],
        'predict_nets_input_compression_levels':   [0, 1, 2],
        'predict_nets_context_compression_levels': [1, 2, 3],
        'predict_nets_output_compression_levels':  [0, 1, 2],
        'predict_nets_learning_rate': 0.01,
        'predict_nets_hidden_dim': 100,
        'predict_nets_learning_disable_step': 2000000,
        'error_average_steps': 4000,
        'save_steps': 100000,
        'sensors_params': get_sensors_params(),
        'save_folder': '/home/petre/projects/NL/plots/'
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
    }
    return params


def get_visualizer_params():
    params = {
        'fps_display_interval': 3,
        'image_display_frames': 1000,
        'plot_brain_error_frames': 25000,
        'waitKey_time': 1,
        'scale_topdown_factor': 10,
        'scale_camera_factor': 20,
        'no_wall_ray_color': 0.1,
        'plots_folder': '/home/petre/projects/NL/plots/'
    }
    return params


def init_demo():
    return {
        'robot_brain': RobotBrain(get_brain_params()),
        'robot_model': RobotModel(get_model_params()),
        'robot_sensors': RobotSensors(get_sensors_params()),
        'robot_environment': RobotEnvironment(get_environment_params()),
        'perf_eval': PerformanceEvaluator(get_evaluator_params()),
        'visualizer': Visualizer(get_visualizer_params())
    }


#@profile
def run_demo(demo_components):
    robot_brain = demo_components['robot_brain']
    robot_model = demo_components['robot_model']
    robot_sensors = demo_components['robot_sensors']
    robot_environment = demo_components['robot_environment']
    perf_eval = demo_components['perf_eval']
    visualizer = demo_components['visualizer']

    while not perf_eval.finished():
        robot_sensors.read_input(robot_environment)
        robot_brain.process_input(robot_sensors)
        robot_model.act_upon_processing(robot_brain)
        robot_environment.step_environment(robot_model, visualizer)
        perf_eval.evaluate(robot_sensors,
                           robot_brain,
                           robot_model,
                           robot_environment)
        visualizer.visualize(robot_sensors,
                             robot_brain,
                             robot_model,
                             robot_environment)


def demo():
    demo_components = init_demo()
    run_demo(demo_components)


if __name__ == '__main__':
    demo()
