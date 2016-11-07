
from robot_brain import RobotBrain
from robot_model import RobotModel
from robot_sensors import RobotSensors
from robot_environment import RobotEnvironment
from performance_evaluator import PerformanceEvaluator
from visualizer import Visualizer


def get_brain_params():
    params = {
    }
    return params


def get_model_params():
    params = {
        'max_angular_velocity': 0.001
    }
    return params


def get_sensors_params():
    params = {
        'num_rays': 40,
        'fov_degrees': 100
    }
    return params


def get_environment_params():
    params = {
        'width': 40,
        'height': 40,
        'init_robot_x': 25,
        'init_robot_y': 25,
        'init_robot_theta': 45,
        'walls': [
            {'x_start': 2,
             'y_start': 11,
             'length': 21,
             'color': 0.2,
             'orientation': 'vertical'},
            {'x_start': 5,
             'y_start': 22,
             'length': 11,
             'color': 0.6,
             'orientation': 'vertical'},
            {'x_start': 10,
             'y_start': 15,
             'length': 15,
             'color': 0.4,
             'orientation': 'horizontal'},
            {'x_start': 32,
             'y_start': 32,
             'length': 7,
             'color': 0.7,
             'orientation': 'horizontal'},
            {'x_start': 22,
             'y_start': 15,
             'length': 7,
             'color': 0.7,
             'orientation': 'vertical'},
            {'x_start': 38,
             'y_start': 20,
             'length': 7,
             'color': 0.7,
             'orientation': 'vertical'},
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
        'scale_topdown_factor': 10,
        'scale_camera_factor': 20
    }
    return params


def demo():
    robot_brain = RobotBrain(get_brain_params())
    robot_model = RobotModel(get_model_params())
    robot_sensors = RobotSensors(get_sensors_params())
    robot_environment = RobotEnvironment(get_environment_params())
    perf_eval = PerformanceEvaluator(get_evaluator_params())
    visualizer = Visualizer(get_visualizer_params())

    while not perf_eval.finished():
        robot_sensors.read_input(robot_environment)
        robot_brain.process_input(robot_sensors)
        robot_model.act_upon_processing(robot_brain)
        robot_environment.step_environment(robot_model)
        perf_eval.evaluate(robot_sensors,
                           robot_brain,
                           robot_model,
                           robot_environment)
        visualizer.visualize(robot_sensors,
                             robot_brain,
                             robot_model,
                             robot_environment)

if __name__ == '__main__':
    demo()
