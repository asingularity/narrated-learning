
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
    }
    return params


def get_sensors_params():
    params = {
    }
    return params


def get_environment_params():
    params = {
    }
    return params


def get_evaluator_params():
    params = {
    }
    return params


def get_visualizer_params():
    params = {
        'fps_display_interval': 3
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
