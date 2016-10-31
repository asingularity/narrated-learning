
from robot_brain import RobotBrain
from robot_model import RobotModel
from robot_sensors import RobotSensors
from robot_environment import RobotEnvironment
from performance_evaluator import PerformanceEvaluator


def demo():
    r_brain = RobotBrain()
    r_model = RobotModel()
    r_sensors = RobotSensors()
    r_env = RobotEnvironment()
    perf_eval = PerformanceEvaluator()

    while not perf_eval.finished():
        r_sensors.read_input(r_env)
        r_brain.process_input(r_sensors)
        r_model.act_upon_processing(r_brain)
        r_env.step_environment(r_model)
        perf_eval.evaluate(r_sensors,
                           r_brain,
                           r_model,
                           r_env)

if __name__ == '__main__':
    demo()
