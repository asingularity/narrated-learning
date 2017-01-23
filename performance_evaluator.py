
class PerformanceEvaluator(object):
    def __init__(self, params):
        self.t = 0
        self.run_time = params['run_time']

    def finished(self):
        return self.t < self.run_time + 1

    def evaluate(self, robot_sensors, robot_brain, robot_model, robot_environment):
        self.t += 1

