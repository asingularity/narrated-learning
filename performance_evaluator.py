
class PerformanceEvaluator(object):
    def __init__(self, params):
        self.t = 0
        self.run_time = params['run_time']

    def finished(self):
        return self.t >= self.run_time

    def step(self):
        self.t += 1

    def evaluate(self):
        pass

