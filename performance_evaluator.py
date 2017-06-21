
from math import sqrt, pi


class PerformanceEvaluator(object):
    def __init__(self, params):
        self.max_trials = params['max_trials']
        self.trial = None
        self.position_errors_per_trial = None
        self.theta_errors_per_trial = None

    def begin_trial_set(self, trial_param_name, trial_param_value):
        self.trial = 0
        self.position_errors_per_trial = []
        self.theta_errors_per_trial = []

    def finalize_trial_set(self):
        pass

    def finished_trial_set(self):
        return self.trial >= self.max_trials

    def add_trial_end_data(self, robot_state, goal_state):
        goal_p_x = goal_state[0]
        goal_p_y = goal_state[1]
        goal_theta = goal_state[2]

        robot_p_x = robot_state[0]
        robot_p_y = robot_state[1]
        robot_theta = robot_state[2]

        self.position_errors_per_trial.append(sqrt(pow(goal_p_x - robot_p_x, 2) + pow(goal_p_y - robot_p_y, 2)))
        self.theta_errors_per_trial.append(min(min(abs(robot_theta - goal_theta), abs(robot_theta + 2 * pi - goal_theta)), abs(robot_theta - 2 * pi - goal_theta)))

        self.trial += 1

    def evaluate_all_sets(self, plots_save_folder):
        pass
