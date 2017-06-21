
from math import pi, sin, cos
import numpy as np
import random
import time

class TaskManager(object):
    def __init__(self, params):
        '''

        trial_set_params_dict: {'learning_rate': [0.01, 0.02, 0.04, 0.01, 0.02, 0.04],
                                'task_length': [10, 10, 10, 20, 20, 20]}

        Task manager is responsible for creating sets of tasks.
        It needs a way to inform other modules of parameter changes?
        :param params:
        '''

        self.run_steps_if_task_mode_disabled = params['run_steps_if_task_mode_disabled']
        self.task_mode_enabled = params['enabled']

        self.sleep_every_trial = params['sleep_every_trial']
        self.num_trials_per_set = params['num_trials_per_set']
        self.t_trial_max = params['max_trial_steps']
        self.min_delta_theta = params['min_delta_theta']  # radians
        self.max_delta_theta = params['max_delta_theta']  # radians
        self.min_distance = params['min_distance']
        self.max_distance = params['max_distance']
        self.constrain_to_params = params['constrain_to_params']

        self.goal_states = None
        self.goal_r_x = None
        self.goal_r_y = None
        self.goal_r_theta = None

        self.sets_param_name = params['sets_param_name']
        self.sets_param_values = params['sets_param_values']

        self.num_sets = len(self.sets_param_values)
        self.position_errors_by_set_and_trial = np.zeros((self.num_sets, self.num_trials_per_set))
        self.theta_errors_by_set_and_trial = np.zeros((self.num_sets, self.num_trials_per_set))

        self.step = 0

        self.trial = 0
        self.global_trial = 0
        self.set = 0
        self.t_trial = 0

        self.all_sets_done = False
        self.task_goal_states = None

    def do_step(self, topdown_info, robot_environment, robot_sensors, robot_brain):

        if self.task_mode_enabled:
            if self.t_trial >= self.t_trial_max or self.goal_r_theta is None:  # need to start new trial
                if self.goal_r_theta is not None:  # not first trial of sim
                    self._add_trial_end_data(topdown_info)

                if self.trial >= self.num_trials_per_set or self.goal_r_theta is None:  # need to start new set
                    self.trial = 0
                    if self.goal_r_theta is not None:
                        self.set += 1
                    if self.set >= self.num_sets:
                        self.all_sets_done = True
                        return
                    else:  # not finished with all sets yet
                        # TODO sets params in robot_brain or other class according to self.sets_param_name and self.sets_param_values
                        if self.sets_param_name == 'brain.random_motor_out':
                            print 'setting ', self.sets_param_name, 'to value:', self.sets_param_values[self.set]
                            robot_brain.set_always_random_motor(self.sets_param_values[self.set])
                        else:
                            assert False, 'unsupported param name: ' + str(self.sets_param_name)
                else:  # new trial, but same set
                    self.trial += 1

                self.global_trial += 1
                self.t_trial = 0

                if self.sleep_every_trial > 0:
                    time.sleep(self.sleep_every_trial)

                self.goal_r_x, self.goal_r_y, self.goal_r_theta = self._choose_new_trial_goal(topdown_info=topdown_info)

                task_goal_nonzero_tiles = robot_environment.get_nonzero_tiles(robot_position_angle=(self.goal_r_x, self.goal_r_y, self.goal_r_theta))
                task_goal_rays = robot_sensors.get_rays(nonzero_tiles=task_goal_nonzero_tiles,
                                                        robot_position_angle=(self.goal_r_x, self.goal_r_y, self.goal_r_theta))
                task_goal_sensory_input = task_goal_rays['ray_colors']
                self.task_goal_states = robot_brain.get_autoencoder_states_for_input(net_input=task_goal_sensory_input)

                robot_brain.reset_for_new_task()
            else:  # don't need to start new trial
                self.t_trial += 1

        self.step += 1

    def finished_sim(self):
        '''
        if task mode enabled, compute based on trials and trial sets. otherwise, max time.
        :return:
        '''

        if self.task_mode_enabled:
            return self.all_sets_done
        else:
            if self.step >= self.run_steps_if_task_mode_disabled:
                return True
            else:
                return False

    def evaluate(self, plots_save_folder):
        '''
        Note: self.task_mode_enabled may be False, but may have been true in the past
        :param plots_save_folder:
        :return:
        '''
        pass

    def get_task_goal_states(self):
        return self.task_goal_states

    def get_current_goal_position_angle(self):
        '''
        used by visualizer
        :return:
        '''

        return self.goal_r_x, self.goal_r_y, self.goal_r_theta

    # *********************************** PRIVATE *************************************

    def _add_trial_end_data(self, topdown_info):
        if False:
            goal_p_x = goal_state[0]
            goal_p_y = goal_state[1]
            goal_theta = goal_state[2]

            robot_p_x = robot_state[0]
            robot_p_y = robot_state[1]
            robot_theta = robot_state[2]

            self.position_errors_per_trial.append(sqrt(pow(goal_p_x - robot_p_x, 2) + pow(goal_p_y - robot_p_y, 2)))
            self.theta_errors_per_trial.append(min(min(abs(robot_theta - goal_theta), abs(robot_theta + 2 * pi - goal_theta)), abs(robot_theta - 2 * pi - goal_theta)))

    def _choose_new_trial_goal(self, topdown_info):
        '''
        if task manager enabled, picks goal state given: current environment state.
        runs autoencoder to get goal state.
        :param topdown_info:
        :return:
        '''

        # get current robot position
        # pick new robot position (based on angle/distance)

        td_info = topdown_info
        r_x = td_info['robot_x']
        r_y = td_info['robot_y']
        r_theta = td_info['robot_theta']
        W = td_info['env_map_copy'].shape[1]
        H = td_info['env_map_copy'].shape[0]

        if self.constrain_to_params:

            # TODO fix this so that staying in bounds takes priority over constrain to params?
            # TODO or, always start robot in position where goal can be defined

            distance = self.min_distance + random.random() * (self.max_distance - self.min_distance)
            #delta_theta = self.min_delta_theta + random.random() * (self.max_delta_theta - self.min_delta_theta)
            r1 = random.random()
            if r1 < 0.5:
                delta_theta = self.min_delta_theta
            else:
                delta_theta = self.max_delta_theta

            goal_r_theta = r_theta + delta_theta
            # make sure normalized same way as robot theta
            while goal_r_theta > 2 * pi:
                goal_r_theta -= 2 * pi
            while goal_r_theta < 0:
                goal_r_theta += 2 * pi

            goal_r_x = r_x + distance * cos(goal_r_theta)
            goal_r_y = r_y + distance * sin(goal_r_theta)
        else:
            goal_r_theta = random.random() * 2.0 * pi
            goal_r_x = 2.0 + random.random() * (W - 4.0)
            goal_r_y = 2.0 + random.random() * (H - 4.0)

        return goal_r_x, goal_r_y, goal_r_theta
