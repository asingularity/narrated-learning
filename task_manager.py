
from math import pi, sin, cos
import numpy as np
import random


class TaskManager(object):
    def __init__(self, params):
        self.enabled = params['enabled']
        self.t_task_max = params['max_task_steps']
        self.min_delta_theta = params['min_delta_theta']  # radians
        self.max_delta_theta = params['max_delta_theta']  # radians
        self.min_distance = params['min_distance']
        self.max_distance = params['max_distance']
        self.t_task = None

        self.goal_states = None
        self.goal_r_x = None
        self.goal_r_y = None
        self.goal_r_theta = None

    def choose_new_task_goal(self, topdown_info):
        '''
        if task manager enabled, picks goal state given: current environment state.
        runs autoencoder to get goal state.
        :param topdown_info:
        :return:
        '''

        if self.enabled:
            # get current robot position
            # pick new robot position (based on angle/distance)

            td_info = topdown_info
            r_x = td_info['robot_x']
            r_y = td_info['robot_y']
            r_theta = td_info['robot_theta']
            W = td_info['env_map_copy'].shape[1]
            H = td_info['env_map_copy'].shape[0]

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

            self.goal_r_theta = goal_r_theta
            self.goal_r_x = r_x + distance * cos(goal_r_theta)
            self.goal_r_y = r_y + distance * sin(goal_r_theta)

            self.t_task = 0

    def get_enabled(self):
        return self.enabled

    def get_current_goal_position_angle(self):
        '''
        used by visualizer.visualize to show goal position and angle
        :return:
        '''

        return self.goal_r_x, self.goal_r_y, self.goal_r_theta

    def finished_task(self):
        '''
        task manager should look at: elapsed time, location vs. goal location. if disabled, return true always
        initially, just elapsed time.
        :param robot_environment:
        :return:
        '''
        if self.enabled:
            self.t_task += 1
            return self.t_task >= self.t_task_max
        else:
            return True
