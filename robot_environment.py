import numpy as np
import os
import cv2
from math import sqrt, atan2, atan


class RobotEnvironment(object):
    def __init__(self, params):
        self.W = params['width']  # Width/Height
        self.H = params['height']

        self.r_x = params['init_robot_x']
        self.r_y = params['init_robot_y']
        self.r_theta = params['init_robot_theta']

        env_map = np.zeros((self.H, self.W))
        walls = params['walls']
        for wall in walls:
            if wall['orientation'] == 'horizontal':
                env_map[wall['y_start'], wall['x_start']:wall['x_start'] + wall['length']] = wall['color']
            elif wall['orientation'] == 'vertical':
                env_map[wall['y_start']:wall['y_start'] + wall['length'], wall['x_start']] = wall['color']
            else:
                assert False, 'invalid orientation: ' + str(wall['orientation'])
        self.env_map = env_map

        # TODO store nonzero elements of env_map explicitly:


        self.theta, self.delta_theta, self.dist = self._load_or_precompute_angles_dist(rows=self.H, cols=self.W)

    def _load_or_precompute_angles_dist(self, rows, cols):
        # TODO: save/load from file
        # if os.path.isfile(params['precomputed_environments_folder'] + '/env_' + str(self.W) + 'x' + str(self.H) + '.env'):

        dist = np.zeros((rows, cols, rows, cols))
        theta = np.zeros((rows, cols, rows, cols))
        delta_theta = np.zeros((rows, cols, rows, cols))

        for r1 in range(rows):
            print 'processing: ', r1, ' of ', rows
            for c1 in range(cols):
                for r2 in range(rows):
                    for c2 in range(cols):
                        # (r1, c1) is robot position
                        dist[r1, c1, r2, c2] = sqrt(pow(r1 - r2, 2) + pow(c1 - c2, 2))
                        theta[r1, c1, r2, c2] = atan2(r2 - r1, c2 - c1)
                        delta_theta[r1, c1, r2, c2] = atan(0.5 / dist[r1, c1, r2, c2])
                        # angle is theta +/- delta_theta

        return theta, delta_theta, dist

    def step_environment(self, robot_model):
        delta_velocity, delta_theta = robot_model.get_delta_configuration()

    def get_topdown_info(self):
        # TODO return info necessary for image. also needs to return rays.

        return self.env_map.copy(), self.r_x, self.r_y, self.r_theta

    def get_camera_image(self):
        return 0
