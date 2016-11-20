import numpy as np
import os
import cv2
from math import sqrt, atan2, atan, pi, sin, cos
import time


class RobotEnvironment(object):
    def __init__(self, params):
        self.W = params['width']  # Width/Height
        self.H = params['height']

        self.r_x = params['init_robot_x']
        self.r_y = params['init_robot_y']
        self.last_r_x = self.r_x
        self.last_r_y = self.r_y
        self.r_theta = params['init_robot_theta']

        self.use_keyboard_input = params['use_keyboard_input']

        self.round_x = int(round(self.r_x))
        self.round_y = int(round(self.r_y))

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

        # store nonzero elements of env_map explicitly:
        self.nonzero_map_y = np.nonzero(env_map)[0]
        self.nonzero_map_x = np.nonzero(env_map)[1]
        self.nonzero_map_color = env_map[np.nonzero(env_map)]

        self.theta, self.delta_theta, self.dist = self._load_or_precompute_angles_dist(rows=self.H, cols=self.W)

        self.nonzero_tiles = self._get_nonzero_tiles()

    def _load_or_precompute_angles_dist(self, rows, cols):
        # TODO: save/load from file
        # if os.path.isfile(params['precomputed_environments_folder'] + '/env_' + str(self.W) + 'x' + str(self.H) + '.env'):

        dist = np.zeros((rows, cols, rows, cols))
        theta = np.zeros((rows, cols, rows, cols))
        delta_theta = np.zeros((rows, cols, rows, cols))

        debug_min_theta = np.inf
        debug_max_theta = -np.inf

        for r1 in range(rows):
            print 'processing: ', r1, ' of ', rows
            for c1 in range(cols):
                for r2 in range(rows):
                    for c2 in range(cols):
                        if r1 == r2 and c1 == c2:
                            # TODO setting to inf is right for display... but wrong for proximity/collision check
                            dist[r1, c1, r2, c2] = 0.0  # np.inf
                        else:

                            theta_temp = atan2(r2 - r1, c2 - c1)

                            while theta_temp > 2 * pi:
                                theta_temp -= 2 * pi
                            while theta_temp < 0:
                                theta_temp += 2 * pi

                            if theta_temp < debug_min_theta:
                                debug_min_theta = theta_temp
                            if theta_temp > debug_max_theta:
                                debug_max_theta = theta_temp

                            # (r1, c1) is robot position
                            dist[r1, c1, r2, c2] = sqrt(pow(r1 - r2, 2) + pow(c1 - c2, 2))
                            theta[r1, c1, r2, c2] = theta_temp
                            delta_theta[r1, c1, r2, c2] = atan(0.5 / dist[r1, c1, r2, c2])
                            # angle is theta +/- delta_theta

        print ('min theta: ' + str(debug_min_theta) + ', max theta: ' + str(debug_max_theta))
        return theta, delta_theta, dist

    def step_environment(self, robot_model, visualizer):
        # visualizer is an input for reading keyboard input
        if self.use_keyboard_input:
            linear_speed, angular_speed = visualizer.get_linear_angular_speed()
        else:
            linear_speed, angular_speed = robot_model.get_delta_configuration()


        # TODO new logic:
        #   do collision detection on current x, y, and also velocity x, y

        self.r_theta += angular_speed

        while self.r_theta > 2 * pi:
            self.r_theta -= 2 * pi
        while self.r_theta < 0:
            self.r_theta += 2 * pi

        self.last_r_x = self.r_x
        self.last_r_y = self.r_y
        self.r_x += linear_speed * cos(self.r_theta)
        self.r_y += linear_speed * sin(self.r_theta)

        self.round_x = int(self.r_x)
        self.round_y = int(self.r_y)

        if self.round_x > self.W - 1:
            self.round_x = self.W - 1
            self.r_x = self.round_x
        if self.round_x < 0:
            self.round_x = 0
            self.r_x = self.round_x
        if self.round_y > self.H - 1:
            self.round_y = self.H - 1
            self.r_y = self.round_y
        if self.round_y < 0:
            self.round_y = 0
            self.r_y = self.round_y

        self.nonzero_tiles = self._get_nonzero_tiles()

        nz_dist = self.nonzero_tiles['nonzero_dist']
        close_nnz_tile_indices = np.nonzero((nz_dist <= 1 + 1e-9))[0]
        if not (self.r_x == self.last_r_x and self.r_y == self.last_r_y):
            if close_nnz_tile_indices.shape[0] > 0:
                m_x_neighbors = self.nonzero_map_x[close_nnz_tile_indices]
                m_y_neighbors = self.nonzero_map_y[close_nnz_tile_indices]

                for m_x, m_y in zip(m_x_neighbors.tolist(), m_y_neighbors.tolist()):
                    if not self.round_x == m_x:
                        if self.round_x > m_x:
                            if self.r_x < self.last_r_x:
                                self.r_x = int(self.r_x) + 0.5
                        if self.round_x < m_x:
                            if self.r_x > self.last_r_x:
                                self.r_x = int(self.r_x) + 0.5
                        self.round_x = int(self.r_x)
                    if not self.round_y == m_y:
                        if self.round_y > m_y:
                            if self.r_y < self.last_r_y:
                                self.r_y = int(self.r_y) + 0.5
                        if self.round_y < m_y:
                            if self.r_y > self.last_r_y:
                                self.r_y = int(self.r_y) + 0.5
                        self.round_y = int(self.r_y)
            else:
                pass

    def _get_nonzero_tiles(self):
        round_x = self.round_x
        round_y = self.round_y
        dist_from_robot = self.dist[round_y, round_x, :, :]
        theta_from_robot = self.theta[round_y, round_x, :, :]
        delta_theta_from_robot = self.delta_theta[round_y, round_x, :, :]

        nonzero_dist = dist_from_robot[self.nonzero_map_y, self.nonzero_map_x]
        nonzero_theta = theta_from_robot[self.nonzero_map_y, self.nonzero_map_x]
        nonzero_delta_theta = delta_theta_from_robot[self.nonzero_map_y, self.nonzero_map_x]

        return {
            'nonzero_dist': nonzero_dist,
            'nonzero_theta': nonzero_theta,
            'nonzero_delta_theta': nonzero_delta_theta,
            'nonzero_color': self.nonzero_map_color
        }

    def get_nonzero_tiles(self):
        return self.nonzero_tiles

    def get_robot_theta(self):
        return {
            'robot_theta': self.r_theta
        }

    def get_topdown_info(self):
        return {
            'env_map_copy': self.env_map.copy(),
            'robot_x': self.r_x,
            'robot_y': self.r_y,
            'round_robot_x': self.round_x,
            'round_robot_y': self.round_y,
            'robot_theta': self.r_theta
        }

    def get_camera_image(self):
        return 0
