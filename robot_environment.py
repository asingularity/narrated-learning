import numpy as np
import os
import cv2
from math import sqrt, atan2, atan, pi, sin, cos
import time
import random


class RobotEnvironment(object):
    def __init__(self, params):
        self.W = params['width']  # Width/Height
        self.H = params['height']

        self.r_x = params['init_robot_x']
        self.r_y = params['init_robot_y']
        self.last_r_x = self.r_x
        self.last_r_y = self.r_y
        self.r_theta = params['init_robot_theta']

        self.round_x = int(round(self.r_x))
        self.round_y = int(round(self.r_y))

        min_num_walls = params['min_num_walls']
        max_num_walls = params['max_num_walls']
        wall_min_length = params['wall_min_length']
        wall_max_length = params['wall_max_length']
        min_space_between_walls = params['min_space_between_walls']

        num_walls = random.randint(min_num_walls, max_num_walls)

        env_map = np.zeros((self.H, self.W, 3))
        overlap_map = np.zeros((self.H, self.W, 3))

        # TODO put walls along edges first?

        for w in range(num_walls):
            wall_placed = False
            while not wall_placed:
                wall_length = random.randint(wall_min_length, wall_max_length)
                wall_color = np.array([random.random(), random.random(), random.random()])
                wall_x_start = random.randint(0, self.W - 1)
                wall_y_start = random.randint(0, self.H - 1)
                wall_orient = random.randint(0, 1)

                wall_placed, env_map, overlap_map = self._attempt_place_wall(wall_color,
                                                       wall_orient,
                                                       wall_length,
                                                       wall_x_start,
                                                       wall_y_start,
                                                       overlap_map,
                                                       env_map,
                                                       self.W,
                                                       self.H,
                                                       min_space_between_walls)

        debug_view = False
        if debug_view:
            resized_image = cv2.resize(src=env_map, dsize=(0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
            cv2.imshow('map init', resized_image)
            cv2.waitKey(600000)

        self.env_map = env_map
        env_map_sum = np.sum(env_map, axis=2)

        # store nonzero elements of env_map explicitly:
        self.nonzero_map_y = np.nonzero(env_map_sum)[0]
        self.nonzero_map_x = np.nonzero(env_map_sum)[1]
        # TODO should store tuple or list in each position:
        self.nonzero_map_color = env_map[np.nonzero(env_map_sum)]
        # print self.nonzero_map_color.shape  # (150, 3)

        self.theta, self.delta_theta, self.dist = self._load_or_precompute_angles_dist(rows=self.H, cols=self.W)

        self.nonzero_tiles = self._get_nonzero_tiles()

    def _attempt_place_wall(self, wall_color, wall_orient, wall_length, wall_x_start, wall_y_start, overlap_map, env_map, W, H, min_space_between_walls):
        if wall_orient == 0:

            wall_x_end = wall_x_start + wall_length - 1
            if wall_x_end > W - 1:
                return False, env_map, overlap_map

            new_overlap_map = overlap_map.copy()
            new_env_map = env_map.copy()

            for x in range(wall_x_start - min_space_between_walls, wall_x_end + min_space_between_walls + 1):
            #for x in range(wall_x_start - 0, wall_x_end + 0 + 1):
                for y in range(wall_y_start - min_space_between_walls, wall_y_start + min_space_between_walls + 1):
                    if x >= 0 and x < W and y >= 0 and y < H:
                        if np.sum(overlap_map[y, x, :]) > 0:
                            return False, env_map, overlap_map
                        new_overlap_map[y, x, :] = wall_color[:]
            for x in range(wall_x_start, wall_x_end + 1):
                new_env_map[x, wall_y_start, :] = wall_color[:]

        elif wall_orient == 1:
            wall_y_end = wall_y_start + wall_length - 1
            if wall_y_end > H - 1:
                return False, env_map, overlap_map

            new_overlap_map = overlap_map.copy()
            new_env_map = env_map.copy()

            for x in range(wall_x_start - min_space_between_walls, wall_x_start + min_space_between_walls + 1):
                #for y in range(wall_y_start - 0, wall_y_end + 0 + 1):
                for y in range(wall_y_start - min_space_between_walls, wall_y_end + min_space_between_walls + 1):
                    if x >= 0 and x < W and y >= 0 and y < H:
                        if np.sum(overlap_map[y, x, :]) > 0:
                            return False, env_map, overlap_map
                        new_overlap_map[y, x, :] = wall_color[:]
            for y in range(wall_y_start, wall_y_end + 1):
                new_env_map[wall_x_start, y, :] = wall_color[:]

        else:
            assert False

        return True, new_env_map, new_overlap_map

    def get_topdown_info(self):
        return {
            'env_map_copy': self.env_map.copy(),
            'robot_x': self.r_x,
            'robot_y': self.r_y,
            'round_robot_x': self.round_x,
            'round_robot_y': self.round_y,
            'robot_theta': self.r_theta
        }

    def get_robot_theta(self):
        return self.r_theta

    def get_nonzero_tiles(self, robot_position_angle=None):
        if robot_position_angle is None:
            return self.nonzero_tiles
        else:
            return self._get_nonzero_tiles(robot_x=robot_position_angle[0], robot_y=robot_position_angle[1])

    def step_environment(self, linear_speed, angular_speed):
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
            #print 'self.round_x > self.W - 1'
            self.round_x = self.W - 1
            self.r_x = self.round_x
            self.r_theta = pi
        if self.round_x < 0:
            #print 'self.round_x < 0'
            self.round_x = 0
            self.r_x = self.round_x
            self.r_theta = 0
        if self.round_y > self.H - 1:
            #print 'self.round_y > self.H - 1'
            self.round_y = self.H - 1
            self.r_y = self.round_y
            self.r_theta = 3.0 * pi / 2.0
        if self.round_y < 0:
            #print 'self.round_y < 0'
            self.round_y = 0
            self.r_y = self.round_y
            self.r_theta = pi / 2.0

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

    def _get_nonzero_tiles(self, robot_x=None, robot_y=None):
        if robot_x is not None and robot_y is not None:
            round_x = max(min(robot_x, self.W - 1), 0)
            round_y = max(min(robot_y, self.H - 1), 0)
        else:
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
