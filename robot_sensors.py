
import numpy as np
from math import pi


class RobotSensors(object):
    def __init__(self, params):
        self.num_rays = params['num_rays']
        fov_degrees = params['fov_degrees']

        # relative to robot angle:
        self.ray_degrees = np.linspace(-fov_degrees / 2.0, fov_degrees / 2.0, self.num_rays)
        self.ray_radians = (pi / 180.0) * self.ray_degrees
        self.relative_ray_radians = self.ray_radians.copy()
        self.ray_colors = np.zeros(self.num_rays)
        self.ray_lengths = np.zeros(self.num_rays)
        # TODO fix this hack:
        self.last_motor_command = np.zeros(2).astype(np.float)

    def get_rays(self):
        return {
            'ray_radians': self.relative_ray_radians,
            'ray_colors': self.ray_colors,
            'ray_lengths': self.ray_lengths
        }

    def get_last_motor_command(self):
        return self.last_motor_command

    def _compute_rays_for_env_and_theta(self, env, robot_theta, robot_x, robot_y):
        ray_colors = np.zeros(self.num_rays)
        ray_lengths = np.zeros(self.num_rays)

        if robot_x is None:
            nonzero_tiles = env.get_nonzero_tiles()
        else:
            nonzero_tiles = env._get_nonzero_tiles(robot_x, robot_y)
        dist = nonzero_tiles['nonzero_dist']
        theta = nonzero_tiles['nonzero_theta']
        d_theta = nonzero_tiles['nonzero_delta_theta']
        color = nonzero_tiles['nonzero_color']

        # for each ray, find which tiles are within theta for it
        #   then pick color of minimum distance tile
        relative_ray_radians = robot_theta + self.ray_radians

        more_than_360 = np.nonzero(relative_ray_radians >= 2 * pi)
        less_than_0 = np.nonzero(relative_ray_radians < 0.0)
        relative_ray_radians[more_than_360] -= 2 * pi
        relative_ray_radians[less_than_0] += 2 * pi

        for k in range(relative_ray_radians.shape[0]):
            ray_theta = relative_ray_radians[k]

            # TODO fix discontinuity issue! This fixes it but at large computational cost:
            #matching_tile_indices = np.nonzero(np.logical_or(np.logical_and(ray_theta > theta - d_theta, ray_theta < theta + d_theta), np.logical_and(ray_theta > theta - d_theta + 2 * pi, ray_theta < theta + d_theta + 2 * pi)))[0]
            matching_tile_indices = np.nonzero(np.logical_and(ray_theta > theta - d_theta, ray_theta < theta + d_theta))[0]
            if matching_tile_indices.shape[0] > 0:
                matching_tile_colors = color[matching_tile_indices]
                matching_tile_dists = dist[matching_tile_indices]
                min_tile = np.argmin(matching_tile_dists)
                ray_colors[k] = matching_tile_colors[min_tile]
                ray_lengths[k] = matching_tile_dists[min_tile]
            else:
                ray_colors[k] = 0
                # TODO make this a parameter (max ray length):
                ray_lengths[k] = 1000

        #self.relative_ray_radians = relative_ray_radians

        return ray_colors, ray_lengths, relative_ray_radians

    def read_input(self, robot_environment, robot_model):
        self.last_motor_command = robot_model.get_last_motor_command()

        robot_info = robot_environment.get_robot_theta()
        robot_theta = robot_info['robot_theta']

        self.ray_colors, self.ray_lengths, self.relative_ray_radians = \
            self._compute_rays_for_env_and_theta(env=robot_environment, robot_theta=robot_theta,
                                                 robot_x=None, robot_y=None)



























