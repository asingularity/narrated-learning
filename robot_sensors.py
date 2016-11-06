
import numpy as np
from math import pi


class RobotSensors(object):
    def __init__(self, params):
        num_rays = params['num_rays']
        fov_degrees = params['fov_degrees']

        # relative to robot angle:
        self.ray_degrees = np.linspace(-fov_degrees / 2.0, fov_degrees / 2.0, num_rays)
        self.ray_radians = (pi / 180.0) * self.ray_degrees
        self.relative_ray_radians = self.ray_radians.copy()
        self.ray_colors = np.zeros(num_rays)
        self.ray_lengths = np.zeros(num_rays)

    def get_rays(self):
        return {
            'ray_radians': self.relative_ray_radians,
            'ray_colors': self.ray_colors,
            'ray_lengths': self.ray_lengths
        }

    def read_input(self, robot_environment):
        nonzero_tiles = robot_environment.get_nonzero_tiles()
        dist = nonzero_tiles['nonzero_dist']
        theta = nonzero_tiles['nonzero_theta']
        d_theta = nonzero_tiles['nonzero_delta_theta']
        color = nonzero_tiles['nonzero_color']

        robot_info = robot_environment.get_robot_info()
        robot_x = robot_info['robot_x']
        robot_y = robot_info['robot_y']
        robot_theta = robot_info['robot_theta']

        # for each ray, find which tiles are within theta for it
        #   then pick color of minimum distance tile
        relative_ray_radians = robot_theta + self.ray_radians

        more_than_360 = np.nonzero(relative_ray_radians >= 2 * pi)
        less_than_0 = np.nonzero(relative_ray_radians < 0.0)
        relative_ray_radians[more_than_360] -= 2 * pi
        relative_ray_radians[less_than_0] += 2 * pi

        for k in range(relative_ray_radians.shape[0]):
            ray_theta = relative_ray_radians[k]

            # TODO fix discontinuity issue!
            #matching_tile_indices = np.nonzero(np.logical_or(np.logical_and(ray_theta > theta - d_theta, ray_theta < theta + d_theta), np.logical_and(ray_theta + 180. > theta + 180. - d_theta, ray_theta + 180. < theta + 180. + d_theta)))[0]
            matching_tile_indices = np.nonzero(np.logical_and(ray_theta > theta - d_theta, ray_theta < theta + d_theta))[0]
            if matching_tile_indices.shape[0] > 0:
                matching_tile_colors = color[matching_tile_indices]
                matching_tile_dists = dist[matching_tile_indices]
                min_tile = np.argmin(matching_tile_dists)
                self.ray_colors[k] = matching_tile_colors[min_tile]
                self.ray_lengths[k] = matching_tile_dists[min_tile]
            else:
                self.ray_colors[k] = 0

                # TODO make this a parameter (max ray length):
                self.ray_lengths[k] = 1000

        self.relative_ray_radians = relative_ray_radians
