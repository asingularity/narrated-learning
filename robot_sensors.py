
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

    def get_rays(self, nonzero_tiles=None, robot_position_angle=None):
        if nonzero_tiles is None or robot_position_angle is None:
            return {
                'ray_radians': self.relative_ray_radians,
                'ray_colors': self.ray_colors,
                'ray_lengths': self.ray_lengths
            }
        else:
            ray_colors, ray_lengths, relative_ray_radians = \
                self._compute_rays_for_env_and_theta(nonzero_tiles=nonzero_tiles,
                                                     robot_theta=robot_position_angle[2])
            return {
                'ray_radians': relative_ray_radians,
                'ray_colors': ray_colors,
                'ray_lengths': ray_lengths
            }

    def read_input(self, nonzero_tiles, robot_theta):
        self.ray_colors, self.ray_lengths, self.relative_ray_radians = \
            self._compute_rays_for_env_and_theta(nonzero_tiles=nonzero_tiles,
                                                 robot_theta=robot_theta)

    def _compute_rays_for_env_and_theta(self, nonzero_tiles, robot_theta):
        ray_colors = np.zeros((self.num_rays, 3))
        ray_lengths = np.zeros(self.num_rays)

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
                matching_tile_colors = color[matching_tile_indices, :]
                matching_tile_dists = dist[matching_tile_indices]
                min_tile = np.argmin(matching_tile_dists)
                ray_colors[k, :] = matching_tile_colors[min_tile, :]
                ray_lengths[k] = matching_tile_dists[min_tile]
            else:
                ray_colors[k, :] = np.zeros(3)
                # TODO make this a parameter (max ray length):
                ray_lengths[k] = 1000

        #self.relative_ray_radians = relative_ray_radians

        return ray_colors.flatten(), ray_lengths, relative_ray_radians


























