import sys
import traceback
import numpy as np
from math import pi
import multiprocessing as mp
from time import time


def compute_a_ray(params):
    try:
        k = params['k']
        ray_theta = params['ray_theta']
        theta = params['theta']
        d_theta = params['d_theta']
        color = params['color']
        dist = params['dist']

        # TODO fix discontinuity issue! This fixes it but at large computational cost:
        matching_tile_indices = np.nonzero(
            np.logical_or(np.logical_and(ray_theta > theta - d_theta, ray_theta < theta + d_theta),
                          np.logical_and(ray_theta > theta - d_theta + 2 * pi, ray_theta < theta + d_theta + 2 * pi)))[0]
        # matching_tile_indices = np.nonzero(np.logical_and(ray_theta > theta - d_theta, ray_theta < theta + d_theta))[0]
        if matching_tile_indices.shape[0] > 0:
            matching_tile_colors = color[matching_tile_indices, :]
            matching_tile_dists = dist[matching_tile_indices]
            min_tile = np.argmin(matching_tile_dists)
            # print k, matching_tile_dists, min_tile, matching_tile_dists[min_tile]
            ray_color = matching_tile_colors[min_tile, :]
            ray_length = matching_tile_dists[min_tile]
        else:
            # print k, None
            ray_color = np.zeros(3)
            # TODO make this a parameter (max ray length):
            ray_length = 1000

            # self.relative_ray_radians = relative_ray_radians

        return {
                'k': k,
                'ray_color': ray_color,
                'ray_length': ray_length
                }
    except:
        # Put all exception text into an exception and raise that
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


class RobotSensors(object):
    def __init__(self, params):
        self.num_rays = params['num_rays']
        fov_degrees = params['fov_degrees']
        num_processes = 6

        # relative to robot angle:
        self.ray_degrees = np.linspace(-fov_degrees / 2.0, fov_degrees / 2.0, self.num_rays)
        self.ray_radians = (pi / 180.0) * self.ray_degrees
        self.relative_ray_radians = self.ray_radians.copy()
        self.ray_colors = np.zeros(self.num_rays)
        self.ray_lengths = np.zeros(self.num_rays)

        self.pool = mp.Pool(processes=num_processes)
        self.ray_params = []
        for k in range(self.num_rays):
            self.ray_params.append({
                'k': k,
                'ray_theta': None,
                'theta': None,
                'd_theta': None,
                'color': None,
                'dist': None
            })

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

        use_mp = False

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

        if not use_mp:

            for k in range(relative_ray_radians.shape[0]):
                ray_theta = relative_ray_radians[k]

                # TODO fix discontinuity issue! This fixes it but at large computational cost:
                matching_tile_indices = np.nonzero(np.logical_or(np.logical_and(ray_theta > theta - d_theta, ray_theta < theta + d_theta), np.logical_and(ray_theta > theta - d_theta + 2 * pi, ray_theta < theta + d_theta + 2 * pi)))[0]
                #matching_tile_indices = np.nonzero(np.logical_and(ray_theta > theta - d_theta, ray_theta < theta + d_theta))[0]
                if matching_tile_indices.shape[0] > 0:
                    matching_tile_colors = color[matching_tile_indices, :]
                    matching_tile_dists = dist[matching_tile_indices]
                    min_tile = np.argmin(matching_tile_dists)
                    #print k, matching_tile_dists, min_tile, matching_tile_dists[min_tile]
                    ray_colors[k, :] = matching_tile_colors[min_tile, :]
                    ray_lengths[k] = matching_tile_dists[min_tile]
                else:
                    #print k, None
                    ray_colors[k, :] = np.zeros(3)
                    # TODO make this a parameter (max ray length):
                    ray_lengths[k] = 1000

            #self.relative_ray_radians = relative_ray_radians

            return ray_colors.flatten(), ray_lengths, relative_ray_radians
        else:
            for k in range(self.num_rays):
                self.ray_params[k]['ray_theta'] = relative_ray_radians[k]
                self.ray_params[k]['theta'] = theta
                self.ray_params[k]['d_theta'] = d_theta
                self.ray_params[k]['color'] = color
                self.ray_params[k]['dist'] = dist

            # range(relative_ray_radians.shape[0])
            res = self.pool.map(compute_a_ray, self.ray_params)
            for res_unit in res:
                k = res_unit['k']
                ray_colors[k, :] = res_unit['ray_color']
                ray_lengths[k] = res_unit['ray_length']

            return ray_colors.flatten(), ray_lengths, relative_ray_radians


























