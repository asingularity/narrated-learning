import time
import sys
from math import sin, cos
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)


class Visualizer(object):
    def __init__(self, params):
        self.fps_frames = 0
        self.frames = 0
        self.last_FPS_time = time.time()
        self.fps_display_interval = params['fps_display_interval']
        self.scale_topdown_factor = params['scale_topdown_factor']
        self.scale_camera_factor = params['scale_camera_factor']
        self.image_display_frames = params['image_display_frames']
        self.plot_brain_error_frames =params['plot_brain_error_frames']
        self.waitKey_time = params['waitKey_time']
        self.no_wall_ray_color = params['no_wall_ray_color']
        self.plots_folder = params['plots_folder']
        self.linear_speed_from_key = 0.0
        self.angular_speed_from_key = 0.0

        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(1, 1, 0)

    def get_linear_angular_speed(self):
        return self.linear_speed_from_key, self.angular_speed_from_key

    def _display_fps(self):
        if time.time() - self.last_FPS_time > self.fps_display_interval:
            fps = self.fps_frames * 1.0 / (time.time() - self.last_FPS_time)
            logging.info('FPS: ' + str(fps))
            self.fps_frames = 0
            self.last_FPS_time = time.time()
        self.fps_frames += 1

    def _display_graphic_map(self, robot_environment, robot_sensors, robot_brain):
        topdown_info = robot_environment.get_topdown_info()

        im = topdown_info['env_map_copy']
        robot_x = topdown_info['robot_x']
        robot_y = topdown_info['robot_y']
        round_x = topdown_info['round_robot_x']
        round_y = topdown_info['round_robot_y']
        robot_theta = topdown_info['robot_theta']

        rays = robot_sensors.get_rays()
        ray_radians = rays['ray_radians']
        ray_colors = rays['ray_colors']
        ray_lengths = rays['ray_lengths']

        autoenc_images = robot_brain.get_autoenc_images()
        resized_autoenc = cv2.resize(src=autoenc_images, dsize=(0, 0), fx=self.scale_camera_factor, fy=self.scale_camera_factor, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('autoenc', resized_autoenc)

        im[round_y, round_x] = 1.0

        resized_image = cv2.resize(src=im, dsize=(0, 0), fx=self.scale_topdown_factor, fy=self.scale_topdown_factor, interpolation=cv2.INTER_NEAREST)

        # TODO rays should be drawn on resized image? so always width 1

        for k in range(ray_colors.shape[0]):

            pt2_x = robot_x + ray_lengths[k] * cos(ray_radians[k])
            pt2_y = robot_y + ray_lengths[k] * sin(ray_radians[k])
            ray_color = ray_colors[k]
            if ray_color == 0:
                ray_color = self.no_wall_ray_color

            cv2.line(resized_image,
                     pt1=(int(robot_x * self.scale_topdown_factor), int(robot_y * self.scale_topdown_factor)),
                     pt2=(int(pt2_x * self.scale_topdown_factor), int(pt2_y * self.scale_topdown_factor)),
                     color=ray_color,
                     thickness=1)

        cv2.imshow('env_map', resized_image)

        camera_image = np.zeros((1, ray_colors.shape[0]))
        camera_image[0, :] = ray_colors[:]
        resized_camera = cv2.resize(src=camera_image, dsize=(0, 0), fx=self.scale_camera_factor, fy=self.scale_camera_factor, interpolation=cv2.INTER_NEAREST)

        cv2.imshow('camera', resized_camera)
        k = cv2.waitKey(self.waitKey_time)

        FWD = 119
        BACK = 115
        LEFT = 97
        RIGHT = 100

        if k == -1:
            self.linear_speed_from_key = 0.0
            self.angular_speed_from_key = 0.0
        else:
            #print 'KEY PRESSED: ' + str(k)
            if k == FWD:
                self.linear_speed_from_key = 0.25
            if k == BACK:
                self.linear_speed_from_key = -0.25
            if k == LEFT:
                self.angular_speed_from_key = -0.2
            if k == RIGHT:
                self.angular_speed_from_key = 0.2

    def _plot_brain_errors(self, robot_brain):
        error_names_autoenc, error_histories_autoenc, error_names_predictor, error_histories_predictor, error_names_no_context_predictor, error_histories_no_context_predictor = robot_brain.get_error_names_histories()
        print self.frames

        for k in range(len(error_names_autoenc)):
            error_name = error_names_autoenc[k]
            error_history = error_histories_autoenc[k, :]
            print error_name
            print error_history.shape
            self.ax.cla()
            self.ax.plot(error_history)
            self.fig.savefig(self.plots_folder + '/' + error_name + '.png', dpi=100)

        for k in range(len(error_names_predictor)):
            error_name = error_names_predictor[k]
            error_history = error_histories_predictor[k, :]
            print error_name
            print error_history.shape
            self.ax.cla()
            self.ax.plot(error_history, 'b-')

            error_name_nc = error_names_no_context_predictor[k]
            error_history_nc = error_histories_no_context_predictor[k, :]
            self.ax.plot(error_history_nc, 'r-')
            self.fig.savefig(self.plots_folder + '/' + error_name + '_' + error_name_nc + '.png', dpi=100)

    def visualize(self, robot_sensors, robot_brain, robot_model, robot_environment):
        if self.frames % self.image_display_frames == 0:
            self._display_graphic_map(robot_environment, robot_sensors, robot_brain)

        if self.frames % self.plot_brain_error_frames == 0:
            self._plot_brain_errors(robot_brain)

        self._display_fps()

        self.frames += 1
