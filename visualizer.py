import time
import sys
import cv2
import numpy as np
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

    def _display_fps(self):
        if time.time() - self.last_FPS_time > self.fps_display_interval:
            fps = self.fps_frames * 1.0 / (time.time() - self.last_FPS_time)
            logging.info('FPS: ' + str(fps))
            self.fps_frames = 0
            self.last_FPS_time = time.time()
        self.fps_frames += 1

    def _display_graphic_map(self, robot_environment, robot_sensors):
        im, robot_x, robot_y, robot_theta = robot_environment.get_topdown_info()
        rays = robot_sensors.get_rays()
        ray_radians = rays['ray_radians']
        ray_colors = rays['ray_colors']

        im[robot_y, robot_x] = 1.0

        resized_image = cv2.resize(src=im, dsize=(0, 0), fx=self.scale_topdown_factor, fy=self.scale_topdown_factor, interpolation=cv2.INTER_NEAREST)

        # TODO rays should be drawn on resized image? so always width 1

        cv2.imshow('env_map', resized_image)

        camera_image = np.zeros((1, ray_colors.shape[0]))
        camera_image[0, :] = ray_colors[:]
        resized_camera = cv2.resize(src=camera_image, dsize=(0, 0), fx=self.scale_camera_factor, fy=self.scale_camera_factor, interpolation=cv2.INTER_NEAREST)

        cv2.imshow('camera', resized_camera)
        cv2.waitKey(1)

    def visualize(self, robot_sensors, robot_brain, robot_model, robot_environment):
        if self.frames % self.image_display_frames == 0:
            self._display_graphic_map(robot_environment, robot_sensors)

        self._display_fps()

        self.frames += 1
