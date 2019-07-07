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

        self.plot_brain_error_frames =params['plot_brain_error_frames']
        self.no_wall_ray_color = params['no_wall_ray_color']
        self.waitKey_time_slow = params['waitKey_time_slow']
        self.image_display_secs_slow = params['image_display_secs_slow']
        self.waitKey_time_fast = params['waitKey_time_fast']
        self.image_display_secs_fast = params['image_display_secs_fast']
        self.auto_switch_to_slow_disp_time = params['auto_switch_to_slow_disp_time']

        self.show_table_ims = params['show_table_ims']

        self.linear_speed_from_key = 0.0
        self.angular_speed_from_key = 0.0

        if params['init_fast']:
            self.toggle_viewer_slow = False
            self.image_display_secs = self.image_display_secs_fast
            self.waitKey_time = self.waitKey_time_fast
        else:
            self.toggle_viewer_slow = True
            self.image_display_secs = self.image_display_secs_slow
            self.waitKey_time = self.waitKey_time_slow

        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.last_image_display_time = time.time()

    def get_linear_angular_speed(self):
        return self.linear_speed_from_key, self.angular_speed_from_key

    def _display_fps(self):
        if time.time() - self.last_FPS_time > self.fps_display_interval:
            fps = self.fps_frames * 1.0 / (time.time() - self.last_FPS_time)
            logging.info('FPS: ' + str(fps) + ', frames: ' + str(self.frames))
            self.fps_frames = 0
            self.last_FPS_time = time.time()
        self.fps_frames += 1

    def _get_topdown_map(self, rays, topdown_info, current_goal_position_angle):
        im = topdown_info['env_map_copy']
        robot_x = topdown_info['robot_x']
        robot_y = topdown_info['robot_y']
        round_x = topdown_info['round_robot_x']
        round_y = topdown_info['round_robot_y']
        robot_theta = topdown_info['robot_theta']

        ray_radians = rays['ray_radians']
        ray_colors = rays['ray_colors'].reshape((len(rays['ray_colors']) / 3, 3))
        ray_lengths = rays['ray_lengths']

        im[round_y, round_x] = 1.0

        resized_image = cv2.resize(src=im, dsize=(0, 0), fx=self.scale_topdown_factor, fy=self.scale_topdown_factor, interpolation=cv2.INTER_NEAREST)

        # rays should be drawn on resized image so always width 1

        goal_x, goal_y, goal_theta = current_goal_position_angle
        if goal_x is not None:
            cv2.circle(img=resized_image,
                       center=(int(goal_x * self.scale_topdown_factor), int(goal_y * self.scale_topdown_factor)),
                       radius=5,
                       color=(1.0, 0, 1.0),
                       thickness=3)
            g2_x = goal_x + 2. * cos(goal_theta)
            g2_y = goal_y + 2. * sin(goal_theta)
            cv2.line(resized_image,
                     pt1=(int(goal_x * self.scale_topdown_factor), int(goal_y * self.scale_topdown_factor)),
                     pt2=(int(g2_x * self.scale_topdown_factor), int(g2_y * self.scale_topdown_factor)),
                     color=(1.0, 1.0, 0),
                     thickness=2)

        for k in range(ray_colors.shape[0]):

            pt2_x = robot_x + ray_lengths[k] * cos(ray_radians[k])
            pt2_y = robot_y + ray_lengths[k] * sin(ray_radians[k])
            ray_color = ray_colors[k, :]
            if np.sum(ray_color) == 0:
                ray_color = self.no_wall_ray_color

            cv2.line(resized_image,
                     pt1=(int(robot_x * self.scale_topdown_factor), int(robot_y * self.scale_topdown_factor)),
                     pt2=(int(pt2_x * self.scale_topdown_factor), int(pt2_y * self.scale_topdown_factor)),
                     color=(ray_color[0], ray_color[1], ray_color[2]),
                     thickness=1)

        return resized_image

    def _display_graphic_map(self, rays, topdown_info, current_goal_position_angle):

        ray_colors = rays['ray_colors'].reshape((len(rays['ray_colors']) / 3, 3))
        resized_image = self._get_topdown_map(rays, topdown_info, current_goal_position_angle)

        cv2.imshow('env_map', resized_image)

        camera_image = np.zeros((1, ray_colors.shape[0], 3))
        camera_image[0, :, 0] = ray_colors[:, 0]
        camera_image[0, :, 1] = ray_colors[:, 1]
        camera_image[0, :, 2] = ray_colors[:, 2]
        resized_camera = cv2.resize(src=camera_image, dsize=(0, 0), fx=self.scale_camera_factor, fy=self.scale_camera_factor, interpolation=cv2.INTER_NEAREST)

        cv2.imshow('camera', resized_camera)

    def _toggle_with_key_press(self, last_key):
        k = last_key

        FWD = 119
        BACK = 115
        LEFT = 97
        RIGHT = 100
        ENTER = 13

        if k == -1:
            self.linear_speed_from_key = 0.0
            self.angular_speed_from_key = 0.0
        else:
            print( 'KEY PRESSED: ' + str(k))
            if k == ENTER:
                self.toggle_viewer_slow = not self.toggle_viewer_slow
                if self.toggle_viewer_slow:
                    self.image_display_secs = self.image_display_secs_slow
                    self.waitKey_time = self.waitKey_time_slow
                else:
                    self.image_display_secs = self.image_display_secs_fast
                    self.waitKey_time = self.waitKey_time_fast

            if k == FWD:
                self.linear_speed_from_key = 0.25
            if k == BACK:
                self.linear_speed_from_key = -0.25
            if k == LEFT:
                self.angular_speed_from_key = -0.2
            if k == RIGHT:
                self.angular_speed_from_key = 0.2

    def _display_table_ims(self, IO_im_list, C_im_list, W_im_list):

        k = 0
        # print('*** IO ***')
        for im in IO_im_list:
            if im is not None:
                cv2.imshow('IO_' + str(k), im)
                # print(k, im.shape, im.dtype, np.amin(im), np.amax(im))
            else:
                pass
                # print(k, None)
            k += 1

        k = 0
        # print('*** C ***')
        for im in C_im_list:
            if im is not None:
                cv2.imshow('C_' + str(k), im)
                # print(k,im.shape, im.dtype, np.amin(im), np.amax(im))
            else:
                pass
                # print(k, None)
            k += 1

        k = 0
        # print('*** W ***')
        for im in W_im_list:
            if im is not None:
                cv2.imshow('W_' + str(k), im)
                # print(k,im.shape, im.dtype, np.amin(im), np.amax(im))
            else:
                pass
                # print(k, None)
            k += 1

    def _concat_with_spacer(self, concat_im, new_im):
        if new_im is not None:

            if new_im.dtype == np.uint8:
                new_im = new_im.astype(np.float) * 1.0 / 255.0

            spacer = 0.2
            # adjust height
            # if concat_im taller: add spacer to new_im below it
            if concat_im.shape[0] > new_im.shape[0]:
                new_im = np.vstack((new_im, spacer * np.ones((concat_im.shape[0] - new_im.shape[0], new_im.shape[1]))))

            # if new_im taller: add spacer to concat_im below it
            if new_im.shape[0] > concat_im.shape[0]:
                concat_im = np.vstack((new_im, spacer * np.ones((new_im.shape[0] - concat_im.shape[0], concat_im.shape[1]))))

            concat_im = np.hstack((concat_im, spacer * np.ones((concat_im.shape[0], 20)), new_im))

        return concat_im

    def _display_table_ims_tiled(self, IO_im_list, C_im_list, W_im_list):
        '''

        | IO | C | W | /// | IO | C | W | /// ...

        :param IO_im_list:
        :param C_im_list:
        :param W_im_list:
        :return:
        '''

        concat_im = None

        cv2.imshow('IO_0', IO_im_list[0])

        for k in range(len(IO_im_list)):
            IO_im = IO_im_list[k]
            if k == 0:
                IO_im = None

            C_im = C_im_list[k]
            W_im = W_im_list[k]

            if concat_im is None and IO_im is not None:
                concat_im = IO_im.copy()
                if concat_im.dtype == np.uint8:
                    concat_im = concat_im.astype(np.float) * 1.0 / 255.0
            else:
                concat_im = self._concat_with_spacer(concat_im=concat_im, new_im=IO_im)

            if concat_im is None:
                concat_im = C_im.copy()
                if concat_im.dtype == np.uint8:
                    concat_im = concat_im.astype(np.float) * 1.0 / 255.0
            else:
                if C_im is not None:
                    concat_im = self._concat_with_spacer(concat_im=concat_im, new_im=C_im)
            if W_im is not None:
                concat_im = self._concat_with_spacer(concat_im=concat_im, new_im=W_im)

        cv2.imshow('all_tables', concat_im)

    def visualize(self, rays, topdown_info, plots_save_folder, current_goal_position_angle, robot_brain):
        self._display_fps()

        if self.image_display_secs is not None:

            display_now = (time.time() - self.last_image_display_time > self.image_display_secs)

            if display_now:
                # print('displaying now:', time.time() - self.last_image_display_time, self.image_display_secs)
                self._display_graphic_map(rays, topdown_info, current_goal_position_angle)

                if self.show_table_ims:
                    table_ims = robot_brain.get_table_ims()

                    IO_im_list, C_im_list, W_im_list = table_ims
                    self._display_table_ims_tiled(IO_im_list=IO_im_list,
                                                  C_im_list=C_im_list,
                                                  W_im_list=W_im_list)

                self.last_image_display_time = time.time()

                last_key = cv2.waitKey(self.waitKey_time)
                self._toggle_with_key_press(last_key=last_key)

        # if self.plot_brain_error_frames is not None and self.frames % self.plot_brain_error_frames == 0:
        #     self._plot_brain_errors(robot_brain, plots_save_folder)

        self.frames += 1
        if self.auto_switch_to_slow_disp_time is not None:
            if self.frames >= self.auto_switch_to_slow_disp_time:
                self.toggle_viewer_slow = True
                self.image_display_secs = self.image_display_secs_slow
                self.waitKey_time = self.waitKey_time_slow
