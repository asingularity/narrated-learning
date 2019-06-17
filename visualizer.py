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
        self.image_display_frames_slow = params['image_display_frames_slow']
        self.waitKey_time_fast = params['waitKey_time_fast']
        self.image_display_frames_fast = params['image_display_frames_fast']
        self.auto_switch_to_slow_disp_time = params['auto_switch_to_slow_disp_time']
        self.linear_speed_from_key = 0.0
        self.angular_speed_from_key = 0.0
        self.toggle_viewer_slow = False
        self.image_display_frames = self.image_display_frames_fast
        self.waitKey_time = self.waitKey_time_fast

        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(1, 1, 1)

    def get_linear_angular_speed(self):
        return self.linear_speed_from_key, self.angular_speed_from_key

    def _display_fps(self):
        if time.time() - self.last_FPS_time > self.fps_display_interval:
            fps = self.fps_frames * 1.0 / (time.time() - self.last_FPS_time)
            logging.info('FPS: ' + str(fps) + ', frames: ' + str(self.frames))
            self.fps_frames = 0
            self.last_FPS_time = time.time()
        self.fps_frames += 1

    def _get_topdown_map(self, rays, topdown_info, current_goal_position_angle, plan_position_angle_list):
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

        if plan_position_angle_list is not None:
            if 0:
                input_td_info = plan_position_angle_list[0]
                context_td_info = plan_position_angle_list[1]
                output_td_info = plan_position_angle_list[2]

                br = 1.0
                for td_info in [input_td_info, context_td_info]: #, output_td_info]:
                    r_x = td_info[0]
                    r_y = td_info[1]
                    r_theta = td_info[2]

                    cv2.circle(img=resized_image,
                               center=(int(r_x * self.scale_topdown_factor), int(r_y * self.scale_topdown_factor)),
                               radius=5,
                               color=(1.0 * br, 1.0 * br, 1.0 * br),
                               thickness=3)
                    g2_x = r_x + 0.5 * cos(r_theta)
                    g2_y = r_y + 0.5 * sin(r_theta)
                    cv2.line(resized_image,
                             pt1=(int(r_x * self.scale_topdown_factor), int(r_y * self.scale_topdown_factor)),
                             pt2=(int(g2_x * self.scale_topdown_factor), int(g2_y * self.scale_topdown_factor)),
                             color=(0.5, 0.5, 0),
                             thickness=2)
                    br *= 0.5

            # plan
            if True:
                br = 1.0
                for td_info in plan_position_angle_list:
                    r_x = td_info[0]
                    r_y = td_info[1]
                    r_theta = td_info[2]

                    cv2.circle(img=resized_image,
                               center=(int(r_x * self.scale_topdown_factor), int(r_y * self.scale_topdown_factor)),
                               radius=5,
                               color=(1.0 * br, 1.0 * br, 1.0 * br),
                               thickness=3)
                    g2_x = r_x + 0.5 * cos(r_theta)
                    g2_y = r_y + 0.5 * sin(r_theta)
                    cv2.line(resized_image,
                             pt1=(int(r_x * self.scale_topdown_factor), int(r_y * self.scale_topdown_factor)),
                             pt2=(int(g2_x * self.scale_topdown_factor), int(g2_y * self.scale_topdown_factor)),
                             color=(0.5 * br, 0.5 * br, 0),
                             thickness=2)

                    br *= 0.85

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

    def _display_graphic_map(self, rays, robot_brain, topdown_info, current_goal_position_angle, plan_position_angle_list):
        ray_colors = rays['ray_colors'].reshape((len(rays['ray_colors']) / 3, 3))
        resized_image = self._get_topdown_map(rays, topdown_info, current_goal_position_angle, plan_position_angle_list)

        cv2.imshow('env_map', resized_image)

        camera_image = np.zeros((1, ray_colors.shape[0], 3))
        camera_image[0, :, 0] = ray_colors[:, 0]
        camera_image[0, :, 1] = ray_colors[:, 1]
        camera_image[0, :, 2] = ray_colors[:, 2]
        resized_camera = cv2.resize(src=camera_image, dsize=(0, 0), fx=self.scale_camera_factor, fy=self.scale_camera_factor, interpolation=cv2.INTER_NEAREST)

        cv2.imshow('camera', resized_camera)
        k = cv2.waitKey(self.waitKey_time)

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
                    self.image_display_frames = self.image_display_frames_slow
                    self.waitKey_time = self.waitKey_time_slow
                else:
                    self.image_display_frames = self.image_display_frames_fast
                    self.waitKey_time = self.waitKey_time_fast

            if k == FWD:
                self.linear_speed_from_key = 0.25
            if k == BACK:
                self.linear_speed_from_key = -0.25
            if k == LEFT:
                self.angular_speed_from_key = -0.2
            if k == RIGHT:
                self.angular_speed_from_key = 0.2

    def _plot_brain_errors(self, robot_brain, plots_save_folder):
        error_names_autoenc, error_histories_autoenc, \
        error_names_predictor, error_histories_predictor, \
        error_names_inverse, error_histories_inverse, \
        error_names_no_context_predictor, error_histories_no_context_predictor = robot_brain.get_error_names_histories()
        print( self.frames)

        for k in range(len(error_names_autoenc)):
            error_name = error_names_autoenc[k]
            error_history = error_histories_autoenc[k]
            print( error_name)
            print( error_history.shape)
            self.ax.cla()
            self.ax.set_ylim([0, 0.12])
            self.ax.plot(error_history)
            self.fig.savefig(plots_save_folder + '/' + error_name + '.png', dpi=100)

        if error_names_predictor is not None:
            for k in range(len(error_names_predictor)):
                error_name = error_names_predictor[k]
                error_history = error_histories_predictor[k]
                print( error_name)
                print( error_history.shape)
                self.ax.cla()
                self.ax.plot(error_history, 'b-')

                #error_name_nc = error_names_no_context_predictor[k]
                #error_history_nc = error_histories_no_context_predictor[k, :]
                #self.ax.plot(error_history_nc, 'r-')
                #  + '_' + error_name_nc
                self.fig.savefig(plots_save_folder + '/' + error_name + '.png', dpi=100)

            for k in range(len(error_names_inverse)):
                error_name = error_names_inverse[k]
                error_history = error_histories_inverse[k]
                print( error_name)
                print( error_history.shape)
                self.ax.cla()
                self.ax.plot(error_history, 'b-')
                self.fig.savefig(plots_save_folder + '/' + error_name + '.png', dpi=100)

    def visualize(self, rays, robot_brain, topdown_info, plots_save_folder, current_goal_position_angle, plan_position_angle_list):
        if self.image_display_frames is not None:
            if self.frames % self.image_display_frames == 0:
                self._display_graphic_map(rays, robot_brain, topdown_info, current_goal_position_angle, plan_position_angle_list)

        if self.plot_brain_error_frames is not None and self.frames % self.plot_brain_error_frames == 0:
            self._plot_brain_errors(robot_brain, plots_save_folder)

        self._display_fps()

        self.frames += 1
        if self.auto_switch_to_slow_disp_time is not None:
            if self.frames >= self.auto_switch_to_slow_disp_time:
                self.toggle_viewer_slow = True
                self.image_display_frames = self.image_display_frames_slow
                self.waitKey_time = self.waitKey_time_slow
