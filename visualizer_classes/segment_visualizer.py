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


class SegmentVisualizer(object):
    def __init__(self, params):
        '''

        'fps_display_interval': 3,
        'image_display_secs_fast': 5,
        'waitKey_time_fast': 1,  # 1, 100, 5000
        'image_display_secs_slow': 0,  # 0: every frame
        'waitKey_time_slow': 1,  # 1, 100, 5000  #
        'scale_camera_factor': 20,
        'auto_switch_to_slow_disp_time': None,
        'init_fast': True  # start with "fast" display

        :param params:
        '''

        self.fps_frames = 0
        self.frames = 0
        self.last_FPS_time = time.time()
        self.fps_display_interval = params['fps_display_interval']
        self.scale_camera_factor = params['scale_camera_factor']
        self.color_enabled = params['color_enabled']

        self.waitKey_time_slow = params['waitKey_time_slow']
        self.image_display_secs_slow = params['image_display_secs_slow']
        self.waitKey_time_fast = params['waitKey_time_fast']
        self.image_display_secs_fast = params['image_display_secs_fast']
        self.auto_switch_to_slow_disp_time = params['auto_switch_to_slow_disp_time']

        self.show_table_ims = True

        if params['init_fast']:
            self.toggle_viewer_slow = False
            self.image_display_secs = self.image_display_secs_fast
            self.waitKey_time = self.waitKey_time_fast
        else:
            self.toggle_viewer_slow = True
            self.image_display_secs = self.image_display_secs_slow
            self.waitKey_time = self.waitKey_time_slow

        self.last_image_display_time = time.time()


    def _display_fps(self):
        if time.time() - self.last_FPS_time > self.fps_display_interval:
            fps = self.fps_frames * 1.0 / (time.time() - self.last_FPS_time)
            logging.info('FPS: ' + str(fps) + ', frames: ' + str(self.frames))
            self.fps_frames = 0
            self.last_FPS_time = time.time()
        self.fps_frames += 1

    def _display_raycast_image(self, raycast_image):
        resized_im = cv2.resize(src=raycast_image, dsize=(0, 0), fx=self.scale_camera_factor,
                                fy=self.scale_camera_factor, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('raycast', resized_im)

        # without this, was not always displaying most up to date image
        for k in range(100):
            cv2.waitKey(1)


    def _toggle_with_key_press(self, last_key):
        k = last_key

        ENTER = 13

        if k == -1:
            pass
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

    def visualize(self, input_im, segment_brain):
        '''

        :param input_im:
        :param segment_brain:
        :return:
        '''

        self._display_fps()

        if self.image_display_secs is not None:

            display_now = (time.time() - self.last_image_display_time > self.image_display_secs)

            if display_now:

                table_ims, table_ims_names = segment_brain.get_table_ims()

                for k in range(len(table_ims)):
                    if table_ims[k] is not None:
                        cv2.imshow(table_ims_names[k], table_ims[k])


                if input_im is not None:
                    self._display_raycast_image(raycast_image=input_im)

                self.last_image_display_time = time.time()

                last_key = cv2.waitKey(self.waitKey_time)
                self._toggle_with_key_press(last_key=last_key)

        self.frames += 1
        if self.auto_switch_to_slow_disp_time is not None:
            if self.frames >= self.auto_switch_to_slow_disp_time:
                self.toggle_viewer_slow = True
                self.image_display_secs = self.image_display_secs_slow
                self.waitKey_time = self.waitKey_time_slow
