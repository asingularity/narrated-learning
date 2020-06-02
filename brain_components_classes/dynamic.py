import time
from math import sqrt
import cv2
import numpy as np
import random
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DynamicPredictor(object):
    def __init__(self, params):
        assert params['color_enabled'] is False

        self.input_dim = params['input_dim']
        self.max_history_length = params['max_history_length']

        self.ims_scale_pixels = float(params['ims_scale_pixels'])

        print()
        print('Initializing DynamicPredictor...')
        print()
        print('    ' + 'input_dim: ', self.input_dim)
        print('    ' + 'max_history_length: ', self.max_history_length)
        print('    ' + 'ims_scale_pixels: ', self.ims_scale_pixels)
        print()

        # *** binning ***
        self.enable_binning = True  # use binary input bins, instead of grayscale values, because weighing them makes more sense
        self.num_bins_per_pixel = 10
        if self.enable_binning:
            self.input_dim = params['input_dim'] * self.num_bins_per_pixel

        # *** network ***
        self.N = 20  # number of state variables
        self.X = 2.0 * (np.random.random(self.N) - 0.5)
        # TODO this should be many delayed values, and history below should be a rolling history!
        self.X_hist_len = 100
        self.X_history = np.random.random((self.N, self.X_hist_len))

        self.W_r = np.random.random((self.N, self.N * self.X_hist_len)) - 0.5  # recurrent weights # TODO adjust via delta rule!
        self.W_in = np.random.random((self.N, self.input_dim)) - 0.5  # input weights

        #evalues, evectors = np.linalg.eig(self.W_r)
        #print()
        #print('spectral radius: ', np.amax(np.abs(evalues)))

        # *** etc ***
        self.t = 0
        self.printed_init_step = False
        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

    def step(self, raycast_image, input_state, input_x_y_theta, goal_context_state_learning, goal_context_state_task, last_motor_command):
        '''

        :param raycast_image: (32x32x3)
        :param input_state:
        :param input_x_y_theta:
        :param goal_context_state_learning:
        :param goal_context_state_task:
        :param last_motor_command:
        :return:
        '''

        assert input_state is None
        assert input_x_y_theta is None
        assert goal_context_state_learning is None
        assert goal_context_state_task is None
        assert last_motor_command is None

        if not self.printed_init_step:
            print()
            print('DynamicPredictor.step: ', self.t)
            print()
            print('    raycast_image.shape: ', raycast_image.shape)
            print('    raycast_image.dtype: ', raycast_image.dtype)
            print('    raycast_image min, max: ', np.amin(raycast_image), np.amax(raycast_image))

        if not self.printed_init_step:
            print()

        if self.t > 3:
            self.printed_init_step = True

        self._step_network()

        self.t += 1
        return None

    def _step_network(self):
        r = 1.0
        a = np.dot(self.W_r, self.X_history.flatten())
        norm_a = sqrt(np.sum(np.square(a)))

        self.X = r * a / norm_a

        # initial state history
        if self.t == self.X_hist_len * 10:
            print()
            print('Making plot of state vars...')
            self.ax.cla()
            #self.ax.plot(self.X, 'b.-')
            self.ax.plot(np.transpose(self.X_history))
            self.fig.savefig("state_vars.png", dpi=100)
            print('Done.')
            print()

        np.roll(self.X_history, 1, axis=1)
        #print(np.amin(self.X), np.amax(self.X))
        self.X_history[:, 0] = self.X[:]

        # norm_X = sqrt(np.sum(np.square(self.X)))  # confirmed that this is 'r'

    def get_table_ims(self):

        ims_list = []
        ims_names_list = []

        return ims_list, ims_names_list
