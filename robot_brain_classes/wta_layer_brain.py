import time
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt

from utils.one_time_messages import OneTimeMessages

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log



class WTALayerBrain(object):
    '''
    kWTA, where k is adaptive and based on remainder
    activations and learning are based on remainder
    '''

    def __init__(self, params):

        # ****************** parameters ******************

        # TODO need to modify sensor input classes to not have to take subsets of larger video
        #   here, but elsewhere:
        # self.in_r = 50
        # self.in_c = 64
        # assert self.in_r + self.rf_dim < self.input_im_dim, str((self.in_r + self.rf_dim, self.input_im_dim))
        # assert self.in_c + self.rf_dim < self.input_im_dim, str((self.in_c + self.rf_dim, self.input_im_dim))
        # ...
        # input_pixels_1 = input_im[self.in_r:self.in_r + self.rf_dim, self.in_c:self.in_c + self.rf_dim]

        self.im_dim = params['image_dim_display']  # TODO 1500

        self.input_im_dim = params['image_dim_NxN_pixels']  # TODO 16
        self.rf_dim = self.input_im_dim  # i.e. 16

        self.lr_base = params['learning_rate']  # TODO 0.0001 * 0.25
        self.bins_per_pixel = params['bins_per_pixel']  # TODO 6

        # per layer
        self.num_rf = params['num_rf']  # TODO 20

        self.layer_start_times = params['layer_start_times']  # TODO np.array([0, 1, 2, 3, 4, 5]) * 300000
        max_time = params['max_time']  # TODO 10000000

        self.learn_off_time = params['learning_off_time']  # TODO 1800000

        self.plot_interval = params['plot_interval_seconds']  # TODO 30

        # ****************** initialization ******************

        self.otm = OneTimeMessages()

        self.t = 0

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

        self.num_layers = self.layer_start_times.shape[0]

        self.input_feature_len = self.rf_dim * self.rf_dim * self.bins_per_pixel

        self.probs = []

        # self.lr = 0.001

        self.lr = self.lr_base * np.ones((self.num_layers, self.num_rf))

        for k in range(self.num_layers):
            self.probs.append(np.zeros((self.num_rf, self.input_feature_len)))

        self.rf_winners = np.zeros((max_time, self.num_rf), np.int)

        self.mean_fr = 10000
        self.rf_counts = np.zeros(max_time)
        self.mean_rf_counts = np.zeros(max_time)
        self.rec_error = np.zeros(max_time)
        self.mean_rec_error = np.zeros(max_time)

        self.im_error = np.zeros(max_time)
        self.mean_im_error = np.zeros(max_time)

        self.last_input_im = None
        self.last_reconstruction_im = None
        self.last_remainder_im = None

        # plot error
        self.last_plot_time = time.time()

        self.last_info = None
        self.last_eff_frames = np.zeros(self.num_rf)

        self.learning_enabled = None

    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        input_pixels_1 = input_im
        input_arr_1 = input_pixels_1.flatten()[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

        self.last_input_im = input_pixels_1.copy()

        error = np.sum(input_exp_1)

        remainder = input_exp_1.copy()
        reconstruction = np.zeros((1, self.input_feature_len))
        rfs_active = np.zeros(self.num_rf)

        self.last_eff_frames[:] = np.nan

        for layer_n in range(self.num_layers):
            if self.t >= self.layer_start_times[layer_n]:

                eff_frame = np.sum(np.abs(self.probs[layer_n] - remainder), axis=1)
                win_rf_index = np.argmin(eff_frame)

                self.rf_winners[self.t, layer_n] = int(win_rf_index)

                # TODO fix for layers, to keep track / plot this:
                self.last_eff_frames[win_rf_index] = eff_frame[win_rf_index]

                lr = self.lr[layer_n, win_rf_index]

                if self.t < self.learn_off_time:
                    self.probs[layer_n][win_rf_index, :] = (1.0 - lr) * self.probs[layer_n][win_rf_index, :] + lr * remainder[0, :]
                    self.learning_enabled = True
                else:
                    self.learning_enabled = False

                self.lr[layer_n, win_rf_index] = self.lr_base

                remainder = remainder - self.probs[layer_n][win_rf_index, :]

                reconstruction = reconstruction + self.probs[layer_n][win_rf_index, :]

        self.last_remainder_im = remainder.copy()

        # TODO look into effect of this later, or alternatives:
        reconstruction[reconstruction > 1] = 1
        reconstruction[reconstruction < 0] = 0

        self.last_reconstruction_im = reconstruction.copy()

        rec_error = np.sum(np.abs(reconstruction - input_exp_1))
        sum_remainder = np.sum(np.abs(remainder))

        self.rec_error[self.t] = rec_error
        self.mean_rec_error[self.t] = np.mean(self.rec_error[max(0, self.t - self.mean_fr):self.t])

        # TODO rename properly
        self.rf_counts[self.t] = sum_remainder  #np.count_nonzero(rfs_active)
        self.mean_rf_counts[self.t] = np.mean(self.rf_counts[max(0, self.t - self.mean_fr):self.t])

        self.t += 1

    def get_table_ims(self):

        ims_list = []
        ims_names_list = []

        if time.time() > self.last_plot_time + self.plot_interval:
            print()
            print('making plot; learning enabled:', self.learning_enabled)
            print()

            try:
                self.ax.cla()
                self.ax.plot(self.rec_error[0:self.t], color='r')
                self.ax.plot(self.mean_rec_error[0:self.t], color='b')
                for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]:
                    self.ax.axhline(y=k, color='g')
                self.fig.savefig("rec_error.png", dpi=100)

                self.ax.cla()
                self.ax.plot(self.rf_counts[0:self.t], color='r')
                self.ax.plot(self.mean_rf_counts[0:self.t], color='b')
                for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]:
                    self.ax.axhline(y=k, color='g')
                self.fig.savefig("sum_remainder.png", dpi=100)

                self.ax.cla()

                self.ax.plot(self.rf_winners[max(0, self.t - 1000):self.t, :], color='b', marker='o', linestyle='')
                self.ax.set_ylim([-1, self.num_rf])
                self.fig.savefig("rf_winners.png", dpi=100)
            except:
                print()
                print('!!!!!!!! Error !!!!!!!!')
                print()
                print('Plot Failed!')
                print()
                print('!!!!!!!! Error !!!!!!!!')
                print()

            self.last_plot_time = time.time()

        tmp_all_im = None
        for layer_n in range(self.num_layers):
            tmp_im = None
            arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.probs[layer_n], num_bins_per_pixel=self.bins_per_pixel)

            for r_tmp in range(weights.shape[0]):

                im0 = arr[r_tmp, :].reshape((self.rf_dim, self.rf_dim))
                im1 = weights[r_tmp, :].reshape((self.rf_dim, self.rf_dim))

                tmp2 = np.hstack((im0, im1))

                if tmp_im is None:
                    tmp_im = tmp2.copy()
                else:
                    tmp3 = np.zeros((2, tmp_im.shape[1]))
                    tmp_im = np.vstack((tmp_im, tmp3, tmp2))

            if tmp_all_im is None:
                tmp_all_im = tmp_im.copy()
            else:
                tmp4 = np.zeros((tmp_im.shape[0], 2))
                tmp_all_im = np.hstack((tmp_all_im, tmp4, tmp_im))

        max_dim = max(tmp_all_im.shape[0], tmp_all_im.shape[1])
        imscale = self.im_dim / max_dim  # 0.2: full table, 2.0
        tmp_im = cv2.resize(tmp_all_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list.append(tmp_im)
        ims_names_list.append('probs_v, probs_w')

        rec_im, rec_w = self._collapse_binned_columns_to_pixels(arr_exp=self.last_reconstruction_im, num_bins_per_pixel=self.bins_per_pixel)
        im0 = rec_im[0, :].reshape((self.rf_dim, self.rf_dim))
        im1 = rec_w[0, :].reshape((self.rf_dim, self.rf_dim))

        rem_im, rem_w = self._collapse_binned_columns_to_pixels(arr_exp=self.last_remainder_im, num_bins_per_pixel=self.bins_per_pixel)
        imr0 = rem_im[0, :].reshape((self.rf_dim, self.rf_dim))
        imr1 = rem_w[0, :].reshape((self.rf_dim, self.rf_dim))

        ims_list.append(np.hstack((self.last_input_im,
                                   np.zeros((self.rf_dim, 2)), im0, np.zeros((self.rf_dim, 2)), im1,
                                   np.zeros((self.rf_dim, 2)), imr0, np.zeros((self.rf_dim, 2)), imr1,)))
        ims_names_list.append('input, reconstruct, remainder')

        return ims_list, ims_names_list

    def _bin_pixels_expand_columns(self, arr, num_bins_per_pixel):
        '''

        keep number of rows, expand in number of columns
        construct binary image: (0.0 or 1.0) per bin

        https://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
            use: digitize, or histogram

        https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.digitize.html
            digitize: returns index of bin to which each pixel belongs
            expanded array will have a set of bins per pixel, in the expanded columns

        :param arr:
        :return:
        '''
        # print('***')
        # print('arr', arr.shape)

        bins = np.linspace(0, 1 + 1e-9, num_bins_per_pixel + 1)  # TODO to fix binning add 1e-9 to 1 in second argument!
        # print('bins', bins.shape, bins)
        bin_indices = np.digitize(arr, bins) - 1  # same shape as arr; which bin, per pixel
        # print('bin_indices', bin_indices.shape, bin_indices)
        arr_exp = np.zeros((arr.shape[0], arr.shape[1] * num_bins_per_pixel), np.float32)
        # print('arr_exp', arr_exp.shape)

        # term1 = np.tile(self.num_bins_per_pixel * np.arange(arr.shape[1]), 2)  # can't remember why this is np.tile(..., 2)
        term1 = num_bins_per_pixel * np.arange(arr.shape[1])  # only works if arr rows is 1?
        term2 = bin_indices.flatten()
        # print('term1', term1.shape, term1)
        # print('term2', term2.shape, term2)
        c = term1 + term2
        r = np.repeat(np.arange(arr.shape[0]), arr.shape[1])

        arr_exp[r, c] = 1.0

        return arr_exp

    def _collapse_binned_columns_to_pixels(self, arr_exp, num_bins_per_pixel):
        '''

        non-trivial: arr_exp may no longer be binary; need to figure out how to collapse multiple values of different weight to one pixel:
            weighted average? take max value?

        :param arr_exp:
        :return:
        '''

        num_pixels = int((arr_exp.shape[1] / num_bins_per_pixel))
        num_rows = arr_exp.shape[0]

        # reshape so we can take max along one dim
        #   ie each row in this matrix is a pixel, temporarily
        # then reshape back

        tmp = arr_exp.reshape((num_rows * num_pixels, num_bins_per_pixel))

        bins = np.linspace(0, 1 + 1e-9, num_bins_per_pixel + 1)

        # max val for display:
        max_index = np.argmax(tmp, axis=1)
        max_vals = bins[max_index]

        weights = tmp[np.arange(tmp.shape[0]), max_index]

        # print(max_vals.shape, weights.shape)

        # weighted mean val for display:
        # tmp1 = np.multiply(tmp, bins[np.newaxis, 0:num_bins_per_pixel])
        # tmp2 = np.sum(tmp1, axis=1)  # 65536
        # tmp3 = np.divide(tmp2, np.sum(tmp, axis=1))
        # max_vals = tmp3

        weights = weights.reshape(num_rows, num_pixels)
        arr = max_vals.reshape(num_rows, num_pixels)

        return arr, weights
