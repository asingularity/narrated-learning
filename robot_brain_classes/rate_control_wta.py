import time
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt
from brain_components_classes.states_history import StatesLimitedHistory
#from cython_eff import compute_eff
from offline_analyses.try_sparsify_3 import get_sparse_features, make_im

from utils.one_time_messages import OneTimeMessages

#from cuda_dist_query import CudaTable

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log

np.set_printoptions(suppress=True, precision=2)

random.seed(0)
np.random.seed(0)


def _compute_match(rfs, input_arr, normalize=True):

    tmp_1 = np.multiply(rfs, input_arr)
    tmp_2_cpu = np.sum(tmp_1, axis=1)

    if normalize:
        tmp_3_cpu = np.sum(rfs, axis=1)
        eff_frame = np.divide(tmp_2_cpu, tmp_3_cpu)
    else:
        eff_frame = tmp_2_cpu

    return eff_frame


def _compute_error(rfs, input_arr, single_rf=False):

    if single_rf:
        err_frame = np.sqrt(np.sum(np.square(rfs - input_arr)))
    else:
        err_frame = np.sqrt(np.sum(np.square(rfs - input_arr), axis=1))

    return err_frame



def _compute_error2(rfs, input_arr, single_rf=False):

    if single_rf:
        err_frame = np.sum(np.abs(rfs - input_arr))
    else:
        err_frame = np.sum(np.abs(rfs - input_arr), axis=1)

    return err_frame


class RateControlWTABRain(object):
    def __init__(self, params):
        self.input_im_dim = params['input_im_dim']
        self.num_rfs = 1600
        self.input_concat_timesteps = 1
        self.lr = 0.01 * 10
        self.rate_lr = 0.0001 * 10  # for threshold
        self.forgetful_kmeans = True
        self.apply_rate_control = True

        self.max_time = 10000000

        # for plotting:
        self.error_mean_time = 50000

        self.do_raster_plots_every_k_im = 4
        self.ims_since_raster = 0

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

        self.weights = np.zeros((self.num_rfs, self.input_state_dim)) # + 1e-9
        #self.weights = np.random.random((self.num_rfs, self.input_state_dim)) * 1e-12

        self.rf_counts = np.zeros(self.num_rfs)

        self.last_layer_input = None

        self.mean_error = np.zeros(self.max_time)
        self.error = np.zeros(self.max_time)

        self.t = 0
        self.num_zero_inputs = 0

        self.last_input_im = None
        self.plots_folder = "."
        self.input_history = StatesLimitedHistory(params={'max_delay': self.input_concat_timesteps,
                                                          'states_dim_list': [self.input_state_dim],
                                                          'store_extra_data': False})

        self._init_plotting()

        # rate control stuff
        self._init_rate_control()

    def _init_rate_control(self):
        # target number of time steps between events
        self.target_isi = self.num_rfs
        self.last_win_time = np.zeros(self.num_rfs) - 1
        self.error_thresholds = 100 * np.ones(self.num_rfs)

    def _init_plotting(self):

        self.fig_test_error = plt.figure(figsize=(40, 20))
        self.ax_test_error = self.fig_test_error.add_subplot(1, 1, 1)
        self.ax_test_error.cla()
        self.ax_test_error.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_test_error.get_yaxis().get_major_formatter().set_scientific(False)

        self.fig_bar = plt.figure(figsize=(40, 20))
        self.ax_bar = self.fig_bar.add_subplot(1, 1, 1)
        self.ax_bar.cla()
        self.ax_bar.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_bar.get_yaxis().get_major_formatter().set_scientific(False)

    def set_plots_folder(self, folder):
        self.plots_folder = folder
        print()
        print('setting plots folder: ', self.plots_folder)
        print()

    def process_input(self, input_im):
        if self.t >= self.max_time:
            return

        input_state = self._get_input_state(input_im=input_im)

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            self.num_zero_inputs += 1
            return

        err_frame = _compute_error(rfs=self.weights, input_arr=input_state)

        best_rf_before = np.argmin(err_frame)

        if self.apply_rate_control:
            # invalidate some based on threshold
            err_frame[np.nonzero(np.greater(err_frame, self.error_thresholds))] = np.inf

        best_rf = np.argmin(err_frame)
        assert not np.isnan(best_rf), str(err_frame)

        self.error[self.t] = _compute_error(rfs=self.weights[best_rf_before, :], input_arr=input_state, single_rf=True)
        self.mean_error[self.t] = np.mean(self.error[max(0, self.t - self.error_mean_time):self.t])

        if self.forgetful_kmeans:
            lr = self.lr
            self.weights[best_rf, :] = lr * input_state + (1.0 - lr) * self.weights[best_rf, :]
        else:
            self.rf_counts[best_rf] += 1
            self.weights[best_rf, :] = self.weights[best_rf, :] + (1.0 / self.rf_counts[best_rf]) * (
                    input_state - self.weights[best_rf, :])

        # rate control
        if self.apply_rate_control:
            last_isi = self.t - self.last_win_time
            lr_apply = self.rate_lr * (last_isi - self.target_isi)

            self.error_thresholds = self.error_thresholds + lr_apply
            #self.error_thresholds = np.multiply(self.error_thresholds, 1.0 + lr_apply)

            self.error_thresholds[self.error_thresholds < 0] = 0
            self.error_thresholds[self.error_thresholds > 200] = 200

        self.rf_counts[best_rf] += 1
        self.last_win_time[best_rf] = self.t
        self.t += 1

    def get_table_ims(self):

        print()
        print('prop zero inputs: ', self.num_zero_inputs / self.t)
        print()

        if self.do_raster_plots_every_k_im is not None:
            self.ims_since_raster += 1
            if self.ims_since_raster > self.do_raster_plots_every_k_im:
                self.do_plots()
                self.ims_since_raster = 0

        ims_list = []
        ims_names_list = []

        rfs_im, rf_ims_dict = make_im(self.weights, num_bins_per_pixel=1,
                                                    input_im_dim=self.input_im_dim,
                                                    im_final_dim=int(self.num_rfs * 3000 / 1600),  # /800 for two-im per rf display
                                                    mod_for_disp=20,
                                                    normalize_weights=True)

        ims_list.append(rfs_im)
        ims_names_list.append('rfs_im')

        return ims_list, ims_names_list

        # plot error over time (including over iterations within batches)
        # show RFs

    def do_plots(self):
        self.ax_test_error.cla()
        self.ax_test_error.plot(self.mean_error[0:self.t], color='k', marker='.')
        self.fig_test_error.savefig(self.plots_folder + "/error.png", dpi=100)

        self.ax_bar.cla()
        self.ax_bar.bar(np.arange(self.num_rfs), self.rf_counts)
        self.fig_bar.savefig(self.plots_folder + '/rf_counts.png', dpi=100)
        self.rf_counts[:] = 0.0

        self.ax_bar.cla()
        self.ax_bar.bar(np.arange(self.num_rfs), self.error_thresholds)
        self.fig_bar.savefig(self.plots_folder + '/rf_thresholds.png', dpi=100)


    def _get_input_state(self, input_im):

        # differential stuff, etc

        input_pixels_1 = input_im
        input_pixels_flat = input_pixels_1.flatten()
        input_arr_1 = input_pixels_flat[np.newaxis, :]

        input_exp_1 = input_arr_1
        self.last_input_im = input_pixels_1.copy()

        layer_input = input_exp_1
        input_state = layer_input.flatten().astype(np.float32)

        # change to diff image
        change_to_diff_image = True
        if change_to_diff_image:
            if self.last_layer_input is not None:
                input_state_diff = input_state - self.last_layer_input.flatten().astype(np.float32)

                input_state_p = input_state_diff.copy()
                input_state_n = -input_state_diff.copy()

                input_state_p[input_state_p < 0] = 0
                input_state_n[input_state_n < 0] = 0

                # may want to comment this
                # input_state_p[input_state_p > 0] = 1  # input_state[input_state_p > 0]
                # input_state_n[input_state_n > 0] = 1  # input_state[input_state_n > 0]

                input_state = np.concatenate((input_state_p, input_state_n))
            else:
                self.last_layer_input = layer_input.copy()

                return None

        self.input_history.store_new_states(newest_states_list=[input_state])
        input_states_seq = self.input_history.get_state_sequence(state_index=0,
                                                                 delay_short=0,
                                                                 delay_long=self.input_concat_timesteps - 1,
                                                                 oldest_first=False)

        sum_input_states = np.sum(input_states_seq, axis=0).astype(np.float32)

        self.last_layer_input = layer_input.copy()

        return sum_input_states


