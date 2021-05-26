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


class RateControlWTABRain(object):
    def __init__(self, params):
        self.input_im_dim = params['input_im_dim']

        # new params
        self.num_rfs = params['num_rfs']  # 400
        self.lr = params['lr']  # 1.0 / 1000
        self.rate_lr = params['rate_lr']  # 1.0 / 1000
        self.max_time = params['max_time']  # 5000000
        self.apply_rate_control = params['apply_rate_control']  # True
        self.do_raster_plots_every_k_im = params['do_raster_plots_every_k_im']  # 4, or None

        self.forgetful_kmeans = True

        # for plotting:
        self.error_mean_time = 50000

        self.ims_since_raster = 0

        self.input_concat_timesteps = 1  # unused

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

        #self.weights = np.zeros((self.num_rfs, self.input_state_dim)) # + 1e-1
        self.weights = np.random.random((self.num_rfs, self.input_state_dim)) * 1e-12

        self.rf_counts = np.zeros(self.num_rfs)
        self.num_error_per_rf = np.zeros(self.num_rfs)

        self.last_layer_input = None

        self.mean_error = np.zeros(self.max_time)
        self.error = np.zeros(self.max_time)

        self.mean_num_bad_rfs = np.zeros(self.max_time)
        self.num_bad_rfs = np.zeros(self.max_time)

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

        # rasters
        self._init_rasters()

        self.sum_input_per_rf = np.zeros(self.num_rfs)
        self.num_input_per_rf = np.zeros(self.num_rfs)

        self.sum_errors_per_rf = {}
        self.num_errors_per_rf = {}
        self.mean_errors = {}

        self.errors_over_time = {}
        self.mean_errors_over_time = {}
        # not well bounded: 'RF-norm-L2', 'RF-norm-L1'
        for error_type in ['L2', 'L1', 'input-norm-L1', 'input-norm-L2', 'RF-norm-dot']:
            self.sum_errors_per_rf[error_type] = np.zeros(self.num_rfs)
            self.num_errors_per_rf[error_type] = np.zeros(self.num_rfs)

            self.mean_errors[error_type] = np.zeros(self.max_time)  # per rf

            self.errors_over_time[error_type] = np.zeros(self.max_time)
            self.mean_errors_over_time[error_type] = np.zeros(self.max_time)

    def _init_rasters(self):
        self.raster_steps = 200
        self.raster_t = 0  # circular; draw vertical line on plot here
        self.input_raster_history = np.zeros((self.input_state_dim, self.raster_steps), np.uint8)
        self.rfs_raster_history = np.zeros((self.num_rfs, self.raster_steps), np.uint8)

    def _init_rate_control(self):
        # target number of time steps between events
        self.target_isi = self.num_rfs
        self.error_thresholds = 1 * np.ones(self.num_rfs)
        self.error_multipliers = 1 * np.ones(self.num_rfs)

        self.last_win_time = np.zeros(self.num_rfs) - 1

        self.mean_isi = self.target_isi * np.ones(self.num_rfs)

        #self.target_fr = 1.0 / self.target_isi
        #self.last_win_time = -np.inf* np.ones(self.num_rfs)
        #self.last_isi = np.inf * np.ones(self.num_rfs)

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

    def _get_error_measures(self, rfs, input_arr):

        assert rfs.shape[1] == input_arr.shape[0]

        L2 = np.sqrt(np.sum(np.square(rfs - input_arr), axis=1))
        L1 = np.sum(np.abs(rfs - input_arr), axis=1)
        input_norm_L1 = np.sum(np.abs(rfs - input_arr), axis=1) / np.sum(input_arr)
        input_norm_L2 = np.sqrt(np.sum(np.square(rfs - input_arr), axis=1)) / np.sum(np.square(input_arr))
        RF_norm_L1 = np.divide(np.sum(np.abs(rfs - input_arr), axis=1), np.sum(np.abs(rfs), axis=1))
        RF_norm_L2 = np.divide(L2, np.sum(np.abs(rfs), axis=1))

        # why is this one negative? because it is actually a positive value: the larger, the better
        RF_norm_dot = -np.divide(np.sum(np.multiply(rfs, input_arr), axis=1), np.sum(rfs, axis=1))

        errors = {
            'L2': L2,
            'L1': L1,
            'input-norm-L1': input_norm_L1,
            'input-norm-L2': input_norm_L2,
            'RF-norm-dot': RF_norm_dot
            # 'RF-norm-L1': RF_norm_L1,  # not well bounded
            #'RF-norm-L2': RF_norm_L2  # not well bounded
        }

        return errors

    def process_input(self, input_events_p, input_events_n):
        if self.t >= self.max_time:
            return

        # input_state = self._get_input_state(input_im=input_im)
        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            self.num_zero_inputs += 1
            return

        # *** selection ***

        errors = self._get_error_measures(rfs=self.weights, input_arr=input_state)
        best_rf = np.argmin(errors['RF-norm-dot'])

        # *** errors for plotting ***

        for error_type in errors:
            error_value = errors[error_type][best_rf]
            self.sum_errors_per_rf[error_type][best_rf] += error_value
            self.num_errors_per_rf[error_type][best_rf] += 1

            tmp = np.divide(self.sum_errors_per_rf[error_type], self.num_errors_per_rf[error_type])
            tmp = tmp[np.nonzero(np.logical_not(np.isnan(tmp)))]
            self.mean_errors[error_type][self.t] = np.mean(tmp)

            self.errors_over_time[error_type][self.t] = error_value
            self.mean_errors_over_time[error_type][self.t] = np.mean(self.errors_over_time[error_type][max(0, self.t - self.error_mean_time):self.t])

        sum_input = np.sum(input_state)
        self.sum_input_per_rf[best_rf] = self.sum_input_per_rf[best_rf] + sum_input
        self.num_input_per_rf[best_rf] += 1

        self.rfs_raster_history[:, self.raster_t] = 0
        self.rfs_raster_history[best_rf, self.raster_t] = 1
        self.rf_counts[best_rf] += 1

        # *** learning ***

        if self.forgetful_kmeans:
            lr = self.lr
            self.weights[best_rf, :] = lr * input_state + (1.0 - lr) * self.weights[best_rf, :]
        else:
            self.weights[best_rf, :] = self.weights[best_rf, :] + (1.0 / self.rf_counts[best_rf]) * (
                    input_state - self.weights[best_rf, :])

        # *** rate control **

        if self.apply_rate_control:
            last_isi = self.t - self.last_win_time[best_rf]

            # +: isi too large: firing rate too low
            # -: isi too small: firing rate too high
            lr_apply = self.rate_lr * (last_isi - self.target_isi)/self.target_isi
            w_rf = self.weights[best_rf, :]

            w_rf = w_rf * (1.0 - lr_apply)
            w_rf[w_rf>1] = 1
            self.weights[best_rf, :] = w_rf[:]

        # *** time step ***

        self.last_win_time[best_rf] = self.t
        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0

    def get_table_ims(self):

        if self.do_raster_plots_every_k_im is not None:
            self.ims_since_raster += 1
            if self.ims_since_raster > self.do_raster_plots_every_k_im:
                self.do_plots()
                self.ims_since_raster = 0

        ims_list = []
        ims_names_list = []

        rfs_im, rf_ims_dict = make_im(self.weights, num_bins_per_pixel=1,
                                                    input_im_dim=self.input_im_dim,
                                                    im_final_dim=int(200 * 3000 / 400),  # /800 for two-im per rf display
                                                    mod_for_disp=int(sqrt(self.num_rfs)),
                                                    normalize_weights=True)

        ims_list.append(rfs_im)
        ims_names_list.append('rfs_im')

        return ims_list, ims_names_list

        # plot error over time (including over iterations within batches)
        # show RFs

    def do_plots(self):

        self.ax_bar.cla()
        self.ax_bar.bar(np.arange(self.num_rfs), self.rf_counts)
        self.fig_bar.savefig(self.plots_folder + '/rf_counts.png', dpi=100)

        self.rf_counts[:] = 0.0

        for error_type in self.sum_errors_per_rf.keys():
            mean_error = self.mean_errors[error_type][0:self.t]

            # mean of per-rf errors; weighing all RFs equally, ignoring relative frequency of RFs
            self.ax_test_error.cla()
            self.ax_test_error.plot(mean_error[0:self.t], color='k', marker='.')
            self.fig_test_error.savefig(self.plots_folder + "/rf_centered_mean_error_" + error_type + ".png", dpi=100)

            # mean error per rf
            self.ax_bar.cla()
            self.ax_bar.bar(np.arange(self.num_rfs), np.divide(self.sum_errors_per_rf[error_type], self.num_errors_per_rf[error_type]))
            self.fig_bar.savefig(self.plots_folder + '/per_rf_mean_error_' + error_type + '.png', dpi=100)

            # this is time averaged error, per time point; weighing all time points equally,
            # taking relative frequency of RFs into account
            self.ax_test_error.cla()
            self.ax_test_error.plot(self.mean_errors_over_time[error_type][0:self.t], color='k', marker='.')
            self.fig_test_error.savefig(self.plots_folder + "/time_averaged_mean_error_" + error_type + ".png", dpi=100)

        self.ax_bar.cla()
        self.ax_bar.bar(np.arange(self.num_rfs), np.divide(self.sum_input_per_rf, self.num_input_per_rf))
        self.fig_bar.savefig(self.plots_folder + '/input_sum_per_rf.png', dpi=100)

        self.ax_bar.cla()
        self.ax_bar.bar(np.arange(self.num_rfs), np.sum(np.abs(self.weights), axis=1))
        self.fig_bar.savefig(self.plots_folder + '/rf_sums.png', dpi=100)

        self.ax_bar.cla()
        num_rf = self.rfs_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.rfs_raster_history, np.arange(num_rf)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.ax_bar.axvline(x=self.raster_t, color='g')
        self.fig_bar.savefig(self.plots_folder + "/raster_rfs.png", dpi=100)

        self.ax_bar.cla()
        num_rf = self.input_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.input_raster_history, np.arange(self.input_state_dim)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.ax_bar.axvline(x=self.raster_t, color='g')
        self.fig_bar.savefig(self.plots_folder + "/raster_inputs.png", dpi=100)


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


