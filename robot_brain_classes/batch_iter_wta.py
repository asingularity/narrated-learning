import time
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt
from brain_components_classes.states_history import StatesLimitedHistory
from cython_eff import compute_eff
from offline_analyses.try_sparsify_3 import get_sparse_features, make_im

from utils.one_time_messages import OneTimeMessages

from cuda_dist_query import CudaTable

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log

np.set_printoptions(suppress=True, precision=2)


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
        err_frame = np.sum(np.abs(rfs - input_arr))
    else:
        err_frame = np.sum(np.abs(rfs - input_arr), axis=1)

    return err_frame


class BatchIterWTABrain(object):
    def __init__(self, params):

        self.input_im_dim = params['input_im_dim']
        #self.max_time = 800000  # after this time, don't do anything at all, just keeps images displayed

        self.max_num_batches = 40
        self.batch_length = 20000
        self.iters_per_batch = 4
        self.num_rfs = 400
        self.input_concat_timesteps = 1
        self.lr = 0.01  # 0.1, 0.001 -> only used if forgetful
        self.forgetful_kmeans = True
        self.enforce_max_firing = True

        # for plotting:
        self.error_mean_time_test = 16000
        self.error_mean_time_train = 16000

        self.do_raster_plots_every_k_im = 4
        self.ims_since_raster = 0

        # init

        assert self.batch_length / self.num_rfs == int(self.batch_length / self.num_rfs)

        self.num_firing_per_train_batch = int(self.batch_length / self.num_rfs)
        self.curr_batch_accum_step = 0

        self.curr_batch = 0
        # max self.t
        self.max_time = self.max_num_batches * self.batch_length

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes
        self.inputs_batch = np.zeros((self.batch_length, self.input_state_dim))
        self.weights = np.zeros((self.num_rfs, self.input_state_dim))
        self.rf_counts = np.zeros(self.num_rfs)

        self.last_layer_input = None

        # track error: in testing (i.e. when accumulating new batch, no constraint on winners),
        #              and in training (and over multiple iterations in a batch)

        # this is for testing (when looking at a batch for "first time"):
        self.mean_error = np.zeros(self.max_time)
        self.error = np.zeros(self.max_time)

        # for within batches during training: this is all created, and plotted, internally in the loop in one step
        self.t = 0

        # unknown if using?

        self.last_input_im = None
        self.plots_folder = "."
        self.input_history = StatesLimitedHistory(params={'max_delay': self.input_concat_timesteps,
                                                          'states_dim_list': [self.input_state_dim],
                                                          'store_extra_data': False})

        # self.rfs_raster_history = np.zeros((self.num_rfs, self.max_time), np.uint8)

        # seq kmeans init

        self.last_event_time = np.zeros(self.num_rfs)
        self.num_resets = 0
        self.num_frames_disp_resets = 0

        self._init_plotting()

    def _init_plotting(self):

        self.fig_train_error = plt.figure(figsize=(20, 20))
        self.ax_train_error = self.fig_train_error.add_subplot(1, 1, 1)
        self.ax_train_error.cla()
        self.ax_train_error.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_train_error.get_yaxis().get_major_formatter().set_scientific(False)

        self.fig_test_error = plt.figure(figsize=(20, 20))
        self.ax_test_error = self.fig_test_error.add_subplot(1, 1, 1)
        self.ax_test_error.cla()
        self.ax_test_error.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_test_error.get_yaxis().get_major_formatter().set_scientific(False)


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

        # accumulate batch while testing on this new batch
        self.inputs_batch[self.curr_batch_accum_step, :] = input_state[:]

        eff_frame = _compute_match(rfs=self.weights, input_arr=input_state)
        best_rf = np.argmax(eff_frame)
        select_criterion = eff_frame[best_rf]
        error = _compute_error(rfs=self.weights[best_rf, :], input_arr=input_state, single_rf=True)

        self.error[self.t] = error
        self.mean_error[self.t] = np.mean(self.error[max(0, self.t - self.error_mean_time_test):self.t])

        self.curr_batch_accum_step += 1

        # training if batch is full
        if self.curr_batch_accum_step == self.batch_length:
            print()
            print('*** starting training on batch ***')
            print()
            self._train_on_batch()
            self.curr_batch_accum_step = 0
            print()
            print('*** ended training on batch ***')
            print()

        self.t += 1

    def _train_on_batch(self):

        # if batch full, do iterations

        train_error = np.zeros(self.iters_per_batch * self.batch_length)
        mean_train_error = np.zeros(self.iters_per_batch * self.batch_length)

        t_train = 0
        for iter_i in range(self.iters_per_batch):

            print()
            print('    starting iter: ' + str(iter_i) + ' of ' + str(self.iters_per_batch))
            print()

            num_active_this_batch = np.zeros(self.num_rfs)

            for batch_t in range(self.batch_length):
                input_state = self.inputs_batch[batch_t, :]

                eff_frame = _compute_match(rfs=self.weights, input_arr=input_state)
                if self.enforce_max_firing:
                    eff_frame[num_active_this_batch >= self.num_firing_per_train_batch] = -np.inf
                best_rf = np.argmax(eff_frame)
                select_criterion = eff_frame[best_rf]
                error = _compute_error(rfs=self.weights[best_rf, :], input_arr=input_state, single_rf=True)

                num_active_this_batch[best_rf] += 1

                # adapt
                forgetful = self.forgetful_kmeans  # False
                if forgetful:
                    lr = self.lr
                    self.weights[best_rf, :] = lr * input_state + (1.0 - lr) * self.weights[best_rf, :]
                else:
                    self.rf_counts[best_rf] += 1
                    self.weights[best_rf, :] = self.weights[best_rf, :] + (1.0 / self.rf_counts[best_rf]) * (input_state - self.weights[best_rf, :])

                # update error in batch
                train_error[t_train] = error
                mean_train_error[t_train] = np.mean(train_error[max(0, t_train-self.error_mean_time_train):t_train])

                # self.rfs_raster_history[best_rf, self.t] = 1

                t_train += 1

            assert np.amax(num_active_this_batch) == np.amin(num_active_this_batch), str((np.amax(num_active_this_batch), np.amin(num_active_this_batch), self.num_firing_per_train_batch))
            assert np.amax(num_active_this_batch) == self.num_firing_per_train_batch

        # plot training error over batch iterations

        self._plot_training_error(mean_train_error)
        self.curr_batch += 1

    def _plot_training_error(self, mean_train_error):
        # plot error
        self.ax_train_error.cla()
        self.ax_train_error.plot(mean_train_error, color='k', marker='.')
        self.fig_train_error.savefig(self.plots_folder + "/train_error_batch_" + str(int(self.curr_batch)) + ".png", dpi=100)

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
        self.fig_test_error.savefig(self.plots_folder + "/test_error.png", dpi=100)

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

                input_state_p[input_state_p > 0] = 1  # input_state[input_state_p > 0]
                input_state_n[input_state_n > 0] = 1  # input_state[input_state_n > 0]

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














