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

from cuda_dist_query import CudaTable

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log

np.set_printoptions(suppress=True, precision=2)

random.seed(0)
np.random.seed(0)


class IterWTABrain(object):
    def __init__(self, params):
        self.input_im_dim = params['input_im_dim']
        self.num_rfs = params['num_rfs']
        self.lr = params['lr']
        self.max_time = params['max_time']

        self.do_raster_plots_every_k_im = params['do_raster_plots_every_k_im']
        self.ims_since_raster = 0

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

        self.weights = np.random.random((self.num_rfs, self.input_state_dim)) * 1e-2  # 1e-12
        self.weights = self.weights.astype(np.float32)

        self.bias = np.ones(self.num_rfs) * 1e-3

        self.plots_folder = "."
        self.input_concat_timesteps = 1
        self.input_history = StatesLimitedHistory(params={'max_delay': self.input_concat_timesteps,
                                                          'states_dim_list': [self.input_state_dim],
                                                          'store_extra_data': False})

        self.raster_steps = 200
        self.raster_t = 0  # circular; draw vertical line on plot here
        self.input_raster_history = np.zeros((self.input_state_dim, self.raster_steps), np.uint8)
        self.rfs_raster_history = np.zeros((self.num_rfs, self.raster_steps), np.uint8)

        self.fig_bar = plt.figure(figsize=(40, 20))
        self.ax_bar = self.fig_bar.add_subplot(1, 1, 1)
        self.ax_bar.cla()
        self.ax_bar.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_bar.get_yaxis().get_major_formatter().set_scientific(False)

        self.error_mean_time = 50000

        self.rf_norm_dot_per_frame = np.zeros(self.max_time)
        self.mean_rf_norm_dot_per_frame = np.zeros(self.max_time)  # time average

        self.net_predicted_per_frame = np.zeros(self.max_time)
        self.mean_net_predicted_per_frame = np.zeros(self.max_time)

        self.t = 0

        self.weights_eff = None

    def _get_weights_eff(self, w):

        # weights -> weights eff via transform to bimoda
        # w_eff = w

        w_eff = w.copy()

        max_w = np.amax(w)

        # this boosts to 0.5
        cutoff = max_w / 2

        # this boosts to 0.9 BUT tiny rfs
        #cutoff = 3 * max_w / 4

        #w_eff[w_eff < cutoff] = 0
        #w_eff[w_eff >= cutoff] = 1

        # this boosts rf norm dot to 0.5:

        # w_eff = np.power(w, 2)

        # does nothing:
        # w_eff = w_eff * 1.0 / max_w

        return w_eff

    def process_input_PROB(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):
        self.weights_eff = self.weights

        if self.t >= self.max_time:
            return

        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            return

        fix_offset = 1e-16

        self.rfs_raster_history[:, self.raster_t] = 0

        RF_norm_dot = np.divide(np.sum(np.multiply(self.weights, input_state), axis=1), np.sum(self.weights, axis=1) + fix_offset)

        best_rf = np.argmax(RF_norm_dot)

        self.rf_norm_dot_per_frame[self.t] = RF_norm_dot[best_rf]
        self.mean_rf_norm_dot_per_frame[self.t] = np.mean(self.rf_norm_dot_per_frame[max(0, self.t - self.error_mean_time):self.t])

        self.rfs_raster_history[best_rf, self.raster_t] = 1

        lr = self.lr
        self.weights[best_rf, :] = lr * input_state + (1.0 - lr) * self.weights[best_rf, :]

        lr = self.lr / 10
        also_learn_this_frame = np.nonzero(np.random.random(self.num_rfs) < np.power(RF_norm_dot, 10))
        self.weights[also_learn_this_frame, :] = lr * input_state + (1.0 - lr) * self.weights[also_learn_this_frame, :]

        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0


    def process_input(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):

        if self.weights_eff is None:
            # this is an init done once
            self.weights_eff = np.zeros_like(self.weights)
            for k in range(self.num_rfs):
                self.weights_eff[k, :] = self._get_weights_eff(w=self.weights[k, :])

        if self.t >= self.max_time:
            return

        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            return

        fix_offset = 1e-16

        self.rfs_raster_history[:, self.raster_t] = 0

        RF_norm_dot = np.divide(np.sum(np.multiply(self.weights_eff, input_state), axis=1), np.sum(self.weights_eff, axis=1) + fix_offset)

        best_rf = np.argmax(RF_norm_dot)

        sum_input = np.sum(input_state)
        sum_rf = np.sum(self.weights_eff[best_rf, :])
        sum_dot = np.sum(np.multiply(input_state, self.weights_eff[best_rf, :]))

        fp = sum_rf - sum_dot
        fn = sum_input - sum_dot

        net_predicted = sum_dot - fp

        self.rf_norm_dot_per_frame[self.t] = RF_norm_dot[best_rf]
        self.mean_rf_norm_dot_per_frame[self.t] = np.mean(self.rf_norm_dot_per_frame[max(0, self.t - self.error_mean_time):self.t])

        self.net_predicted_per_frame[self.t] = net_predicted
        self.mean_net_predicted_per_frame[self.t] = np.mean(self.net_predicted_per_frame[max(0, self.t - self.error_mean_time):self.t])

        self.rfs_raster_history[best_rf, self.raster_t] = 1

        lr = self.lr
        self.weights[best_rf, :] = lr * input_state + (1.0 - lr) * self.weights[best_rf, :]

        self.weights_eff[best_rf, :] = self._get_weights_eff(w=self.weights[best_rf, :])

        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0

    def get_final_errors_dict(self):
        d = {}
        return d

    def set_plots_folder(self, folder):
        self.plots_folder = folder
        print()
        print('setting plots folder: ', self.plots_folder)
        print()


    def old_code(self):
        return
        #self.bias = self.bias + 0.01 * self.lr * 0.1
        self.bias[rf_out_events] = self.bias[rf_out_events] - 2 * self.lr * 0.1

        self.bias[self.bias < 0] = 0
        self.bias[self.bias > 1] = 1

        # instead of bias, sort in order of RF norm dot and subtract form input?



        if False:  # self.t > 5000:

            # net predicted negative: means on this frame we made more incorrect predictions than correct ones

            print()
            print('*****************************************')
            print('sum input: ', sum_input)
            print('sum rf: ', sum_rf)
            print('sum rf*input: ', sum_dot)
            print('fp', fp)
            print('fn', fn)

            print()



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
                                                    im_final_dim=int(200 * 3000 / 800),  # /800 for two-im per rf display
                                                    mod_for_disp=int(sqrt(self.num_rfs)),
                                                    normalize_weights=True)

        ims_list.append(rfs_im)
        ims_names_list.append('rfs_im')

        rfs_im, rf_ims_dict = make_im(self.weights_eff, num_bins_per_pixel=1,
                                                    input_im_dim=self.input_im_dim,
                                                    im_final_dim=int(200 * 3000 / 800),  # /800 for two-im per rf display
                                                    mod_for_disp=int(sqrt(self.num_rfs)),
                                                    normalize_weights=True)

        ims_list.append(rfs_im)
        ims_names_list.append('eff_rfs_im')

        return ims_list, ims_names_list


    def do_plots(self):

        self.ax_bar.cla()
        num_rf = self.rfs_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.rfs_raster_history, np.arange(num_rf)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.ax_bar.axvline(x=self.raster_t, color='g')
        self.fig_bar.savefig(self.plots_folder + "/raster_rfs.png", dpi=100)

        self.ax_bar.cla()
        self.ax_bar.plot(self.mean_rf_norm_dot_per_frame[0:self.t], color='k', marker='.')
        self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_rf_norm_dot.png", dpi=100)

        self.ax_bar.cla()
        self.ax_bar.plot(self.mean_net_predicted_per_frame[0:self.t], color='k', marker='.')
        self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_net_predict.png", dpi=100)

        skip_input_raster = True
        if not skip_input_raster:
            self.ax_bar.cla()
            num_rf = self.input_raster_history.shape[0]
            raster_plot = np.transpose(np.multiply(self.input_raster_history, np.arange(self.input_state_dim)[:, np.newaxis]))
            t = np.arange(raster_plot.shape[0])
            self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
            self.ax_bar.axvline(x=self.raster_t, color='g')
            self.fig_bar.savefig(self.plots_folder + "/raster_inputs.png", dpi=100)



    def process_input_EXPERIMENT(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):
        if self.t >= self.max_time:
            return

        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            return

        fix_offset = 1e-16
        weights_eff = self.weights
        RF_norm_dot = np.divide(np.sum(np.multiply(weights_eff, input_state), axis=1), np.sum(weights_eff, axis=1) + fix_offset)
        rf_out_events = np.nonzero(RF_norm_dot > 0.5)[0]

        self.rfs_raster_history[:, self.raster_t] = 0
        self.rfs_raster_history[rf_out_events, self.raster_t] = 1

        lr = self.lr
        self.weights[rf_out_events, :] = lr * input_state + (1.0 - lr) * self.weights[rf_out_events, :]

        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0


    def process_input_ITER(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):
        if self.t >= self.max_time:
            return

        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            return

        #weights_eff = self.weights - self.bias[:, np.newaxis]
        #weights_eff[weights_eff < 0] = 0


        fix_offset = 1e-16
        max_rfs_active = 20
        keep_going = True

        # print('*******************')
        # print(self.t)
        self.rfs_raster_history[:, self.raster_t] = 0

        k = 0
        while keep_going:

            RF_norm_dot = np.divide(np.sum(np.multiply(self.weights, input_state), axis=1), np.sum(self.weights, axis=1)+fix_offset)
            rf_out_events = np.nonzero(RF_norm_dot > 0.5)[0]

            if len(rf_out_events) > 0 and k < max_rfs_active:
                rf_out_val = RF_norm_dot[rf_out_events]

                best_rf = rf_out_events[np.argmax(rf_out_val)]

                # print('    ', best_rf, RF_norm_dot[best_rf])

                self.rfs_raster_history[best_rf, self.raster_t] = 1

                lr = self.lr
                self.weights[best_rf, :] = lr * input_state + (1.0 - lr) * self.weights[best_rf, :]

                input_state = input_state - self.weights[best_rf, :]
                input_state[input_state < 0] = 0

            else:

                # learn one anyways, but no event

                best_rf = np.argmax(RF_norm_dot)

                # experiment to force better RF-norm-dot
                #lr = self.lr * pow(np.amax(RF_norm_dot), 3)

                lr = self.lr
                self.weights[best_rf, :] = lr * input_state + (1.0 - lr) * self.weights[best_rf, :]

                keep_going = False

            k += 1

        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0
