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

        self.t = 0

        # new mechanism
        self.lr_factor = self.lr # * 0.1
        self.factor_per_rf = np.ones(self.num_rfs)

        self.last_weights_eff = self.weights.copy()

        self.num_per_rf = np.zeros(self.num_rfs)

    def process_input(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):

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

        weights_eff = np.power(self.weights, self.factor_per_rf[:, np.newaxis])

        self.last_weights_eff = weights_eff

        RF_norm_dot = np.divide(np.sum(np.multiply(weights_eff, input_state), axis=1), np.sum(weights_eff, axis=1) + fix_offset)

        best_rf = np.argmax(RF_norm_dot)

        self.rf_norm_dot_per_frame[self.t] = RF_norm_dot[best_rf]
        self.mean_rf_norm_dot_per_frame[self.t] = np.mean(self.rf_norm_dot_per_frame[max(0, self.t - self.error_mean_time):self.t])

        # target_rf_norm_dot = 0.9
        # if RF_norm_dot[best_rf] < target_rf_norm_dot:
        #     self.factor_per_rf[best_rf] *= (1.0 + self.lr_factor)
        # else:
        #     self.factor_per_rf[best_rf] *= (1.0 - self.lr_factor)
        learn_factor = True
        if learn_factor:
            self.factor_per_rf *= (1.0 + self.lr_factor * (1.0 / self.num_rfs))
            self.factor_per_rf[best_rf] *= (1.0 - self.lr_factor)

        self.rfs_raster_history[best_rf, self.raster_t] = 1

        lr = self.lr
        self.weights[best_rf, :] = lr * input_state + (1.0 - lr) * self.weights[best_rf, :]

        self.num_per_rf[best_rf] += 1

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

        rfs_im, rf_ims_dict = make_im(self.last_weights_eff, num_bins_per_pixel=1,
                                                    input_im_dim=self.input_im_dim,
                                                    im_final_dim=int(200 * 3000 / 400),  # /800 for two-im per rf display
                                                    mod_for_disp=int(sqrt(self.num_rfs)),
                                                    normalize_weights=True)

        ims_list.append(rfs_im)
        ims_names_list.append('rfs_eff_im')

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
        self.ax_bar.bar(np.arange(self.num_rfs), self.factor_per_rf)
        self.fig_bar.savefig(self.plots_folder + '/factor_per_rf.png', dpi=100)

        skip_input_raster = True
        if not skip_input_raster:
            self.ax_bar.cla()
            num_rf = self.input_raster_history.shape[0]
            raster_plot = np.transpose(np.multiply(self.input_raster_history, np.arange(self.input_state_dim)[:, np.newaxis]))
            t = np.arange(raster_plot.shape[0])
            self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
            self.ax_bar.axvline(x=self.raster_t, color='g')
            self.fig_bar.savefig(self.plots_folder + "/raster_inputs.png", dpi=100)

        self.ax_bar.cla()
        self.ax_bar.bar(np.arange(self.num_rfs), self.num_per_rf)
        self.fig_bar.savefig(self.plots_folder + '/num_per_rf.png', dpi=100)

        #self.num_per_rf = np.zeros(self.num_rfs)
