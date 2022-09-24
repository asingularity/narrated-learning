import time
import cv2

import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt
from brain_components_classes.states_history import StatesLimitedHistory
from offline_analyses.try_sparsify_3 import get_sparse_features, make_im

from utils.one_time_messages import OneTimeMessages

# WHYI IS THIS BROKEN NOW
#from cuda_dist_query import CudaTable

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log

np.set_printoptions(suppress=True, precision=2)

random.seed(0)
np.random.seed(0)


class IndirectRFBrain(object):
    def __init__(self, params):
        self.input_im_dim = params['input_im_dim']
        self.num_rfs = params['num_rfs']
        self.lr = params['lr']
        self.max_time = params['max_time']

        self.do_plots_every_k_sec = params['do_plots_every_k_sec']
        self.last_plot_time = time.time()

        self.t = 0

        # **************** input ****************

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

        self.plots_folder = "."
        self.input_concat_timesteps = 1
        self.input_history = StatesLimitedHistory(params={'max_delay': self.input_concat_timesteps,
                                                          'states_dim_list': [self.input_state_dim],
                                                          'store_extra_data': False})

        # **************** network ****************

        self.rf_weights = np.random.random((self.num_rfs, self.input_state_dim)) * 1e-2  # 1e-12

        # **************** plots ****************

        self.raster_steps = 200
        self.raster_t = 0  # circular; draw vertical line on plot here
        self.input_raster_history = np.zeros((self.input_state_dim, self.raster_steps), np.uint8)
        self.rfs_raster_history = np.zeros((self.num_rfs, self.raster_steps), np.uint8)

        self.fig_bar = plt.figure(figsize=(40, 20))
        self.ax_bar = self.fig_bar.add_subplot(1, 1, 1)
        self.ax_bar.cla()
        self.ax_bar.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_bar.get_yaxis().get_major_formatter().set_scientific(False)

        # self.fig_bar = plt.figure(figsize=(40, 40))
        # self.ax_bar_list = []
        # self.rfs_to_plot = 16  # 8 or self.num_rfs
        #
        # for k in range(self.rfs_to_plot):
        #     subpl = self.fig_bar.add_subplot(self.rfs_to_plot, 1, k + 1)
        #     subpl.cla()
        #     subpl.get_xaxis().get_major_formatter().set_scientific(False)
        #     subpl.get_yaxis().get_major_formatter().set_scientific(False)
        #     self.ax_bar_list.append(subpl)

    def get_final_errors_dict(self):
        d = {}
        return d

    def get_table_ims(self):

        ims_list = []
        ims_names_list = []

        rfs_im, rf_ims_dict = make_im(self.rf_weights,
                                      num_bins_per_pixel=1,
                                      input_im_dim=self.input_im_dim,
                                      im_final_dim=int(50 * 3000 / 800),  # /800 for two-im per rf display
                                      mod_for_disp=int(sqrt(self.num_rfs)),
                                      normalize_weights=True)

        ims_list.append(rfs_im)
        ims_names_list.append('prob_weights')

        return ims_list, ims_names_list

    def set_plots_folder(self, folder):
        self.plots_folder = folder
        print()
        print('setting plots folder: ', self.plots_folder)
        print()

    def do_plots(self):
        # look at dynamic_coincidence.py, others

        self.ax_bar.cla()
        num_rf = self.input_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.input_raster_history[0:num_rf, :],
                                               np.arange(num_rf)[:, np.newaxis]))

        t = np.arange(raster_plot.shape[0])
        self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.fig_bar.savefig(self.plots_folder + "/raster_input.png", dpi=100)

        # self.ax_2.cla()
        # num_rf = self.rfs_0_raster_history.shape[0]
        # raster_plot = np.transpose(np.multiply(self.rfs_0_raster_history[0:num_rf, max(0, self.t - 200):self.t],
        #                                        np.arange(num_rf)[:, np.newaxis]))
        # t = np.arange(raster_plot.shape[0])
        # self.ax_2.plot(t, raster_plot, color='b', marker='.', linestyle='')
        # self.fig.savefig(self.plots_folder + "/raster_" + self.layer_name + "_input_output.png", dpi=100)


    def process_input(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):

        if self.t >= self.max_time:
            return

        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        # for prediction, Don't skip zeros!!!
        # if np.count_nonzero(input_state) == 0:
        #     return

        self.input_history.process_new_states(newest_states_list=[input_state])

        if self.t < self.input_concat_timesteps:
            self.t += 1
            return

        if self.input_concat_timesteps > 1:
            state_seq = self.input_history.get_state_sequence(delay_long=self.input_concat_timesteps - 1, delay_short=0)
            print(state_seq.shape, self.input_state_dim, self.input_concat_timesteps, input_state.shape)  # (1, 512) 512 1 (512,)

            # input_state = np.sum(state_seq, axis=0)
            # input_state[input_state > 1] = 1
            # nnz_input = np.nonzero(input_state)[0]

        if self.do_plots_every_k_sec is not None:
            if time.time() - self.last_plot_time > self.do_plots_every_k_sec:
                self.do_plots()
                self.last_plot_time = time.time()

        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0
