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


class DynamicCoincidenceBrain(object):
    def __init__(self, params):
        self.input_im_dim = params['input_im_dim']
        self.num_rfs = params['num_rfs']
        self.lr = params['lr']
        self.max_time = params['max_time']

        self.do_plots_every_k_sec = params['do_plots_every_k_sec']
        self.last_plot_time = time.time()

        self.t = 0

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

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

        self.ax_bar_list = []
        self.rfs_to_plot = 8  # self.num_rfs

        for k in range(self.rfs_to_plot):
            subpl = self.fig_bar.add_subplot(self.rfs_to_plot, 1, k + 1)
            subpl.cla()
            subpl.get_xaxis().get_major_formatter().set_scientific(False)
            subpl.get_yaxis().get_major_formatter().set_scientific(False)
            self.ax_bar_list.append(subpl)

        self.rf_decaying = np.zeros((self.num_rfs, self.input_state_dim))

        self.rf_weights = np.random.random((self.num_rfs, self.input_state_dim)) * 1e-2  # 1e-12

        self.thresholds = 0.5 * np.ones(self.num_rfs)

        self.match_history_len = 2000
        self.match_history = StatesLimitedHistory(params={'max_delay': self.match_history_len,
                                                          'states_dim_list': [self.num_rfs],
                                                          'store_extra_data': False})

        self.spikes_history = StatesLimitedHistory(params={'max_delay': self.match_history_len,
                                                          'states_dim_list': [self.num_rfs],
                                                          'store_extra_data': False})

        self.thresh_history = StatesLimitedHistory(params={'max_delay': self.match_history_len,
                                                          'states_dim_list': [self.num_rfs],
                                                          'store_extra_data': False})

        self.max_match = 0.0

    def get_final_errors_dict(self):
        d = {}
        return d

    def get_table_ims(self):

        ims_list = []
        ims_names_list = []

        rfs_im, rf_ims_dict = make_im(self.rf_weights,
                                      num_bins_per_pixel=1,
                                      input_im_dim=self.input_im_dim,
                                      im_final_dim=int(50 * 3000 / 400),  # /800 for two-im per rf display
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

        match_hist = self.match_history.get_state_sequence(delay_short=0, delay_long=self.match_history_len - 1).transpose()
        spikes_hist = self.spikes_history.get_state_sequence(delay_short=0, delay_long=self.match_history_len - 1).transpose()
        thresh_hist = self.thresh_history.get_state_sequence(delay_short=0, delay_long=self.match_history_len - 1).transpose()

        if self.t > self.match_history_len:
            print()
            print('max_match', self.max_match)
            print('starting plot...')

            for k in range(self.rfs_to_plot):
                #print('    rf:', k)
                subpl = self.ax_bar_list[k]
                subpl.cla()
                subpl.plot(match_hist[k, :], color='b')
                subpl.plot(thresh_hist[k, :], color='g')

                sp_times = np.nonzero(spikes_hist[k, :].flatten())[0]
                for sp_time in sp_times:
                    subpl.axvline(x=sp_time, color='r')
                subpl.set_ylim([-0.1, 1.1])

            self.fig_bar.savefig(self.plots_folder + "/match_history.png", dpi=100)

            print('done plot.')
            self.max_match = 0.0

        if False: #self.t > self.match_history_len:
            # time plots:
            self.ax_bar.cla()
            match_hist = self.match_history.get_state_sequence(delay_short=0, delay_long=self.match_history_len - 1)
            print('doing plot:', match_hist.shape)   # (10000, 64)  # NEEDS NOT TRANSPOSE
            #self.ax_bar.plot(np.arange(self.match_history_len), match_hist)
            self.ax_bar.plot(match_hist, marker='.')
            self.fig_bar.savefig(self.plots_folder + "/match_history.png", dpi=100)

            #
            # self.ax_bar.cla()
            # num_rf = self.rfs_raster_history.shape[0]
            # raster_plot = np.transpose(np.multiply(self.rfs_raster_history, np.arange(num_rf)[:, np.newaxis]))
            # t = np.arange(raster_plot.shape[0])
            # self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
            # self.ax_bar.axvline(x=self.raster_t, color='g')
            # self.fig_bar.savefig(self.plots_folder + "/raster_rfs.png", dpi=100)

    def process_input(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):

        if self.t >= self.max_time:
            return

        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            return

        # TODO for prediction, Don't skip zeros!!!

        self.input_history.process_new_states(newest_states_list=[input_state])

        if self.t < self.input_concat_timesteps:
            self.t += 1
            return

        state_seq = self.input_history.get_state_sequence(delay_long=self.input_concat_timesteps - 1, delay_short=0)
        # print(state_seq.shape)  # (10000, 128): (M, L)

        input_state = np.sum(state_seq, axis=0)
        input_state[input_state > 1] = 1
        nnz_input = np.nonzero(input_state)[0]

        self.rf_decaying = self.rf_decaying * 0.9
        self.thresholds = self.thresholds * 0.999

        match = np.divide(np.sum(np.multiply(self.rf_weights, input_state), axis=1), np.sum(self.rf_weights, axis=1))

        self.max_match = max(self.max_match, np.amax(match))

        # ******************* first spike rule *******************
        # no WTA:
        # spikes = np.greater(match, self.thresholds).astype(np.int)

        # ******************* second spike rule *******************

        # if more than one spike, now we do a (k?-) WTA
        # this rule, often, nobody spikes because best match is only one below its threshold for example
        # BUT its a good rule
        # however: instead, should be which is most above its threshold?

        # spikes = np.zeros(self.num_rfs)
        # max_rf = np.argmax(match)
        # max_match_val = match[max_rf]
        # if max_match_val > self.thresholds[max_rf]:
        #     spikes[max_rf] = 1

        # ******************* third spike rule *******************

        spikes = np.zeros(self.num_rfs)

        prop_match_thresh = np.divide(match, self.thresholds)

        max_rf = np.argmax(prop_match_thresh)
        max_match_val = match[max_rf]
        if max_match_val > self.thresholds[max_rf]:
            spikes[max_rf] = 1

        # adjust threshold: set to right below match for spikes
        #   then, decay over time

        spike_rfs = np.nonzero(spikes)[0]
        self.thresholds[spike_rfs] = match[spike_rfs] * 0.99

        # adjust weights
        # increase for 1-input: proportional to coincidence
        # decrease for 0-input

        enable_learning = True
        if enable_learning:
            # original, old not good rule:
            #self.rf_weights[spike_rfs, :] = self.lr * input_state + (1.0 - self.lr) * self.rf_weights[spike_rfs, :]

            if len(spike_rfs) > 0:
                lr_all = self.lr * np.multiply(self.rf_decaying[spike_rfs, :], self.rf_weights[spike_rfs, :])
                #print(np.amin(lr_all), np.amax(lr_all))
                self.rf_weights[spike_rfs, :] = np.multiply(lr_all, input_state) + np.multiply((1.0 - lr_all), self.rf_weights[spike_rfs, :])

        self.rf_decaying[spike_rfs, :] = 1.0

        self.thresh_history.process_new_states(newest_states_list=[self.thresholds])
        self.match_history.process_new_states(newest_states_list=[match])
        self.spikes_history.process_new_states(newest_states_list=[spikes])

        # self.rf_decaying[np.argmax(match), nnz_input] += 1
        # self.rf_decaying[np.argmax(match), nnz_input] = 1
        # self.rf_weights = self.rf_decaying

        if self.do_plots_every_k_sec is not None:
            if time.time() - self.last_plot_time > self.do_plots_every_k_sec:
                self.do_plots()
                self.last_plot_time = time.time()

        # WTA stuff
        enable_learning = False
        if enable_learning:
            best_rf = np.argmax(match)
            self.rf_weights[best_rf, :] = self.lr * input_state + (1.0 - self.lr) * self.rf_weights[best_rf, :]

        self.t += 1


# should have shifting RFs over time, without long term learing, just due to dynamics
# i.e. should show representational shift without a further mechanism

# what is the simplest way to show concept?
# assume for now "single time step"
# there is a shared "remainder"?
# inhibition... this makes sense if predictive field is not the same as predicted field

# this is where "balanced inhibition excitation" matters





