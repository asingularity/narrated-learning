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
from math import sqrt

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


def _compute_error(rfs, input_arr, single_rf=False):

    if single_rf:
        err_frame = np.sqrt(np.sum(np.square(rfs - input_arr)))
    else:
        err_frame = np.sqrt(np.sum(np.square(rfs - input_arr), axis=1))

    return err_frame


class RateControlSomWtaBrain(object):
    def __init__(self, params):
        self.input_im_dim = params['input_im_dim']
        self.num_rfs = 12 * 12
        self.lr = 0.0001
        self.rate_lr = 0.0001

        self.max_time = 100000000

        # for plotting:
        self.error_mean_time = 50000

        assert sqrt(self.num_rfs) == int(sqrt(self.num_rfs))

        self.do_raster_plots_every_k_im = 4
        self.ims_since_raster = 0

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

        self.weights = np.zeros((self.num_rfs, self.input_state_dim)) # + 1e-9
        #self.weights = np.random.random((self.num_rfs, self.input_state_dim)) * 1e-12

        self.rf_counts = np.zeros(self.num_rfs)

        self.mean_error = np.zeros(self.max_time)
        self.error = np.zeros(self.max_time)

        self.t = 0
        self.num_zero_inputs = 0

        self.plots_folder = "."

        self._init_plotting()

        # rate control SOM stuff
        self._init_som_rate_control()

    def _init_som_rate_control(self):
        # target number of time steps between events
        self.target_isi = self.num_rfs
        self.target_fr = 1.0 / self.target_isi
        self.last_win_time = -np.inf* np.ones(self.num_rfs)

        # grid

        # neighborhood functions parameters per RF

        self.neighborhood_lrs = self.lr * np.ones((self.num_rfs, self.num_rfs))
        self.neighboorhood_sizes = np.ones(self.num_rfs) * 0.1
        self.last_isi = np.inf * np.ones(self.num_rfs)
        self.rf_rf_distance = np.zeros((self.num_rfs, self.num_rfs))

        print()
        print('starting init of rf rf dist...')
        print()

        rf_i = 0
        for rf_r in range(int(sqrt(self.num_rfs))):
            for rf_c in range(int(sqrt(self.num_rfs))):
                rf_i_2 = 0
                for rf2_r in range(int(sqrt(self.num_rfs))):
                    for rf2_c in range(int(sqrt(self.num_rfs))):

                        self.rf_rf_distance[rf_i, rf_i_2] = sqrt(pow((rf_r - rf2_r), 2) + pow((rf_c - rf2_c), 2) )
                        rf_i_2 += 1
                rf_i += 1
        print('...done')
        print()

    def process_input(self, input_events_p, input_events_n):
        if self.t >= self.max_time:
            return

        input_state = np.concatenate((input_events_p, input_events_n))

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            self.num_zero_inputs += 1
            return

        err_frame = _compute_error(rfs=self.weights, input_arr=input_state)
        best_rf = np.argmin(err_frame)

        self.error[self.t] = _compute_error(rfs=self.weights[best_rf, :], input_arr=input_state, single_rf=True)
        self.mean_error[self.t] = np.mean(self.error[max(0, self.t - self.error_mean_time):self.t])

        # all RFs learn, weighed by neighborhood function of best_rf
        '''
            >>> neighborhood_lrs = np.random.random((16, 16))
            >>> lr_per_rf =neighborhood_lrs[5, :][:, np.newaxis]
            >>> lr_per_rf.shape
            (16, 1)
            >>> weights = np.random.random((16, 100))
            >>> input_state = np.random.random((100))
            >>> np.multiply(weights, lr_per_rf).shape
            (16, 100)
            >>> np.multiply(lr_per_rf, input_state).shape
            (16, 100)

        '''
        if self.t > 0:

            lr_per_rf = self.neighborhood_lrs[best_rf, :][:, np.newaxis]  # by default, this function should be 1.0 for best_rf
            lr_per_rf[best_rf, 0] = 100 * self.lr
            self.weights = np.multiply(lr_per_rf, input_state) + np.multiply(1.0 - lr_per_rf, self.weights)

        # adjust neighborhood functions based on ISI
        #   adjust for all RFs, but given actual last ISI (or zero if never)

        # last_isi should init np.inf
        self.last_isi[best_rf] = self.t - self.last_win_time[best_rf]
        fr_estimate = 1.0 / self.last_isi

        # positive: fr too high
        # negative: fr too low
        fr_error = fr_estimate - self.target_fr

        # if rate too high: increase neighborhood (influence)
        # if rate too low: decrease neighborhood (influence)
        self.neighboorhood_sizes = self.neighboorhood_sizes + self.rate_lr * fr_error

        # set neighborhood_lrs from neighborhood_sizes
        # gauss = np.exp(- dst / ( 2.0 * sigma**2 )  )

        term_1 = - self.rf_rf_distance
        term_2 = 2.0 * np.square(self.neighboorhood_sizes)[:, np.newaxis]
        self.neighborhood_lrs = self.lr * np.exp(np.divide(term_1, term_2))

        self.rf_counts[best_rf] += 1
        self.last_win_time[best_rf] = self.t
        self.t += 1

    def set_plots_folder(self, folder):
        self.plots_folder = folder
        print()
        print('setting plots folder: ', self.plots_folder)
        print()

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
                                                    im_final_dim=int(200 * 3000 / 400),  # /800 for two-im per rf display
                                                    mod_for_disp=int(sqrt(self.num_rfs)),
                                                    normalize_weights=True)

        # 400:
        # rfs_im, rf_ims_dict = make_im(self.weights, num_bins_per_pixel=1,
        #                                             input_im_dim=self.input_im_dim,
        #                                             im_final_dim=int(self.num_rfs * 3000 / 1600),  # /800 for two-im per rf display
        #                                             mod_for_disp=20,
        #                                             normalize_weights=True)

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
        self.ax_bar.bar(np.arange(self.num_rfs), self.neighboorhood_sizes)
        self.fig_bar.savefig(self.plots_folder + '/neighboorhood_sizes.png', dpi=100)
        #
        # self.ax_bar.cla()
        # self.ax_bar.bar(np.arange(self.num_rfs), self.error_multipliers)
        # self.fig_bar.savefig(self.plots_folder + '/error_multipliers.png', dpi=100)
