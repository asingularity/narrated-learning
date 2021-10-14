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


class CBrain(object):
    def __init__(self, params):
        self.input_im_dim = params['input_im_dim']
        self.num_rfs = params['num_rfs']
        self.lr = params['lr']
        self.max_time = params['max_time']

        self.do_raster_plots_every_k_im = params['do_raster_plots_every_k_im']
        self.ims_since_raster = 0

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

        self.prob_weights = np.random.random((self.num_rfs, self.input_state_dim)) * 1e-2  # 1e-12

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

        self.t = 0

        # new stuff
        self.m_len = 5000
        self.measure_history = StatesLimitedHistory(params={'max_delay': self.m_len,
                                                            'states_dim_list': [self.num_rfs],
                                                            'store_extra_data': False})

    def _calculate_measure(self, w, input_state):

        if 0:
            fix_offset = 1e-16
            rf_norm_dot = np.divide(np.sum(np.multiply(w, input_state), axis=1), np.sum(w, axis=1) + fix_offset)
            total_err = - rf_norm_dot
            return total_err
        else:
            fp = w - input_state
            fp[fp < 0] = 0
            fn = input_state - w
            fn[fn < 0] = 0

            tp = np.sum(np.multiply(w, input_state), axis=1)

            # print(fp.shape, fn.shape)  # (400, 128)
            #print('fp+fn')
            #print(np.sum(fp + fn, axis=1) )
            #print('tp')
            #print(tp)

            #total_err = np.sum(fp + fn, axis=1) - tp
            fp_sum = np.sum(fp, axis=1)
            #print('tp', tp)
            #print('fp_sum', fp_sum)
            total_err = fp_sum - tp
            #print('total_err', total_err)
            return total_err

    def process_input(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):

        if self.t >= self.max_time:
            return

        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            return

        self.rfs_raster_history[:, self.raster_t] = 0

        # calculate measure
        measure_per_rf = self._calculate_measure(w=self.prob_weights, input_state=input_state)

        # learn
        m_hist = self.measure_history.get_state_sequence(delay_long=self.m_len, delay_short=0)
        # print(m_hist.shape)  # (1001, 400)
        if self.t > self.m_len:
            #print('t', self.t)
            #print('m_hist')
            #print(m_hist)
            #print('measure_per_rf')
            #print(measure_per_rf)
            learn_now = np.nonzero(measure_per_rf < np.amin(m_hist, axis=0))[0]
            if len(learn_now) > 0:
                #print(self.t, learn_now)
                lr = self.lr
                self.prob_weights[learn_now, :] = lr * input_state + (1.0 - lr) * self.prob_weights[learn_now, :]

        self.measure_history.process_new_states(newest_states_list=[measure_per_rf])

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

        rfs_im, rf_ims_dict = make_im(self.prob_weights, num_bins_per_pixel=1,
                                                    input_im_dim=self.input_im_dim,
                                                    im_final_dim=int(200 * 3000 / 800),  # /800 for two-im per rf display
                                                    mod_for_disp=int(sqrt(self.num_rfs)),
                                                    normalize_weights=True)

        ims_list.append(rfs_im)
        ims_names_list.append('prob_weights')

        return ims_list, ims_names_list

    def do_plots(self):

        # time plots:
        self.ax_bar.cla()
        num_rf = self.rfs_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.rfs_raster_history, np.arange(num_rf)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.ax_bar.axvline(x=self.raster_t, color='g')
        self.fig_bar.savefig(self.plots_folder + "/raster_rfs.png", dpi=100)
