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

        # TODO make into NEW PARAMS
        # TODO in sweep, compare total RFs given all layers to a flat simulation of that many total RFs
        self.num_layers = params['num_layers']  # 3
        self.layer_learn_time = params['layer_learn_time']  # 500000
        # for assigning error by RF, and for plotting:
        self.error_mean_time = 50000

        self.ims_since_raster = 0

        self.input_concat_timesteps = 1  # unused

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

        # per layer
        self.weights = []
        self.rf_counts = []

        self.parent_rfs = []
        self.child_rfs = []
        self.layer_init_times = []

        for layer_n in range(self.num_layers):
            self.layer_init_times.append(self.layer_learn_time * layer_n)
            layer_weights = np.random.random((self.num_rfs, self.input_state_dim)) * 1e-12
            self.weights.append(layer_weights)

            self.rf_counts.append(np.zeros(self.num_rfs))

            # if layer_n > 0:
            #       for each RF in this layer, what RF in previous layer is it a subset of?
            #       self.parent_rfs.append(np.zeros((self.num_rfs, self.num_rfs)))

            if layer_n < self.num_layers - 1:
                # for each RF in this layer, what RFs in next layer are subset of it?
                self.child_rfs.append(np.zeros((self.num_rfs, self.num_rfs)))

        # TODO update these for multi-layer
        self._init_rasters()
        self._init_errors_history()

        self.t = 0
        self.num_zero_inputs = 0

        self.plots_folder = "."
        self.input_history = StatesLimitedHistory(params={'max_delay': self.input_concat_timesteps,
                                                          'states_dim_list': [self.input_state_dim],
                                                          'store_extra_data': False})

        self._init_plotting()

    def _init_rasters(self):
        self.raster_steps = 200
        self.raster_t = 0  # circular; draw vertical line on plot here
        self.input_raster_history = np.zeros((self.input_state_dim, self.raster_steps), np.uint8)
        self.rfs_raster_history = np.zeros((self.num_rfs, self.raster_steps), np.uint8)

    def _init_errors_history(self):
        self.sum_errors_per_rf = {}
        self.num_errors_per_rf = {}
        self.mean_errors = {}

        self.errors_over_time = {}
        self.mean_errors_over_time = {}
        # not well bounded: 'RF-norm-L2', 'RF-norm-L1'
        for error_type in ['L2', 'L1', 'input-norm-L1', 'input-norm-L2', 'RF-norm-dot']:
            self.sum_errors_per_rf[error_type] = np.zeros((self.num_layers, self.num_rfs))
            self.num_errors_per_rf[error_type] = np.zeros((self.num_layers, self.num_rfs))

            self.mean_errors[error_type] = np.zeros(self.max_time)  # per rf

            self.errors_over_time[error_type] = np.zeros(self.max_time)
            self.mean_errors_over_time[error_type] = np.zeros(self.max_time)

            # TODO is there anything else needed to estimate proportion of error "caused by" this RF?

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

    def get_final_errors_dict(self):
        d = {}
        for error_type in self.sum_errors_per_rf.keys():

            # TODO fix later:
            # TODO this should be self.max_time - 1, but cant be because we skip zero inputs without self.t += 1 !!!

            d[error_type + '__mean-by-rf'] = self.mean_errors[error_type][self.t - 1]
            d[error_type + '__mean-by-t'] = self.mean_errors_over_time[error_type][self.t - 1]
        return d

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
        # should be 1-np.divide instead of -np.divide? yes
        RF_norm_dot = 1 - np.divide(np.sum(np.multiply(rfs, input_arr), axis=1), np.sum(rfs, axis=1))

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

        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            self.num_zero_inputs += 1
            return

        errors_dicts_per_layer = []
        winner_per_layer = []
        winner_per_layer_rel = []

        prev_layer_winner = None
        for layer_n in range(self.num_layers):

            #print(self.t, self.layer_init_times[layer_n])
            if self.t == self.layer_init_times[layer_n]:
                # at appropriate time, make mapping from RFs in prev layer to this layer: set from_rfs, to_rfs
                if layer_n == 0:
                    prop_child_rfs_per_rf_parent = None
                else:
                    # what proportion of new layer RFs should be init and assigned to prev layer RFs
                    prop_child_rfs_per_rf_parent = self._get_prop_child_rfs(layer_n=layer_n - 1)

                # init weights
                # assign child_rfs, parent_rfs
                # init RFs to parent RF
                self._init_layer_mapping(parent_layer_n=layer_n - 1,
                                         child_layer_n=layer_n,
                                         prop_child_rfs_per_rf_parent=prop_child_rfs_per_rf_parent)

            # do WTA, error update, learning
            layer_winner, errors_dict, layer_winner_rel = self._process_layer(layer_n=layer_n,
                                                                              input_state=input_state,
                                                                              prev_layer_parent=prev_layer_winner)

            winner_per_layer.append(layer_winner)
            winner_per_layer_rel.append(layer_winner_rel)
            errors_dicts_per_layer.append(errors_dict)

            prev_layer_winner = layer_winner

        self._update_errors(errors_dicts_per_layer=errors_dicts_per_layer, winner_per_layer=winner_per_layer, winner_per_layer_rel=winner_per_layer_rel)

        # *** time step ***

        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0

    def _get_prop_child_rfs(self, layer_n):
        prop_child_rfs = self.num_errors_per_rf['RF-norm-dot'][layer_n, :] * 1.0 / np.sum(self.num_errors_per_rf['RF-norm-dot'][layer_n, :])
        return prop_child_rfs

    def _init_layer_mapping(self, parent_layer_n, child_layer_n, prop_child_rfs_per_rf_parent):

        print()
        print('doing init layer mapping for parent, child: ', parent_layer_n, child_layer_n)
        print()

        if prop_child_rfs_per_rf_parent is None:
            assert child_layer_n == 0
            # no parent for first layer, no mapping to be made
            return

        assert parent_layer_n == child_layer_n - 1

        # gracefully handle if prop is such that no child RFs for a parent, here and in _process_layer

        tmp = prop_child_rfs_per_rf_parent * self.num_rfs
        num_child_rfs_per_parent = np.round(tmp).astype(np.int)

        print(np.sum(prop_child_rfs_per_rf_parent))
        print(prop_child_rfs_per_rf_parent)
        print(tmp)
        print(np.sum(tmp))
        print(num_child_rfs_per_parent)
        print(np.sum(num_child_rfs_per_parent))
        print()

        child_index = 0
        for parent_rf in range(self.num_rfs):
            num_child = num_child_rfs_per_parent[parent_rf]
            for ch in range(num_child):
                self.child_rfs[parent_layer_n][parent_rf, child_index] = 1
                self.weights[child_layer_n][child_index, :] = self.weights[parent_layer_n][parent_rf, :]
                child_index += 1

                # TODO hack
                if child_index > self.num_rfs - 1:
                    child_index = self.num_rfs - 1

        print('end of mapping: ')
        print('child_index', child_index)
        print()

    def _update_errors(self, errors_dicts_per_layer, winner_per_layer, winner_per_layer_rel):

        # how to compute errors for assigning at layer init vs. per layer vs. overall?
        #   option 1: all errors: STORE:
        #       for each error type
        #           for each layer
        #               per-RF mean error (sum / num)
        #               RF-centered mean error over time (averaged over RFs, regardless of relative activity)
        #               time-averaged mean error over time (accounting for relative activity)
        #   option 2: heirarchical error: STORE:
        #       (per RF in layer 0, child RF's error at maximum tree depth)
        #       for each error type:
        #           per-layer-0-RF mean max-tree-depth error (sum/num)
        #           layer-0-RF-centered mean max-tree-depth error over time
        #           time-averaged mean max-tree-depth error over time
        #
        #   !!! do option 2 !!!
        #       handle case where no child RFs in this layer exist for this prev_layer_winner- tree depth should be clear!

        final_error_dict = None
        best_rf = None
        best_rf_rel = None
        best_layer = None

        for layer_n in range(self.num_layers):
            if errors_dicts_per_layer[layer_n] is not None:
                final_error_dict = errors_dicts_per_layer[layer_n].copy()
                best_rf = winner_per_layer[layer_n]
                best_rf_rel = winner_per_layer_rel[layer_n]
                best_layer = layer_n

        assert final_error_dict is not None
        assert best_rf is not None

        # *** errors for plotting ***
        for error_type in final_error_dict:

            #print(error_type, final_error_dict[error_type].shape, best_rf)

            error_value = final_error_dict[error_type][best_rf_rel]

            self.errors_over_time[error_type][self.t] = error_value
            self.mean_errors_over_time[error_type][self.t] = np.mean(self.errors_over_time[error_type][max(0, self.t - self.error_mean_time):self.t])

            # per layer

            self.sum_errors_per_rf[error_type][best_layer, best_rf] += error_value
            self.num_errors_per_rf[error_type][best_layer, best_rf] += 1

            # mean over all layers
            tmp = np.divide(self.sum_errors_per_rf[error_type], self.num_errors_per_rf[error_type])
            tmp = tmp[np.nonzero(np.logical_not(np.isnan(tmp)))]
            self.mean_errors[error_type][self.t] = np.mean(tmp)

    def _process_layer(self, layer_n, input_state, prev_layer_parent):

        # handle case where no child RFs in this layer exist for prev_layer_parent:
        #   errors_dict and best_rf are None in this case

        no_rfs_this_layer = False

        if layer_n == 0:
            assert prev_layer_parent is None
            rf_indices = np.arange(self.num_rfs)
        else:

            if prev_layer_parent is None:
                # haven't init prev layer yet
                return None, None, None

            thing = self.child_rfs[layer_n - 1][prev_layer_parent, :].flatten()
            rf_indices = np.nonzero(thing)[0]

            #print()
            #print(layer_n)
            #print(prev_layer_parent)
            #print(thing.shape)
            #print(rf_indices)
            #print()

            if len(rf_indices) == 0:
                no_rfs_this_layer = True

        if no_rfs_this_layer:
            errors_dict = None
            best_rf = None
            best_rf_rel = None
        else:
            # *** selection ***
            try:
                errors_dict = self._get_error_measures(rfs=self.weights[layer_n][rf_indices, :], input_arr=input_state)
            except:
                print('***************')
                print(layer_n)
                print(rf_indices)
                print(len(rf_indices))
                print(self.weights[layer_n].shape)
                raise

            best_rf_rel = np.argmin(errors_dict['RF-norm-dot'])
            best_rf = rf_indices[best_rf_rel]

            # TODO this is a hack; should also do raster for other layers
            if layer_n == 0:
                self.rfs_raster_history[:, self.raster_t] = 0
                self.rfs_raster_history[best_rf, self.raster_t] = 1
                #self.rf_counts[best_rf] += 1

            # *** learning ***

            lr = self.lr

            do_learn = True
            if layer_n < self.num_layers - 1:
                if self.t >= self.layer_init_times[layer_n + 1]:
                    do_learn = False


            if do_learn:
                self.weights[layer_n][best_rf, :] = lr * input_state + (1.0 - lr) * self.weights[layer_n][best_rf, :]

        return best_rf, errors_dict, best_rf_rel

    def get_table_ims(self):

        if self.do_raster_plots_every_k_im is not None:
            self.ims_since_raster += 1
            if self.ims_since_raster > self.do_raster_plots_every_k_im:
                self.do_plots()
                self.ims_since_raster = 0

        ims_list = []
        ims_names_list = []

        for layer_n in range(self.num_layers):

            rfs_im, rf_ims_dict = make_im(self.weights[layer_n], num_bins_per_pixel=1,
                                                                 input_im_dim=self.input_im_dim,
                                                                 im_final_dim=int(200 * 3000 / 800),  # /800 for two-im per rf display
                                                                 mod_for_disp=int(sqrt(self.num_rfs)),
                                                                 normalize_weights=True)

            ims_list.append(rfs_im)
            ims_names_list.append('rfs_im_' + str(layer_n))

        return ims_list, ims_names_list

        # plot error over time (including over iterations within batches)
        # show RFs

    def do_plots(self):

        if 0:
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

            for layer_n in range(self.num_layers):
                # mean error per rf
                self.ax_bar.cla()
                self.ax_bar.bar(np.arange(self.num_rfs), np.divide(self.sum_errors_per_rf[error_type][layer_n, :], self.num_errors_per_rf[error_type][layer_n, :]))
                self.fig_bar.savefig(self.plots_folder + '/per_rf_mean_error_' + error_type + '_layer_' + str(layer_n) + '.png', dpi=100)

            # this is time averaged error, per time point; weighing all time points equally,
            # taking relative frequency of RFs into account
            self.ax_test_error.cla()
            self.ax_test_error.plot(self.mean_errors_over_time[error_type][0:self.t], color='k', marker='.')
            self.fig_test_error.savefig(self.plots_folder + "/time_averaged_mean_error_" + error_type + ".png", dpi=100)

        if 0:
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

