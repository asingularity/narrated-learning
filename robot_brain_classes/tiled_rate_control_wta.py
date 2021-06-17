import time
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt
from brain_components_classes.states_history import StatesLimitedHistory
from cython_tiled_wta import cy_compute_error_measures, cy_kmeans_do_learning, cy_tile_the_input
from offline_analyses.try_sparsify_3 import make_im

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


class RateControlWTABRain(object):
    def __init__(self, params):

        # TODO MAKE PARAM
        # TODO re-enable for tiling
        self.enable_errors_plots = False

        # tiling params
        self.input_im_dim_NxN = params['input_im_dim']  # full input image dim: N, where NxN
        self.tile_dim_NxN = params['tile_im_dim']  # tile input dim: M, where MxM

        assert (self.input_im_dim_NxN / self.tile_dim_NxN) == np.round(self.input_im_dim_NxN / self.tile_dim_NxN)
        self.num_tiles_NxN = int(self.input_im_dim_NxN / self.tile_dim_NxN)

        # this is now: input state dim per tile
        self.tile_input_state_dim = 2 * self.tile_dim_NxN * self.tile_dim_NxN  # why 2? + and - changes

        # new params
        self.num_rfs = params['num_rfs']  # 400
        self.lr = params['lr']  # 1.0 / 1000
        self.rel_lr_bg = params['rel_lr_bg']   # 0.1
        self.max_time = params['max_time']  # 5000000
        self.do_raster_plots_every_k_im = params['do_raster_plots_every_k_im']  # 4, or None

        # for plotting:
        self.error_mean_time = 50000

        self.ims_since_raster = 0

        self.input_concat_timesteps = 1  # unused

        if 'network_type' in params:
            self.network_type = params['network_type']
        else:
            self.network_type = 'seq-kmeans'

        assert self.network_type == 'seq-kmeans' or self.network_type == 'seq-knn' or self.network_type == 'nn-inits-kmeans'

        self.nn_inits_kmeans_time = None  # only used for nn-inits-kmeans

        if 'seq-kmeans' in self.network_type:
            #self.weights = np.zeros((self.num_rfs, self.input_state_dim)) # + 1e-1
            # tiles are shared, so we only need one set of "tile state dim x tile state dim" weights
            self.weights = np.random.random((self.num_rfs, self.tile_input_state_dim)) * 1e-12
            self.weights = self.weights.astype(np.float32)

        elif 'seq-knn' in self.network_type:
            self.cuda_table = CudaTable(num_entries=self.num_rfs,
                                        input_dim=self.tile_input_state_dim)
            self.init_I_row_num = None

            self.num_row_changes_for_disp = 0
            self.frames_since_row_change_disp = 0
        elif 'nn-inits-kmeans' in self.network_type:
            # seq-nn init:
            self.cuda_table = CudaTable(num_entries=self.num_rfs,
                                        input_dim=self.tile_input_state_dim)
            self.init_I_row_num = None

            self.num_row_changes_for_disp = 0
            self.frames_since_row_change_disp = 0

            # seq-kmeans init later:
            self.weights = np.zeros((self.num_rfs, self.tile_input_state_dim))

            # time to transition them:
            self.nn_inits_kmeans_time = 100000

        self.rf_counts = np.zeros(self.num_rfs, np.float32)

        self.t = 0
        self.num_zero_inputs = 0

        self.plots_folder = "."

        self._init_plotting()

        # rasters
        # TODO raster combines all tiles
        self._init_rasters()

        self.sum_input_per_rf = np.zeros(self.num_rfs)
        self.num_input_per_rf = np.zeros(self.num_rfs)

        self.sum_errors_per_rf = {}
        self.num_errors_per_rf = {}
        self.mean_errors = {}

        # TILING: errors should now be average over all tiles, for now
        self.errors_over_time = {}
        self.mean_errors_over_time = {}

        # TODO re-enable other errors later: 'L2', 'L1', 'input-norm-L1', 'input-norm-L2',
        for error_type in ['RF-norm-dot']:
            self.sum_errors_per_rf[error_type] = np.zeros(self.num_rfs)
            self.num_errors_per_rf[error_type] = np.zeros(self.num_rfs)

            self.mean_errors[error_type] = np.zeros(self.max_time)  # per rf

            self.errors_over_time[error_type] = np.zeros(self.max_time)
            self.mean_errors_over_time[error_type] = np.zeros(self.max_time)

        self.last_reconstruction_im_info = None

    def _init_rasters(self):
        self.raster_steps = 200
        self.raster_t = 0  # circular; draw vertical line on plot here

        input_state_dim = int(2 * self.input_im_dim_NxN * self.input_im_dim_NxN)
        self.input_raster_history = np.zeros((input_state_dim, self.raster_steps), np.uint8)

        # TODO this needs to be set properly for multiple tiles!
        self.rfs_raster_history = np.zeros((self.num_rfs, self.raster_steps), np.uint8)

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

            d[error_type + '__mean-by-rf'] = self.mean_errors[error_type][self.t - 1]
            d[error_type + '__mean-by-t'] = self.mean_errors_over_time[error_type][self.t - 1]
        return d

    def set_plots_folder(self, folder):
        self.plots_folder = folder
        print()
        print('setting plots folder: ', self.plots_folder)
        print()

    def _get_error_measures(self, rfs, input_states_tiles):

        assert input_states_tiles.shape[0] == (self.num_tiles_NxN * self.num_tiles_NxN)
        assert input_states_tiles.shape[1] == self.tile_input_state_dim

        if rfs.shape[0] == self.tile_input_state_dim:
            rfs = rfs[np.newaxis, :]

        assert rfs.shape[1] == input_states_tiles.shape[1]

        # re-enable other errors later, have to modify for multiple tiles:
        #L2 = np.sqrt(np.sum(np.square(rfs - input_arr), axis=1))
        #L1 = np.sum(np.abs(rfs - input_arr), axis=1)
        #input_norm_L1 = np.sum(np.abs(rfs - input_arr), axis=1) / np.sum(input_arr)
        #input_norm_L2 = np.sqrt(np.sum(np.square(rfs - input_arr), axis=1)) / np.sum(np.square(input_arr))

        # why is this one negative? because it is actually a positive value: the larger, the better
        fix_offset = 1e-16

        # OLD:
        # RF_norm_dot = -np.divide(np.sum(np.multiply(rfs, input_arr), axis=1), np.sum(rfs, axis=1)+fix_offset)

        # TODO CYTHON modify for multiple tiles
        #   need to: tile / repeat the rfs and input arrs; then reshape the result correctly
        #   instead, do a cy loop so we don't run out of memory; otherwise not saving: may as well reuse tiles

        # float32?
        RF_norm_dot = np.zeros((int(self.num_tiles_NxN * self.num_tiles_NxN), self.num_rfs), np.float32)

        # print('cy_compute_error_measures')
        cy_compute_error_measures(rfs, input_states_tiles, fix_offset, RF_norm_dot)

        # re-enable other errors later: 'L2', 'L1', 'input-norm-L1', 'input-norm-L2',
        errors = {
            #'L2': L2,
            #'L1': L1,
            #'input-norm-L1': input_norm_L1,
            #'input-norm-L2': input_norm_L2,
            'RF-norm-dot': RF_norm_dot
        }

        return errors

    def process_input(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):

        if self.t >= self.max_time:
            return

        input_state_all = np.concatenate((input_events_p, input_events_n))
        assert input_state_all.dtype == np.float32, str(input_state_all.dtype)

        if input_state_all is None:  # this doesn't appear to ever be able to happen
            return

        if np.count_nonzero(input_state_all) == 0:
            self.num_zero_inputs += 1
            return

        self.input_raster_history[:, self.raster_t] = input_state_all[:]

        # input_state_all not used again this function

        input_states_tiles = np.zeros((int(self.num_tiles_NxN * self.num_tiles_NxN), self.tile_input_state_dim), np.float32)

        # CYTHON write this function:
        #   input_state becomes 2D: (num_tiles X self.tile_input_state_dim)
        # print('cy_tile_the_input')
        # print('input_events_p.shape', input_events_p.shape,
        #       '\ninput_events_n.shape', input_events_n.shape,
        #       '\nevent_coords_c.shape', event_coords_c.shape,
        #       '\nevent_coords_r.shape', event_coords_r.shape,
        #       '\nnp.amin(event_coords_r)', np.amin(event_coords_r),
        #       '\nnp.amax(event_coords_r)', np.amax(event_coords_r),
        #       '\nnp.amin(event_coords_c)', np.amin(event_coords_c),
        #       '\nnp.amax(event_coords_c)', np.amax(event_coords_c),
        #       '\nself.num_tiles_NxN', self.num_tiles_NxN,
        #       '\nself.tile_input_state_dim', self.tile_input_state_dim,
        #       '\nself.tile_dim_NxN', self.tile_dim_NxN
        #       )
        # print()

        cy_tile_the_input(input_events_p, input_events_n,
                          event_coords_r, event_coords_c,
                          self.num_tiles_NxN, self.tile_input_state_dim, self.tile_dim_NxN,
                          input_states_tiles)
        
        assert input_states_tiles.shape[0] == (self.num_tiles_NxN * self.num_tiles_NxN)
        assert input_states_tiles.shape[1] == self.tile_input_state_dim

        # *** selection ***
        best_rf_per_tile = None

        if 'seq-kmeans' in self.network_type or ('nn-inits-kmeans' in self.network_type and self.t > self.nn_inits_kmeans_time):
            # OLD:
            # errors = self._get_error_measures(rfs=self.weights, input_arr=input_state)
            # best_rf = np.argmin(errors['RF-norm-dot'])
            # REFACTOR:

            errors_all_rfs = self._get_error_measures(rfs=self.weights, input_states_tiles=input_states_tiles)

            assert errors_all_rfs['RF-norm-dot'].shape[0] == (self.num_tiles_NxN * self.num_tiles_NxN)
            assert errors_all_rfs['RF-norm-dot'].shape[1] == self.num_rfs

            best_rf_per_tile = np.argmin(errors_all_rfs['RF-norm-dot'], axis=1)

        elif 'seq-knn' in self.network_type or ('nn-inits-kmeans' in self.network_type and self.t <= self.nn_inits_kmeans_time):
            assert False, 'not yet supported for tiling'
            best_rf, best_rf_weights = self._step_seq_nn(input_state=input_state)

            # learning happens at same time for this one

            # return best_rf, best_rf_errors_dict

        if 'nn-inits-kmeans' in self.network_type and self.t == self.nn_inits_kmeans_time:
            assert False, 'not yet supported for tiling'
            print('Doing init of kmeans from seq-nn...')
            self.weights[:, :] = self.cuda_table.table_i[:, :]
            print('Init done: weights copied.')

        assert best_rf_per_tile is not None

        # *** errors for plotting ***
        if self.enable_errors_plots:

            # TODO modify for multiple tiles: for now average per RF over all tiles, since RF is reused?

            best_rf_weights = self.weights[best_rf, :]

            best_rf_errors_dict = self._get_error_measures(rfs=best_rf_weights, input_arr=input_state)

            for error_type in best_rf_errors_dict:
                error_value = best_rf_errors_dict[error_type][0]
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

        learn_happened = False
        if 'seq-kmeans' in self.network_type or ('nn-inits-kmeans' in self.network_type and self.t > self.nn_inits_kmeans_time):
            lr = self.lr

            # CYTHON for every tile in tile_input_arrs, learn best_rf_per_tile. some might be same RF! needs loop: cy
            #   python will apply only one of the changes, if multiple changes to same RF
            # add background learning if needed later

            # print('cy_kmeans_do_learning')
            cy_kmeans_do_learning(best_rf_per_tile, self.weights, input_states_tiles)

            # OLD:
            # self.weights[best_rf, :] = lr * input_state + (1.0 - lr) * self.weights[best_rf, :]

            # background learning
            # maybe this bg_lr should be per-rf dependent on its activity rate?
            # result: most converge to a non-useful average of inputs

            # lr_bg = self.rel_lr_bg * self.lr
            #if lr_bg > 0.0:
            #    self.weights = (1.0 - lr_bg) * self.weights

            learn_happened = True

        elif 'seq-knn' in self.network_type or ('nn-inits-kmeans' in self.network_type and self.t <= self.nn_inits_kmeans_time):
            assert False, 'not yet supported for tiling'
            learn_happened = True  # learning happens at same time as selection for this one, above

        assert learn_happened

        # store for making an image
        self.last_reconstruction_im_info = {
            'input_events_p': input_events_p,
            'input_events_n': input_events_n,
            'input_state_all': input_state_all,
            'best_rf_per_tile': best_rf_per_tile,  # for "reconstructed input events" image
            'input_states_tiles': input_states_tiles,  # for "actual input events" image
            'original_input_image': original_input_image  # for "original input" image
        }

        # *** time step ***

        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0

    def _step_seq_nn(self, input_state):

        input_state = input_state.astype(np.float32)

        dists = self.cuda_table.query(query_input=input_state)
        row_replaced = False

        # assert input_state.shape[0] == self.input_dim

        sorted_dist_indices = np.argsort(dists)
        new_min_ind = sorted_dist_indices[0]
        new_min_dist = dists[new_min_ind]

        error = np.sum(np.abs(self.cuda_table.table_i[new_min_ind, :] - input_state))

        if self.init_I_row_num is None:
            self.init_I_row_num = 0

        if self.init_I_row_num < self.cuda_table.get_num_rows():
            # necessary so dist matrix helper is not so slow at start
            dists[self.init_I_row_num] = np.inf
            try:
                self.cuda_table.set_matrix_row(row_index=self.init_I_row_num,
                                               row_input=input_state,
                                               row_to_table_dists=dists,
                                               fast_init=True)
            except AssertionError:
                print('\nError! Invalid GPU data type. input_state.dtype: ' + str(input_state.dtype) + '\n')
                raise

            row_replaced = True
            self.init_I_row_num += 1
        else:
            if not self.cuda_table.post_init_done:
                self.cuda_table.post_init()

            table_min_dist, table_min_dist_r, table_min_dist_c = self.cuda_table.get_min_dist()

            if new_min_dist > table_min_dist:
                # minimum distance of new row to current rows is greater than current minimum row-row distance
                # so: replace one row of current minimum, with new row

                # get one of the row indices of current minimum dist pair
                r_r_ind = table_min_dist_r  # could be table_min_dist_c

                dists[r_r_ind] = np.inf

                # replace the current min dist row, with the new row
                self.cuda_table.set_matrix_row(row_index=r_r_ind,
                                               row_input=input_state,
                                               row_to_table_dists=dists)

                row_replaced = True

        if row_replaced:
            self.num_row_changes_for_disp += 1
        self.frames_since_row_change_disp += 1

        #error = new_min_dist

        best_rf = new_min_ind
        best_rf_weights = self.cuda_table.table_i[best_rf, :]

        return best_rf, best_rf_weights

    def get_table_ims(self):

        if self.do_raster_plots_every_k_im is not None:
            self.ims_since_raster += 1
            if self.ims_since_raster > self.do_raster_plots_every_k_im:
                if self.enable_errors_plots:
                    self.do_plots()
                    self.ims_since_raster = 0

        ims_list = []
        ims_names_list = []

        weights = None
        if 'seq-kmeans' in self.network_type or ('nn-inits-kmeans' in self.network_type and self.t > self.nn_inits_kmeans_time):
            weights = self.weights
        elif 'seq-knn' in self.network_type  or ('nn-inits-kmeans' in self.network_type and self.t <= self.nn_inits_kmeans_time):
            weights = self.cuda_table.table_i

        rfs_im, rf_ims_dict = make_im(weights, num_bins_per_pixel=1,
                                               input_im_dim=self.tile_dim_NxN,
                                               im_final_dim=int(200 * 3000 / 400),  # /800 for two-im per rf display
                                               mod_for_disp=int(sqrt(self.num_rfs)),
                                               normalize_weights=True,
                                               borders_px=3)

        ims_list.append(rfs_im)
        ims_names_list.append('rfs_im')

        # defined above:

        # self.last_reconstruction_im_info = {
        #     'input_events_p': input_events_p,
        #     'input_events_n': input_events_n,
        #     'input_state_all': input_state_all,
        #     'best_rf_per_tile': best_rf_per_tile,  # for "reconstructed input events" image
        #     'input_states_tiles': input_states_tiles,  # for "actual input events" image
        #     'original_input_image': original_input_image  # for "original input" image
        # }

        input_state_all = self.last_reconstruction_im_info['input_state_all'][np.newaxis, :]

        input_all_im, _ = make_im(input_state_all, num_bins_per_pixel=1,
                                               input_im_dim=self.input_im_dim_NxN,
                                               im_final_dim=int(200 * 3000 / 1600),  # /800 for two-im per rf display
                                               mod_for_disp=1,
                                               normalize_weights=True,
                                               borders_px=1)

        ims_list.append(input_all_im)
        ims_names_list.append('input_all_im')



        # make an image showing just the input tiles this input: input_states_tiles

        # TODO whole thing is transposed
        # TODO within tile is not right: because positive/negative together? already account for in drawing tool...?
        # TODO compare to the full image for events that we get in --  get this from pre processor

        input_states_tiles = self.last_reconstruction_im_info['input_states_tiles']

        tile_inputs_im, tile_inputs_im_dict = make_im(input_states_tiles, num_bins_per_pixel=1,
                                               input_im_dim=self.tile_dim_NxN,
                                               im_final_dim=int(200 * 3000 / 1600),  # /800 for two-im per rf display
                                               mod_for_disp=self.num_tiles_NxN,
                                               normalize_weights=True,
                                               borders_px=1)  # 1

        ims_list.append(tile_inputs_im)
        ims_names_list.append('tile_inputs_im')



        # TODO add a full reconstruction event im over all tiles, side by side with original event im and original im
        #   need pre processor to also pass back original im for reference


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

        weights = None
        if 'seq-kmeans' in self.network_type or ('nn-inits-kmeans' in self.network_type and self.t > self.nn_inits_kmeans_time):
            weights = self.weights
        elif 'seq-knn' in self.network_type or ('nn-inits-kmeans' in self.network_type and self.t <= self.nn_inits_kmeans_time):
            weights = self.cuda_table.table_i

        self.ax_bar.cla()
        self.ax_bar.bar(np.arange(self.num_rfs), np.sum(np.abs(weights), axis=1))
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
