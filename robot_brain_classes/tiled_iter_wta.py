import time
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt
from brain_components_classes.states_history import StatesLimitedHistory
from cython_tiled_wta import cy_compute_error_measures, cy_kmeans_do_learning, cy_tile_the_input, cy_tile_the_input_layer_N, cy_kmeans_do_learning_per_tile_lr
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


class IterWTABRainLayerN(object):
    def __init__(self, params):

        # ******************************* params *******************************

        # easiest: everything specified relative to previous layer?
        self.input_num_tiles_NxN = params['input_num_tiles_NxN']
        self.input_num_rfs_per_tile = params['input_num_rfs_per_tile']  # dim for one input tile (num_rfs per tile in previous layer)
        self.tile_dim_NxN = params['tile_dim_NxN']

        self.num_tiles_NxN = int(self.input_num_tiles_NxN / self.tile_dim_NxN)
        self.tile_input_state_dim = self.input_num_rfs_per_tile * self.tile_dim_NxN * self.tile_dim_NxN

        self.num_rfs = params['num_rfs']  # 400: per tile
        self.lr = params['lr']  # 1.0 / 1000

        self.input_concat_timesteps = params['input_concat_timesteps']
        self.normalize_concat = True  # TODO investigate if this can be false; rf norm dot can go over 1 then ...

        self.max_time = params['max_time']  # 5000000

        self.num_iters_per_frame = params['num_iters_per_frame']  # 6
        self.do_conditional_lr = params['do_conditional_lr']  # False
        self.subtract_remainder = params['subtract_remainder']  # True
        self.force_all_frames_wta = True  # force every frame for now to coherently see error

        # ******************************* input concat + weights *******************************

        total_input_state_dim = int(self.input_num_tiles_NxN * self.input_num_tiles_NxN * self.input_num_rfs_per_tile)

        self.input_history = StatesLimitedHistory(params={'max_delay': self.input_concat_timesteps,
                                                          'states_dim_list': [total_input_state_dim],
                                                          'store_extra_data': False})

        self.weights = np.random.random((self.num_rfs, self.tile_input_state_dim)) * 1e-12
        self.weights = self.weights.astype(np.float32)

        # ******************************* plotting *******************************

        self.plots_folder = "."

        self.raster_steps = 200
        self.raster_t = 0  # circular; draw vertical line on plot here
        self.rfs_raster_history = np.zeros((self.num_rfs * self.num_tiles_NxN * self.num_tiles_NxN, self.raster_steps), np.uint8)

        self.fig_bar = plt.figure(figsize=(40, 20))
        self.ax_bar = self.fig_bar.add_subplot(1, 1, 1)
        self.ax_bar.cla()
        self.ax_bar.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_bar.get_yaxis().get_major_formatter().set_scientific(False)

        # ******************************* vars to track *******************************

        # histories of "errors"
        self.error_mean_time = 50000

        self.rf_norm_dot_per_frame = np.zeros(self.max_time)
        self.mean_rf_norm_dot_per_frame = np.zeros(self.max_time)  # time average

        self.rf_size_per_frame = np.zeros(self.max_time)
        self.mean_rf_size_per_frame = np.zeros(self.max_time)  # time average

        self.reconstruct_err_per_frame = np.zeros(self.max_time)
        self.mean_reconstruct_err_per_frame = np.zeros(self.max_time)

        # ******************************* init sim *******************************

        self.t = 0

    def get_input_num_tiles_NxN(self):
        return self.input_num_tiles_NxN

    def get_input_num_rfs_per_tile(self):
        return self.input_num_rfs_per_tile

    def get_tile_dim_NxN(self):
        return self.tile_dim_NxN

    def get_num_tiles_NxN(self):
        return self.num_tiles_NxN

    def get_num_rfs_per_tile(self):
        return self.num_rfs

    def set_plots_folder(self, folder):
        self.plots_folder = folder
        print()
        print('setting plots folder: ', self.plots_folder)
        print()

    def do_plots(self, extra_info=''):

        max_tiles_to_plot = 16  # so we don't plot a million things

        #print('STARTING PLOTS')
        self.ax_bar.cla()
        num_rf = self.rfs_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.rfs_raster_history, np.arange(num_rf)[:, np.newaxis]))

        t = np.arange(raster_plot.shape[0])

        self.ax_bar.plot(t, raster_plot[:, 0:min(max_tiles_to_plot * self.num_rfs, num_rf)], color='b', marker='.', linestyle='')
        self.ax_bar.axvline(x=self.raster_t, color='g')

        plot_y = 0
        for k in range(min(max_tiles_to_plot, self.num_tiles_NxN * self.num_tiles_NxN)):
            self.ax_bar.axhline(y=plot_y, color='r')
            plot_y += self.num_rfs

        if len(extra_info) > 0:
            self.fig_bar.savefig(self.plots_folder + "/raster_rfs_" + extra_info + ".png", dpi=100)
        else:
            self.fig_bar.savefig(self.plots_folder + "/raster_rfs.png", dpi=100)

        self.ax_bar.cla()
        self.ax_bar.plot(self.mean_rf_norm_dot_per_frame[0:self.t], color='k', marker='.')

        if len(extra_info) > 0:
            self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_rf_norm_dot_" + extra_info + ".png", dpi=100)
        else:
            self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_rf_norm_dot.png", dpi=100)

        self.ax_bar.cla()
        self.ax_bar.plot(self.mean_rf_size_per_frame[0:self.t], color='k', marker='.')

        if len(extra_info) > 0:
            self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_rf_size_" + extra_info + ".png", dpi=100)
        else:
            self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_rf_size.png", dpi=100)

        self.ax_bar.cla()
        self.ax_bar.plot(self.mean_reconstruct_err_per_frame[0:self.t], color='k', marker='.')
        self.ax_bar.ticklabel_format(useOffset=False)
        self.ax_bar.get_xaxis().get_major_formatter().set_useOffset(False)
        self.ax_bar.get_yaxis().get_major_formatter().set_useOffset(False)

        if len(extra_info) > 0:
            self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_reconstruct_err_" + extra_info + ".png", dpi=100)
        else:
            self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_reconstruct_err.png", dpi=100)


    def process_input(self, input_events, do_learn, do_run_layer, extra_info=None):

        output_events = np.zeros((self.num_tiles_NxN, self.num_tiles_NxN, self.num_rfs), np.float32)

        if do_run_layer:
            # ************************ tiled input and raster ************************

            assert input_events.shape[0] == self.input_num_tiles_NxN
            assert input_events.shape[1] == self.input_num_tiles_NxN
            assert input_events.shape[2] == self.input_num_rfs_per_tile

            # flatten
            concat_input_events = self._get_input_with_concat(input_events=input_events.flatten(), normalize=self.normalize_concat)

            # reshape
            input_events_by_tile_r_c = concat_input_events.reshape(input_events.shape)
            #print (np.nonzero(input_events_by_tile_r_c)[0])

            tiles_inputs = np.zeros((int(self.num_tiles_NxN * self.num_tiles_NxN), self.tile_input_state_dim), np.float32)
            tiles_inputs_reconstruction = np.zeros((int(self.num_tiles_NxN * self.num_tiles_NxN), self.tile_input_state_dim), np.float32)

            cy_tile_the_input_layer_N(input_events_by_tile_r_c,
                                      self.input_num_tiles_NxN,
                                      self.input_num_rfs_per_tile,
                                      tiles_inputs,
                                      self.tile_dim_NxN,
                                      self.num_tiles_NxN)

            input_states_tiles = tiles_inputs

            # ************************ prepare for iter ************************

            tile_r = (np.arange(self.num_tiles_NxN * self.num_tiles_NxN, dtype=np.int) / self.num_tiles_NxN).astype(np.int)
            tile_c = (np.arange(self.num_tiles_NxN * self.num_tiles_NxN, dtype=np.int) % self.num_tiles_NxN).astype(np.int)

            rf_offset = np.arange(self.num_tiles_NxN * self.num_tiles_NxN) * self.num_rfs
            self.rfs_raster_history[:, self.raster_t] = 0

            sum_rf_norm_dot = 0

            # ************************ do iter ************************

            input_states_tiles_orig = input_states_tiles.copy()

            for k in range(self.num_iters_per_frame):

                assert self.weights.shape[0] == self.num_rfs
                assert self.weights.shape[1] == self.tile_input_state_dim

                assert input_states_tiles.shape[0] == (self.num_tiles_NxN * self.num_tiles_NxN)
                assert input_states_tiles.shape[1] == self.tile_input_state_dim

                #print('******')
                #print(np.nonzero(input_states_tiles)[1], input_states_tiles[np.nonzero(input_states_tiles)])

                rf_norm_dot = self._get_rf_norm_dot(rfs=self.weights, input_states_tiles=input_states_tiles)

                assert rf_norm_dot.shape[0] == (self.num_tiles_NxN * self.num_tiles_NxN)
                assert rf_norm_dot.shape[1] == self.num_rfs

                # find best RF per tile
                best_rf_per_tile = np.argmax(rf_norm_dot, axis=1)
                #print(best_rf_per_tile, np.amax(rf_norm_dot))

                # learn best RF per tile
                if self.do_conditional_lr:
                    per_tile_lr = self.lr * np.square(np.amax(rf_norm_dot, axis=1))
                else:
                    per_tile_lr = np.ones(self.num_tiles_NxN * self.num_tiles_NxN, np.float32) * self.lr

                if do_learn:
                    cy_kmeans_do_learning_per_tile_lr(best_rf_per_tile, self.weights, input_states_tiles, per_tile_lr)

                # take subset of best_rf_per_tile
                rf_norm_dot_best = np.amax(rf_norm_dot, axis=1)

                sum_rf_norm_dot += np.sum(rf_norm_dot_best)

                if self.force_all_frames_wta:
                    # hack: with this and max_rfs_active == 1, it's what we had before (hard 1-wta)
                    good_indices = np.arange(self.num_tiles_NxN * self.num_tiles_NxN)
                else:
                    # if second: ONLY for tiles with rf norm dot over 0.5, set output event and subtract RF from input_state for that tile (<0 -> 0)
                    good_indices = np.nonzero(rf_norm_dot_best > 0.5)[0]

                tiles_inputs_reconstruction[good_indices, :] = tiles_inputs_reconstruction[good_indices, :] + self.weights[best_rf_per_tile[good_indices], :]

                output_events[tile_r[good_indices], tile_c[good_indices], best_rf_per_tile[good_indices]] = 1
                self.rfs_raster_history[(rf_offset + best_rf_per_tile)[good_indices], self.raster_t] = 1

                if self.subtract_remainder:
                    tmp = input_states_tiles[good_indices, :]
                    tmp = tmp - self.weights[best_rf_per_tile[good_indices], :]
                    tmp[np.nonzero(tmp < 0)] = 0

                    input_states_tiles[good_indices, :] = tmp[:]

            # ************************ store history, update times ************************

            mean_rf_norm_dot = sum_rf_norm_dot / (self.num_tiles_NxN * self.num_tiles_NxN * self.num_iters_per_frame)

            self.rf_norm_dot_per_frame[self.t] = mean_rf_norm_dot
            self.mean_rf_norm_dot_per_frame[self.t] = np.mean(self.rf_norm_dot_per_frame[max(0, self.t - self.error_mean_time):self.t])

            self.rf_size_per_frame[self.t] = np.mean(np.sum(self.weights, axis=1))
            self.mean_rf_size_per_frame[self.t] = np.mean(self.rf_size_per_frame[max(0, self.t - self.error_mean_time):self.t])

            # reconstruct error

            #tmp = np.mean(np.abs(tiles_inputs_reconstruction - input_states_tiles), axis=1)  # mean over inputs

            tiles_inputs_reconstruction[np.nonzero(tiles_inputs_reconstruction > 1)] = 1  # TODO ???

            #tmp = np.divide(np.sum(np.abs(tiles_inputs_reconstruction - input_states_tiles_orig), axis=1), np.sum(input_states_tiles_orig, axis=1))
            #tmp = np.sum(np.abs(tiles_inputs_reconstruction - input_states_tiles_orig), axis=1) * 1.0 / input_states_tiles_orig.shape[1]

            tmp = np.sum(input_states_tiles, axis=1) * 1.0 / input_states_tiles.shape[1]

            # if extra_info is not None:
            #     print()
            #     print(extra_info)
            #     print()
            #     print(input_states_tiles)
            #     print()
            #     print(tmp)

            reconstruct_err = np.mean(tmp)  # mean over tiles

            self.reconstruct_err_per_frame[self.t] = reconstruct_err
            self.mean_reconstruct_err_per_frame[self.t] = np.mean(self.reconstruct_err_per_frame[max(0, self.t - self.error_mean_time):self.t])

            self.t += 1

            self.raster_t += 1
            if self.raster_t >= self.raster_steps:
                self.raster_t = 0

        return output_events

    def get_final_errors_dict(self):

        d = {}

        d['reconstruct-err'] = self.mean_reconstruct_err_per_frame[self.t - 1]
        d['rf-norm-dot__mean-by-t'] = self.mean_rf_norm_dot_per_frame[self.t - 1]
        d['mean-sum-rf'] = np.mean(np.sum(self.weights, axis=1))

        return d

    def _get_input_with_concat(self, input_events, normalize=False):
        '''

        :param input_events:
        :param normalize: if > 1 -> set to 1
        :return:
        '''

        self.input_history.store_new_states(newest_states_list=[input_events])
        input_states_seq = self.input_history.get_state_sequence(state_index=0,
                                                                 delay_short=0,
                                                                 delay_long=self.input_concat_timesteps - 1,
                                                                 oldest_first=False)

        sum_input_states = np.sum(input_states_seq, axis=0).astype(np.float32)

        if normalize:
            sum_input_states[sum_input_states > 1] = 1

        return sum_input_states

    def _get_rf_norm_dot(self, rfs, input_states_tiles):

        assert input_states_tiles.shape[0] == (self.num_tiles_NxN * self.num_tiles_NxN)
        assert input_states_tiles.shape[1] == self.tile_input_state_dim

        assert rfs.shape[1] == input_states_tiles.shape[1]

        # why is this one negative? because it is actually a positive value: the larger, the better
        fix_offset = 1e-16

        RF_norm_dot = np.zeros((int(self.num_tiles_NxN * self.num_tiles_NxN), self.num_rfs), np.float32)
        # print('Layer N: rfs:', rfs.shape, ', input_states_tiles:', input_states_tiles.shape)
        cy_compute_error_measures(rfs, input_states_tiles, fix_offset, RF_norm_dot)

        RF_norm_dot = -RF_norm_dot  # undo negative in cython

        return RF_norm_dot




class IterWTABRainLayer0(object):
    def __init__(self, params):

        # ******************************* params *******************************

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
        self.max_time = params['max_time']  # 5000000
        self.do_conditional_lr = params['do_conditional_lr']  # False

        self.skip_zero_inputs = params['skip_zero_inputs']

        # ******************************* input concat + weights *******************************

        if 'input_concat_timesteps' in params:
            self.input_concat_timesteps = params['input_concat_timesteps']
            assert self.input_concat_timesteps == 1, "Layer 0 does not implement multiple time steps"
        else:
            self.input_concat_timesteps = 1  # TODO use

        self.weights = np.random.random((self.num_rfs, self.tile_input_state_dim)) * 1e-12
        self.weights = self.weights.astype(np.float32)

        # ******************************* plotting *******************************

        self.plots_folder = "."

        self.raster_steps = 200
        self.raster_t = 0  # circular; draw vertical line on plot here

        input_state_dim = int(2 * self.input_im_dim_NxN * self.input_im_dim_NxN)
        self.input_state_dim = input_state_dim

        self.input_raster_history = np.zeros((input_state_dim, self.raster_steps), np.uint8)

        self.rfs_raster_history = np.zeros((self.num_rfs * self.num_tiles_NxN * self.num_tiles_NxN, self.raster_steps), np.uint8)

        self.fig_bar = plt.figure(figsize=(40, 20))
        self.ax_bar = self.fig_bar.add_subplot(1, 1, 1)
        self.ax_bar.cla()
        self.ax_bar.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_bar.get_yaxis().get_major_formatter().set_scientific(False)

        self.last_reconstruction_im_info = None

        # ******************************* vars to track *******************************

        self.error_mean_time = 50000

        self.rf_norm_dot_per_frame = np.zeros(self.max_time)
        self.mean_rf_norm_dot_per_frame = np.zeros(self.max_time)  # time average

        self.reconstruct_err_per_frame = np.zeros(self.max_time)
        self.mean_reconstruct_err_per_frame = np.zeros(self.max_time)

        # ******************************* profiling *******************************

        # profiling
        self.loop_time = 0
        self.timed_events = None
        self.timed_event_names = None

        # ******************************* init sim *******************************

        self.t = 0
        self.num_zero_inputs = 0


    def get_input_num_tiles_NxN(self):
        return self.input_im_dim_NxN  # number of pixels

    def get_input_num_rfs_per_tile(self):
        return 1  # per pixel it is one binary event feature, one-dimensional

    def get_tile_dim_NxN(self):
        return self.tile_dim_NxN

    def get_num_tiles_NxN(self):
        return self.num_tiles_NxN

    def get_num_rfs_per_tile(self):
        return self.num_rfs

    def set_plots_folder(self, folder):
        self.plots_folder = folder
        print()
        print('setting plots folder: ', self.plots_folder)
        print()

    def _get_rf_norm_dot(self, rfs, input_states_tiles):

        assert input_states_tiles.shape[0] == (self.num_tiles_NxN * self.num_tiles_NxN)
        assert input_states_tiles.shape[1] == self.tile_input_state_dim

        assert rfs.shape[1] == input_states_tiles.shape[1]

        # why is this one negative? because it is actually a positive value: the larger, the better
        fix_offset = 1e-16

        RF_norm_dot = np.zeros((int(self.num_tiles_NxN * self.num_tiles_NxN), self.num_rfs), np.float32)

        # print('cy_compute_error_measures')
        # print('Layer 0: rfs:', rfs.shape, ', input_states_tiles:', input_states_tiles.shape)
        cy_compute_error_measures(rfs, input_states_tiles, fix_offset, RF_norm_dot)

        RF_norm_dot = -RF_norm_dot  # undo negative in cython

        return RF_norm_dot


    def process_input(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image, do_learn, do_run_layer):


        # ************************ tiled input and raster ************************

        output_events = np.zeros((self.num_tiles_NxN, self.num_tiles_NxN, self.num_rfs), np.float32)

        if self.t >= self.max_time:
            return output_events

        input_state_all = np.concatenate((input_events_p, input_events_n))
        assert input_state_all.dtype == np.float32, str(input_state_all.dtype)

        if input_state_all is None:  # this doesn't appear to ever be able to happen
            return output_events

        if np.count_nonzero(input_state_all) == 0:
            self.num_zero_inputs += 1

            if self.skip_zero_inputs:
                return output_events

        self.input_raster_history[:, self.raster_t] = input_state_all[:]

        # input_state_all not used again this function

        input_states_tiles = np.zeros((int(self.num_tiles_NxN * self.num_tiles_NxN), self.tile_input_state_dim), np.float32)
        tiles_inputs_reconstruction = np.zeros((int(self.num_tiles_NxN * self.num_tiles_NxN), self.tile_input_state_dim), np.float32)

        cy_tile_the_input(input_events_p, input_events_n,
                          event_coords_r, event_coords_c,
                          self.num_tiles_NxN, self.tile_input_state_dim, self.tile_dim_NxN,
                          input_states_tiles)

        assert input_states_tiles.shape[0] == (self.num_tiles_NxN * self.num_tiles_NxN)
        assert input_states_tiles.shape[1] == self.tile_input_state_dim

        # ************************ prepare for iter ************************

        # TODO add in iters later to layer 0 !!!

        self.rfs_raster_history[:, self.raster_t] = 0

        # ************************ do iter ************************
        input_states_tiles_orig = input_states_tiles.copy()

        rf_norm_dot = self._get_rf_norm_dot(rfs=self.weights, input_states_tiles=input_states_tiles)

        assert rf_norm_dot.shape[0] == (self.num_tiles_NxN * self.num_tiles_NxN)
        assert rf_norm_dot.shape[1] == self.num_rfs

        best_rf_per_tile = np.argmax(rf_norm_dot, axis=1)

        if self.do_conditional_lr:
            per_tile_lr = self.lr * np.square(np.amax(rf_norm_dot, axis=1))
        else:
            per_tile_lr = np.ones(self.num_tiles_NxN * self.num_tiles_NxN, np.float32) * self.lr

        if do_learn:
            cy_kmeans_do_learning_per_tile_lr(best_rf_per_tile, self.weights, input_states_tiles, per_tile_lr)

        # store for making an image
        # TODO update for iter !!! if iter > 1

        self.last_reconstruction_im_info = {
            'input_events_p': input_events_p,
            'input_events_n': input_events_n,
            'input_state_all': input_state_all,
            'best_rf_per_tile': best_rf_per_tile,  # for "reconstructed input events" image
            'input_states_tiles': input_states_tiles,  # for "actual input events" image
            'original_input_image': original_input_image  # for "original input" image
        }

        # output_events = np.zeros((self.num_tiles_NxN, self.num_tiles_NxN, self.num_rfs), np.float32)

        tiles_inputs_reconstruction[:, :] = tiles_inputs_reconstruction[:, :] + self.weights[best_rf_per_tile, :]
        tiles_inputs_reconstruction[np.nonzero(tiles_inputs_reconstruction > 1)] = 1  # TODO ???

        tile_r = (np.arange(self.num_tiles_NxN * self.num_tiles_NxN, dtype=np.int) / self.num_tiles_NxN).astype(np.int)
        tile_c = (np.arange(self.num_tiles_NxN * self.num_tiles_NxN, dtype=np.int) % self.num_tiles_NxN).astype(np.int)

        output_events = np.zeros((self.num_tiles_NxN, self.num_tiles_NxN, self.num_rfs), np.float32)
        output_events[tile_r, tile_c, best_rf_per_tile] = 1

        rf_offset = np.arange(self.num_tiles_NxN * self.num_tiles_NxN) * self.num_rfs
        self.rfs_raster_history[(rf_offset + best_rf_per_tile), self.raster_t] = 1

        # ************************ store history, update times ************************

        rf_norm_dot_best = np.amax(rf_norm_dot, axis=1)
        sum_rf_norm_dot = np.sum(rf_norm_dot_best)
        mean_rf_norm_dot = sum_rf_norm_dot / (self.num_tiles_NxN * self.num_tiles_NxN * 1)  # TODO 1 -> num iters

        self.rf_norm_dot_per_frame[self.t] = mean_rf_norm_dot
        self.mean_rf_norm_dot_per_frame[self.t] = np.mean(self.rf_norm_dot_per_frame[max(0, self.t - self.error_mean_time):self.t])

        # reconstruct error

        tmp_sum = np.sum(input_states_tiles_orig, axis=1)

        #tmp = np.divide(np.sum(np.abs(tiles_inputs_reconstruction - input_states_tiles_orig), axis=1), tmp_sum)  # mean over inputs
        #tmp[np.nonzero(tmp_sum==0)] = 0

        #tmp = np.sum(np.abs(tiles_inputs_reconstruction - input_states_tiles_orig), axis=1) * 1.0 / input_states_tiles_orig.shape[1]

        tmp_inp = input_states_tiles - self.weights[best_rf_per_tile, :]
        tmp_inp[tmp_inp < 0] = 0
        tmp = np.sum(tmp_inp, axis=1) * 1.0 / input_states_tiles.shape[1]

        reconstruct_err = np.mean(tmp)  # mean over tiles

        self.reconstruct_err_per_frame[self.t] = reconstruct_err
        self.mean_reconstruct_err_per_frame[self.t] = np.mean(self.reconstruct_err_per_frame[max(0, self.t - self.error_mean_time):self.t])

        self.t += 1

        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0

        return output_events

    def get_final_errors_dict(self):

        d = {}

        d['reconstruct-err'] = self.mean_reconstruct_err_per_frame[self.t - 1]
        d['rf-norm-dot__mean-by-t'] = self.mean_rf_norm_dot_per_frame[self.t - 1]
        d['mean-sum-rf'] = np.mean(np.sum(self.weights, axis=1))

        return d


    def get_table_ims(self):

        if self.timed_events is not None:
            print('profiling:')
            profile_dict = {}
            tmp_thing = self.timed_events / self.loop_time
            asd = 0
            for name in self.timed_event_names:
                profile_dict[name] = tmp_thing[asd]
                asd += 1

            print(profile_dict)
            print()

        ims_list = []
        ims_names_list = []

        rfs_im, rf_ims_dict = make_im(self.weights, num_bins_per_pixel=1,
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

        if self.last_reconstruction_im_info is None:
            return ims_list, ims_names_list

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

        input_states_tiles = self.last_reconstruction_im_info['input_states_tiles']

        tile_inputs_im, tile_inputs_im_dict = make_im(input_states_tiles, num_bins_per_pixel=1,
                                               input_im_dim=self.tile_dim_NxN,
                                               im_final_dim=int(200 * 3000 / 1600),  # /800 for two-im per rf display
                                               mod_for_disp=self.num_tiles_NxN,
                                               normalize_weights=True,
                                               borders_px=1)  # 1

        ims_list.append(tile_inputs_im)
        ims_names_list.append('tile_inputs_im')

        # add a full reconstruction event im over all tiles, side by side with original event im and original im
        #   need pre processor to also pass back original im for reference

        best_rf_per_tile = self.last_reconstruction_im_info['best_rf_per_tile']
        #print(best_rf_per_tile.shape)  # (256,)
        #print(input_states_tiles.shape)  # (256, 128)

        rec = self.weights[best_rf_per_tile, :]
        #print(rec.shape)  # (256, 128)

        rec_im, _ = make_im(rec, num_bins_per_pixel=1,
                                 input_im_dim=self.tile_dim_NxN,
                                 im_final_dim=int(200 * 3000 / 1600),  # /800 for two-im per rf display
                                 mod_for_disp=self.num_tiles_NxN,
                                 normalize_weights=True,
                                 borders_px=1)  # 1

        ims_list.append(rec_im)
        ims_names_list.append('rec_im')

        return ims_list, ims_names_list

        # plot error over time (including over iterations within batches)
        # show RFs

    def do_plots(self, extra_info=''):

        max_tiles_to_plot = 16  # so we don't plot a million things

        #print('STARTING PLOTS')
        self.ax_bar.cla()
        num_rf = self.rfs_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.rfs_raster_history, np.arange(num_rf)[:, np.newaxis]))

        t = np.arange(raster_plot.shape[0])

        self.ax_bar.plot(t, raster_plot[:, 0:min(max_tiles_to_plot * self.num_rfs, num_rf)], color='b', marker='.', linestyle='')
        self.ax_bar.axvline(x=self.raster_t, color='g')

        plot_y = 0
        for k in range(min(max_tiles_to_plot, self.num_tiles_NxN * self.num_tiles_NxN)):
            self.ax_bar.axhline(y=plot_y, color='r')
            plot_y += self.num_rfs

        if len(extra_info) > 0:
            self.fig_bar.savefig(self.plots_folder + "/raster_rfs_" + extra_info + ".png", dpi=100)
        else:
            self.fig_bar.savefig(self.plots_folder + "/raster_rfs.png", dpi=100)

        self.ax_bar.cla()
        self.ax_bar.plot(self.mean_rf_norm_dot_per_frame[0:self.t], color='k', marker='.')

        if len(extra_info) > 0:
            self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_rf_norm_dot_" + extra_info + ".png", dpi=100)
        else:
            self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_rf_norm_dot.png", dpi=100)

        self.ax_bar.cla()
        raster_plot = np.transpose(np.multiply(self.input_raster_history, np.arange(self.input_state_dim)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.ax_bar.axvline(x=self.raster_t, color='g')
        self.fig_bar.savefig(self.plots_folder + "/raster_inputs.png", dpi=100)

        if len(extra_info) > 0:
            self.fig_bar.savefig(self.plots_folder + "/raster_inputs_" + extra_info + ".png", dpi=100)
        else:
            self.fig_bar.savefig(self.plots_folder + "/raster_inputs.png", dpi=100)

        self.ax_bar.cla()
        self.ax_bar.plot(self.mean_reconstruct_err_per_frame[0:self.t], color='k', marker='.')
        self.ax_bar.ticklabel_format(useOffset=False)
        self.ax_bar.get_xaxis().get_major_formatter().set_useOffset(False)
        self.ax_bar.get_yaxis().get_major_formatter().set_useOffset(False)

        if len(extra_info) > 0:
            self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_reconstruct_err_" + extra_info + ".png", dpi=100)
        else:
            self.fig_bar.savefig(self.plots_folder + "/time_averaged_mean_reconstruct_err.png", dpi=100)

    def do_plots_OLD(self):

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
