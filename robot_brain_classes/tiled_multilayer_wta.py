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


def _compute_match(rfs, input_arr):

    tmp_3_cpu = np.sum(rfs, axis=1)
    tmp_1 = np.multiply(rfs, input_arr)
    tmp_2_cpu = np.sum(tmp_1, axis=1)
    eff_frame = np.divide(tmp_2_cpu, tmp_3_cpu)

    return eff_frame


def _compute_error(rfs, input_arr, single_rf=False):

    if single_rf:
        err_frame = np.sum(np.abs(rfs - input_arr))
    else:
        err_frame = np.sum(np.abs(rfs - input_arr), axis=1)

    return err_frame


class CandidateQueue(object):
    def __init__(self, params):

        self.queue_length = params['queue_length']
        self.input_dim = params['input_dim']

        # what to use for determining WTA winner? _error (norm) min or _match max?
        self.kmeans_dist_metric = params['kmeans_dist_metric'] #  0: normalized match, 1: norm, as in seq-knn

    def step(self, new_input, rfs):
        '''
            # pop_rf, pop_rf_mean_err_reduce = self.cq.step(new_input=state_to_learn, rfs=self.weights)
            # push new rf onto queue; test all candidates on current input vs. rfs default winner; pop end of queue RF
            # returns popped rf, and its mean error reduction per frame (integrated over time; NOT per its activation)


            (1) push new_input onto queue
                - implemented with circular buffer

            (2) update total err reduce per candidate

            (3) pop oldest candidate and its mean err reduction per frame

                mean err reduction for a candidate:
                    say the queue length is 10000
                    candidate would've won the WTA against the best RF in the kmeans rfs 432 times
                    mean err reduction = sum of differences between (candidate-error vs. best-rf-error) for all times it would've won,
                                         divided by total time steps (queue len) 10000

        '''



        return pop_rf, pop_rf_mean_err_reduce

class SeqNNSeqKMeansBrain(object):
    '''
        first seq-nn
        then seq-kmeans with seq-nn as init
            (try vs. zero init as well)
            (seq-kmeans: try forgetting vs. non-forgetting)

        this will become the basis of the single layer
    '''

    def __init__(self, params):

        # params

        self.input_im_dim = params['input_im_dim']

        self.max_time = 800000
        self.skip_start_frames = 30 * 30
        self.init_kmeans_with_seq_nn = False  # if False, inits kmeans table at self.seq_nn_learn_time with just zeros
        self.seq_nn_learn_time = 0
        self.input_concat_timesteps = 1
        self.display_im_dim = 3000
        self.num_rfs = 800
        self.lr = 0.01  # 0.1, 0.001
        self.kmeans_dist_metric = 0  #  0: normalized match, 1: norm, as in seq-knn
        self.forgetful_kmeans = False
        self.kmeans_enable_adaptation = True

        self.cq_on = False

        self.enable_reset_rfs = False
        self.reset_rf_time = 40000

        # for plotting:
        self.error_mean_time = 40000

        # init
        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

        self.seq_nn_t_start = self.skip_start_frames
        self.seq_nn_t_end = self.seq_nn_t_start + self.seq_nn_learn_time

        self.seq_kmeans_t_start = self.seq_nn_t_end + 1
        self.seq_kmeans_t_end = np.inf

        self.last_input_im = None
        self.plots_folder = "."

        self.input_history = StatesLimitedHistory(params={'max_delay': self.input_concat_timesteps,
                                                          'states_dim_list': [self.input_state_dim],
                                                          'store_extra_data': False})

        self.rfs_raster_history = np.zeros((self.num_rfs, self.max_time), np.uint8)

        self.last_layer_input = None

        self.t = 0

        self._init_seq_nn()
        self._init_seq_kmeans()

        self._init_plotting()

        self.mean_error = np.zeros(self.max_time)
        self.error = np.zeros(self.max_time)

        self._init_candidate_queue()

    def _init_candidate_queue(self):
        self.cq = CandidateQueue(params={
            'queue_length': 4000,
            'input_dim': self.input_state_dim,
            'kmeans_dist_metric': self.kmeans_dist_metric
        })

    def _init_plotting(self):

        self.fig_1 = plt.figure(figsize=(30, 20))
        self.ax_1 = self.fig_1.add_subplot(2, 1, 1)
        self.ax_1.cla()
        self.ax_1.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_1.get_yaxis().get_major_formatter().set_scientific(False)

        #self.fig_bar = plt.figure(figsize=(40, 20))
        self.ax_bar = self.fig_1.add_subplot(2, 1, 2)
        self.ax_bar.cla()
        self.ax_bar.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_bar.get_yaxis().get_major_formatter().set_scientific(False)

        self.do_raster_plots_every_k_im = 4
        self.ims_since_raster = 0

    def _init_seq_nn(self):
        self.cuda_table = CudaTable(num_entries=self.num_rfs,
                                    input_dim=self.input_state_dim)
        self.init_I_row_num = None

        self.num_row_changes_for_disp = 0
        self.frames_since_row_change_disp = 0

    def _init_seq_kmeans(self):
        self.weights = np.zeros((self.num_rfs, self.input_state_dim))
        self.last_event_time = np.zeros(self.num_rfs)
        self.rf_counts = np.zeros(self.num_rfs)
        self.num_resets = 0
        self.num_frames_disp_resets = 0

    def set_plots_folder(self, folder):
        self.plots_folder = folder
        print()
        print('setting plots folder: ', self.plots_folder)
        print()

    def process_input(self, input_im):

        if self.t >= self.max_time:
            return

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

                return

        self.input_history.store_new_states(newest_states_list=[input_state])
        input_states_seq = self.input_history.get_state_sequence(state_index=0,
                                                                 delay_short=0,
                                                                 delay_long=self.input_concat_timesteps - 1,
                                                                 oldest_first=False)

        sum_input_states = np.sum(input_states_seq, axis=0).astype(np.float32)
        # print()
        # print('input_state.shape', input_state.shape)
        # print('input_states_seq.shape', input_states_seq.shape)  # (10, 864)
        # print('sum_input_states.shape:', sum_input_states.shape)
        # print()
        # exit(1)

        best_rf = None
        error = None

        if self.t < self.seq_nn_t_start:
            best_rf = 0
            error = 0
        elif self.seq_nn_t_start <= self.t < self.seq_nn_t_end:
            if self.init_kmeans_with_seq_nn:
                best_rf, error = self._step_seq_nn(input_state=sum_input_states)
            else:
                best_rf = 0
                error = 0
        elif self.t == self.seq_nn_t_end:
            print()
            print('*** Finished Seq-NN, starting seq-kmeans! ***')
            print()
            self._transition_nn_to_kmeans()
            best_rf = 0
            error = 0
        elif self.seq_kmeans_t_start <= self.t <= self.seq_kmeans_t_end:
            best_rf, error = self._step_seq_kmeans(input_state=sum_input_states)

        assert error is not None
        # TODO error, mean error
        self.error[self.t] = error
        self.mean_error[self.t] = np.mean(self.error[max(0, self.t-self.error_mean_time):self.t])

        self.rfs_raster_history[best_rf, self.t] = 1

        self.last_layer_input = layer_input.copy()
        self.t += 1

    def _step_seq_nn(self, input_state):

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

        return new_min_ind, error

    def _step_seq_kmeans(self, input_state):
        state_to_learn = input_state.copy()
        error = None

        if self.kmeans_dist_metric == 0:
            # this one gets a less skewed histogram. normalization by weight required:

            eff_frame = _compute_match(rfs=self.weights, input_arr=state_to_learn)

            best_rf = np.argmax(eff_frame)
            error = _compute_error(rfs=self.weights[best_rf, :], input_arr=state_to_learn, single_rf=True)

        elif self.kmeans_dist_metric == 1:
            # this one gets a skewed histogram. normalization by weight would make only one winner all the time:

            err_frame = _compute_error(rfs=self.weights, input_arr=state_to_learn)

            best_rf = np.argmin(err_frame)
            error = err_frame[best_rf]
        else:
            error = None
            assert False, 'unrecognized self.kmeans_dist_metric: ' + str(self.kmeans_dist_metric)

        if self.kmeans_enable_adaptation:
            forgetful = self.forgetful_kmeans  # False
            if forgetful:
                lr = self.lr
                self.weights[best_rf, :] = lr * state_to_learn + (1.0 - lr) * self.weights[best_rf, :]
            else:
                self.rf_counts[best_rf] += 1
                self.weights[best_rf, :] = self.weights[best_rf, :] + (1.0 / self.rf_counts[best_rf]) * (state_to_learn - self.weights[best_rf, :])
                #self.weights[delete_from::, :] = 0.0

        if self.cq_on:
            assert self.kmeans_enable_adaptation is False, 'cannot be enabled together for now! in sequence is ok'

            if self.t < self.num_rfs:
                # just assign newest one
                self.weights[self.t, :] = state_to_learn[:]
            else:

                # this is probably? not necessary:
                # assert self.kmeans_dist_metric == 0, 'below assumes match: higher the better; implement other option!'

                # push new rf onto queue; test all candidates on current input vs. rfs default winner; pop end of queue RF
                # returns popped rf, and its mean error reduction per frame (integrated over time; NOT per its activation)

                pop_rf, pop_rf_mean_err_reduce = self.cq.step(new_input=state_to_learn, rfs=self.weights)

                if pop_rf is not None:
                    if pop_rf_mean_err_reduce > worst_rf_mean_err_reduce:

                        # overwrite worst RF with popped RF
                        self.weights[worst_rf_index, :] = pop_rf[:]

                        # TODO reset mean err reduce for this RF


        # activity constraint
        if self.enable_reset_rfs:
            self.last_event_time[best_rf] = self.t
            reset_rfs = np.nonzero(self.t - self.last_event_time > self.reset_rf_time)[0]
            self.weights[reset_rfs, :] = 0.0
            self.num_resets += len(reset_rfs)

        self.num_frames_disp_resets += 1

        assert error is not None

        return best_rf, error

    def _transition_nn_to_kmeans(self):

        if self.init_kmeans_with_seq_nn:
            pass
            # use cuda table to init rows for further refinement via kmeans
            self.weights[:, :] = self.cuda_table.table_i[:, :]
        else:
            pass
            # init with zeros

    def get_table_ims(self):

        if self.do_raster_plots_every_k_im is not None:
            self.ims_since_raster += 1
            if self.ims_since_raster > self.do_raster_plots_every_k_im:
                self.do_plots()
                self.ims_since_raster = 0

        #print('    mean frames between row replaces: ', self.frames_since_row_change_disp / (self.num_row_changes_for_disp + 1e-9))
        print('    num resets per frame: ', self.num_resets / self.num_frames_disp_resets)
        self.num_frames_disp_resets = 0
        self.num_resets = 0

        #self.num_row_changes_for_disp = 0
        #self.frames_since_row_change_disp = 0

        ims_list = []
        ims_names_list = []

        # show:
        #   cuda table (stops changing after transition)
        #   kmeans centers (zeros until transition)


        # cuda
        cuda_im, cuda_ims_dict = make_im(self.cuda_table.table_i.copy(), num_bins_per_pixel=1,
                                                                         input_im_dim=self.input_im_dim,
                                                                         im_final_dim=int(self.num_rfs * 3000 / 1600),  # /800 for two-im per rf display
                                                                         mod_for_disp=20,
                                                                         normalize_weights=True)

        ims_list.append(cuda_im)
        ims_names_list.append('cuda_im')

        # kmeans
        rfs_im, rf_ims_dict = make_im(self.weights, num_bins_per_pixel=1,
                                                    input_im_dim=self.input_im_dim,
                                                    im_final_dim=int(self.num_rfs * 3000 / 1600),  # /800 for two-im per rf display
                                                    mod_for_disp=20,
                                                    normalize_weights=True)

        ims_list.append(rfs_im)
        ims_names_list.append('rfs_im')

        return ims_list, ims_names_list

    def do_plots(self):

        # activity histogram
        self.ax_bar.cla()
        self.ax_bar.bar(np.arange(self.num_rfs), np.sum(self.rfs_raster_history[:, max(0, self.t - 20000):self.t], axis=1))
        #self.fig_bar.savefig(self.plots_folder + "/rfs_activity.png", dpi=100)

        self.ax_1.cla()
        num_rf = self.rfs_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.rfs_raster_history[0:num_rf, max(0, self.t - 200):self.t], np.arange(num_rf)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax_1.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.fig_1.savefig(self.plots_folder + "/raster_rfs.png", dpi=100)

        # plot error
        self.ax_bar.cla()
        self.ax_1.cla()
        self.ax_1.plot(self.mean_error[0:self.t], color='k', marker='.')
        self.fig_1.savefig(self.plots_folder + "/mean_error_vs_t.png", dpi=100)


class TiledMultilayerWTABrain(object):
    def __init__(self, params):
        '''

        :param params:
        '''

    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

    def get_table_ims(self):
        ims_list = []
        ims_names_list = []

        return ims_list, ims_names_list


class SingleTiledLayer(object):
    def __init__(self, params):
        '''

        :param params:
        '''

    def step(self, layer_input, change_to_diff_image):
        pass

