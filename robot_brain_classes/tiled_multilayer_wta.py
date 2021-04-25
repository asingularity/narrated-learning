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

        self.err_reduce_lr = params['err_reduce_lr']

        self.rf_queue = np.zeros((self.queue_length, self.input_dim))
        self.circ_index_oldest = 0  # will increment once before first used
        self.circ_index_newest = -1  # will increment once before first used

        self.mean_error_reduce = np.zeros(self.queue_length)

        self.t = 0

    def step(self, new_input, rfs, best_rf_error, select_criterion):
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

        self._update_error_reduction(new_input=new_input, best_rf_select_criterion=select_criterion, best_rf_error=best_rf_error)  # get all rfs from queue with error lower than best_rf_error and they would be hypothetical winners
        self._push_onto_queue(new_input=new_input)
        pop_rf, pop_rf_mean_err_reduce = self._pop_oldest_candidate()

        self.t += 1

        return pop_rf, pop_rf_mean_err_reduce

    def _update_error_reduction(self, new_input, best_rf_select_criterion, best_rf_error):
        if self.kmeans_dist_metric == 0:
            eff_frame = _compute_match(rfs=self.rf_queue, input_arr=new_input)
            win_candidates = np.nonzero(eff_frame > best_rf_select_criterion)[0]
            err_frame = _compute_error(rfs=self.rf_queue, input_arr=new_input)

            debug_print_here = False
            if debug_print_here:
                print()
                print('****************')
                print(self.t)
                print(len(win_candidates))
                print(np.amin(eff_frame), np.amax(eff_frame))
                print(np.sum(np.sum(self.rf_queue)))
                print()

        elif self.kmeans_dist_metric == 1:
            assert best_rf_error == best_rf_select_criterion

            err_frame = _compute_error(rfs=self.rf_queue, input_arr=new_input)
            win_candidates = np.nonzero(err_frame < best_rf_error)
        else:
            assert False, 'Invalid kmeans distance metric! can only be 1 or 0. ' + str(self.kmeans_dist_metric)

        # TODO parameter; has to match the one in other class; refactor!
        lr_error_reduce = self.err_reduce_lr  # this is just a guess, on same order as queue len

        mean_err_before = self.mean_error_reduce[win_candidates]
        # all rfs except winner adapt as if they did not reduce error at all (since we care about integral):
        self.mean_error_reduce = (1.0 - lr_error_reduce) * self.mean_error_reduce + lr_error_reduce * 0.0
        # winner has reduced error
        self.mean_error_reduce[win_candidates] = (1.0 - lr_error_reduce) * mean_err_before + lr_error_reduce * (best_rf_error - err_frame[win_candidates])

    def _push_onto_queue(self, new_input):

        self.circ_index_newest += 1
        if self.circ_index_newest == self.queue_length:
            self.circ_index_newest = 0

        self.circ_index_oldest += 1
        if self.circ_index_oldest == self.queue_length:
            self.circ_index_oldest = 0

        debug_print_here = False
        if debug_print_here:
            print()
            print('pushing onto queue...')
            print('    ', 'self.circ_index_newest:', self.circ_index_newest)
            print('    ', 'self.circ_index_oldest:', self.circ_index_oldest)
            print()

        self.rf_queue[self.circ_index_newest, :] = new_input[:]

    def _pop_oldest_candidate(self):
        pop_rf = None
        pop_rf_mean_err_reduce = None

        if self.t < self.queue_length:
            return pop_rf, pop_rf_mean_err_reduce

        pop_rf = self.rf_queue[self.circ_index_oldest, :]
        pop_rf_mean_err_reduce = self.mean_error_reduce[self.circ_index_oldest]

        self.mean_error_reduce[self.circ_index_oldest] = 0

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
        self.kmeans_cq_length = 4000  # 4000
        self.kmeans_err_reduce_lr = 0.1 / self.kmeans_cq_length
        self.kmeans_queue_err_reduce_lr = 1.0 / self.kmeans_cq_length
        self.cq_on = False
        self.kmeans_enable_adaptation = True  # TODO should be false at first if cq_on is True!

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
            'queue_length': self.kmeans_cq_length,
            'input_dim': self.input_state_dim,
            'kmeans_dist_metric': self.kmeans_dist_metric,
            'err_reduce_lr': self.kmeans_queue_err_reduce_lr
        })

        self.mean_error_reduce = np.zeros(self.num_rfs)
        self.rf_ages = np.zeros(self.num_rfs)
        self.rfs_init_already = 0

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

        self.fig_bar2 = plt.figure(figsize=(20, 20))
        self.ax_bar2 = self.fig_bar2.add_subplot(1, 1, 1)
        self.ax_bar2.cla()
        self.ax_bar2.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_bar2.get_yaxis().get_major_formatter().set_scientific(False)

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
        select_criterion = None

        if self.kmeans_dist_metric == 0:
            # this one gets a less skewed histogram. normalization by weight required:

            eff_frame = _compute_match(rfs=self.weights, input_arr=state_to_learn)

            best_rf = np.argmax(eff_frame)
            select_criterion = eff_frame[best_rf]
            error = _compute_error(rfs=self.weights[best_rf, :], input_arr=state_to_learn, single_rf=True)

            # get next best
            eff_frame[best_rf] = -np.inf
            next_best_rf = np.argmax(eff_frame)
            next_error = _compute_error(rfs=self.weights[next_best_rf, :], input_arr=state_to_learn, single_rf=True)

        elif self.kmeans_dist_metric == 1:
            # this one gets a skewed histogram. normalization by weight would make only one winner all the time:

            err_frame = _compute_error(rfs=self.weights, input_arr=state_to_learn)

            best_rf = np.argmin(err_frame)
            error = err_frame[best_rf]
            select_criterion = error
            # get next best
            err_frame[best_rf] = np.inf
            next_best_rf = np.argmin(err_frame)
            next_error = err_frame[next_best_rf]

        else:
            best_rf = None
            error = None

            next_best_rf = None
            next_error = None

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
            # TODO should this be true?
            #assert self.kmeans_enable_adaptation is False, 'cannot be enabled together for now! in sequence is ok'

            if self.rfs_init_already < self.num_rfs:
                # just assign newest one
                self.weights[self.rfs_init_already, :] = state_to_learn[:]
                self.rfs_init_already += 1
            else:
                # update all RF error reduces
                # need:
                #   best_rf, error
                #   next_best_rf, next_error

                lr_error_reduce = self.kmeans_err_reduce_lr  # this is just a guess, on same order as queue len
                # this may not follow unless using distance metric 1:
                debug_print_here_0 = (self.kmeans_dist_metric == 1)
                if debug_print_here_0:
                    if error > next_error:
                        print()
                        print("!!! error of winner should always be less than error of next best; otherwise? weird?" + " error: " + str(error) + " next error: " + str(next_error))
                        print()
                    else:
                        pass
                        #print()
                        #print('ALL GOOD')
                        #print()

                mean_err_before = self.mean_error_reduce[best_rf]
                # all rfs except winner adapt as if they did not reduce error at all (since we care about integral):
                self.mean_error_reduce = (1.0 - lr_error_reduce) * self.mean_error_reduce + lr_error_reduce * 0.0
                # winner has reduced error
                debug_print_here = False
                if debug_print_here:
                    print()
                    print('    setting mean error reduce best rf:')
                    print('    lr_error_reduce:', lr_error_reduce)
                    print('    mean_err_before:', mean_err_before)
                    print('    error, rf:', error, best_rf)
                    print('    next_error, rf:', next_error, next_best_rf)
                    print('    (next_error - error):', (next_error - error))
                    print()
                self.mean_error_reduce[best_rf] = (1.0 - lr_error_reduce) * mean_err_before + lr_error_reduce * (next_error - error)
                self.rf_ages = self.rf_ages + 1

                # get worst rf:
                #   only qualify if lifetime of rf (since last replacement) is greater than threshold (maybe self.kmeans_cq_length?)
                #   get worst_rf_index, worst_rf_mean_err_reduce
                qualified_rfs = np.nonzero(self.rf_ages > self.kmeans_cq_length)[0]

                pop_rf, pop_rf_mean_err_reduce = self.cq.step(new_input=state_to_learn, rfs=self.weights, best_rf_error=error, select_criterion=select_criterion)

                if pop_rf is not None and len(qualified_rfs) > 0:

                    worst_rf_index = qualified_rfs[np.argmin(self.mean_error_reduce[qualified_rfs])]
                    worst_rf_mean_err_reduce = self.mean_error_reduce[worst_rf_index]

                    if pop_rf_mean_err_reduce > worst_rf_mean_err_reduce:
                        self.num_resets += 1
                        # overwrite worst RF with popped RF
                        self.weights[worst_rf_index, :] = pop_rf[:]
                        self.mean_error_reduce[worst_rf_index] = 0.0
                        self.rf_ages[worst_rf_index] = 0

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
        print('    num resets per frame: ', self.num_resets / self.num_frames_disp_resets, self.num_resets, self.num_frames_disp_resets)
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

        self.ax_bar2.cla()
        self.ax_bar2.bar(np.arange(self.num_rfs), self.mean_error_reduce)
        self.fig_bar2.savefig(self.plots_folder + '/mean_err_reduce.png', dpi=100)
        print('egadasd', np.amin(self.mean_error_reduce), np.amax(self.mean_error_reduce))


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

