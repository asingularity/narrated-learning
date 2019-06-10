
from matrix_vector_dist_parallel import knn_query as knn_parallel_query
from cuda_dist_query import CudaTable
import time
import cv2
import pickle
import random
from math import sqrt

import numpy as np
np.set_printoptions(suppress=True)
#from PVM.PVM_framework import MLP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pycuda.driver as cuda
import pycuda.autoinit
import skcuda
import skcuda.misc as misc
import pycuda.gpuarray as gpuarray
from brain_components_classes.states_history import StatesLimitedHistory


class ConfidencePredictorEnsemble(object):
    def __init__(self, params):
        print( 'initializing ensemble...')

        self.do_random_init = params['do_random_init']  # False  # initialize with random (first) entries
        self.do_adaptation = params['do_adaptation']  # True  # WTA-based learning
        self.do_replacements = params['do_replacements']  # True  # replace low effectiveness over time

        self.max_history_length = params['max_history_length']
        self.max_delay = params['max_delay']

        self.replacement_every_k_steps = params['replacement_every_k_steps']  # 100 for 800 rows, 10 for 8000 rows

        self.use_context_in_knn_diff = True

        self.plots_prefix = params['plots_prefix']
        self.env_width_height = params['env_width_height']

        self.entries_per_layer = params['entries_per_layer']
        self.num_layers = len(self.entries_per_layer)
        self.dim = params['dim']  # input dim

        input_output_dt = 2  # predict time
        input_context_dt = 0  # context future time - NOT USED RIGHT NOW

        # input (t), prediction (t+1), context (t+2)
        self.post_init_done = False

        self.last_warn_dist_time = time.time() - 10
        self.warn_dist_int = 0.1

        self.cuda_tables_list = []
        self.error_histories_list = []
        self.mean_error_histories_list = []
        self.error_steps_list = []
        self.table_use_hist_list = []
        self.effectiveness_sum_list = []
        self.effectiveness_num_list = []
        self.row_ages_list = []
        self.last_replacement_t_list = []
        self.input_output_dt_steps_list = []
        self.input_context_dt_steps_list = []
        self.last_step_layer_dists = []
        self.tmp_ind_list = []

        self.debug_x_y_theta_output = []
        self.debug_x_y_theta_input = []

        total_gb = 0

        # logging:
        self.replacements_by_layer = []
        self.disp_every_k_sec = 10
        self.last_disp_time = time.time()

        layer_input_dim_list = []  # for initializing states_history object for input
        k_end = None
        for k, layer_entries in enumerate(self.entries_per_layer):
            if k == 0:
                layer_input_dim = self.dim
            else:
                # input dim of layer is number of rows of previous layer
                layer_input_dim = self.entries_per_layer[k - 1]

            if k < self.num_layers - 1:
                # context dim of layer is number of rows of next layer
                layer_context_dim = self.entries_per_layer[k + 1]
            else:
                # last layer has no context
                layer_context_dim = 0

            if k < self.num_layers - 1:
                include_layers = ['ioc', 'ic_only']  # ioc: learning, ic: running
            else:
                include_layers = ['io_only', 'i_only']  # last layer: io: learning, i: running

            layer_output_dim = layer_input_dim
            self.cuda_tables_list.append(CudaTable(num_entries=layer_entries,
                                                   input_dim=layer_input_dim,
                                                   output_dim=layer_output_dim,
                                                   context_dim=layer_context_dim,
                                                   include_layers=include_layers))
            layer_size_gb = self.cuda_tables_list[len(self.cuda_tables_list) - 1].get_size_gb()
            print( 'Init of layer', k, 'with rows X cols, size_GB,', '(', layer_entries, 'X', ('('+str(layer_input_dim) + ' + ' + str(layer_context_dim) + ' + ' + str(layer_output_dim)+')'), layer_size_gb)

            total_gb += layer_size_gb

            self.replacements_by_layer.append({})

            self.error_histories_list.append(np.zeros(self.max_history_length))
            self.mean_error_histories_list.append(np.zeros(self.max_history_length))
            self.error_steps_list.append(0)

            self.debug_x_y_theta_output.append(np.zeros((layer_entries, 3)))
            self.debug_x_y_theta_input.append(np.zeros((layer_entries, 3)))

            self.table_use_hist_list.append(np.zeros(layer_entries))
            self.effectiveness_sum_list.append(np.zeros(layer_entries))
            self.effectiveness_num_list.append(np.ones(layer_entries))
            self.row_ages_list.append(np.zeros(layer_entries))
            self.last_replacement_t_list.append(0)

            self.input_output_dt_steps_list.append(input_output_dt)
            self.input_context_dt_steps_list.append(input_context_dt)

            self.last_step_layer_dists.append(None)
            self.tmp_ind_list.append(0)

            layer_input_dim_list.append(layer_input_dim)
            k_end = k

        # last layer output would be a nonexisting next layer's input dim
        layer_input_dim_list.append(self.entries_per_layer[k_end])

        print ('finished initializing ensemble. total gb: ', total_gb)

        self.layer_input_history = StatesLimitedHistory(params={'max_delay': self.max_delay,
                                                                'states_dim_list': layer_input_dim_list})

        self.plots_save_folder = params['plots_save_folder']
        self.error_average_steps = params['error_average_steps']

        self.t = 0

    def _scale_dists(self, dists, min_d, max_d):
        '''
            such that linear for:
            min distance pair -> output confidence = 1.0
            max distance pair -> output confidence = 0.0
            hard nonlinearity otherwise (maxed out to 0 or 1)
        '''

        dists_copy = dists.copy()
        dists_copy = np.argsort(dists_copy)
        dists_copy = (dists_copy * 1.0 / len(dists_copy)).astype(dists.dtype)

        return dists_copy

    def step(self, input_state, x_y_theta, learn=True):
        '''
        for learning mode

        :param input_state:
        :param x_y_theta:
        :param learn:
        :return:
        '''
        self.t += 1

        # for each layer:
        # run forward on newest input
        # train on previous

        layer_input = input_state.copy()
        dists = None

        new_states_list = [layer_input]
        extra_data_list = [x_y_theta]
        for k in range(self.num_layers):

            if k == self.num_layers - 1:
                layer_context = None
            else:
                layer_context = self.last_step_layer_dists[k + 1]
                if layer_context is None:
                    layer_context = np.zeros(self.cuda_tables_list[k].context_dim, np.float32)

            dists = self.cuda_tables_list[k].query(query_input=layer_input,
                                                   query_output=None,
                                                   query_context=layer_context)

            ind = np.argmin(dists)
            self.table_use_hist_list[k][ind] += 1
            self.row_ages_list[k][:] = self.row_ages_list[k][:] + 1

            min_d, _, _ = self.cuda_tables_list[k].get_min_dist()  # TODO base on input + context
            max_d, _, _ = self.cuda_tables_list[k].get_max_dist()  # TODO base on input + context

            scaled_ic_dists = self._scale_dists(dists, min_d, max_d)

            # *** how scaling works ***
            # in normal network operations, all lookups are based only on input + context
            #   (prediction output is "implicit" and used only in learning)
            # so, we scale dists to Input + Context only to be [0, 1], disregarding output column
            # all I, O, C vectors, thus, are I+C-scaled vectors from other layers
            # when dist is used in learning phase, on I+O+C, there is no scaling done on that dist afterwards
            #   (as it is not propagated anywhere)

            self.last_step_layer_dists[k] = scaled_ic_dists.copy()
            layer_input = scaled_ic_dists.copy()

            #if k < self.num_layers - 1:
            new_states_list.append(layer_input)
            extra_data_list.append(x_y_theta)

        # processing stores *scaled* dists
        self.layer_input_history.process_new_states(newest_states_list=new_states_list, extra_data_list=extra_data_list)

        # learning
        # should only happen if guaranteed enough history already

        for k in range(self.num_layers):
            dt_input_output = self.input_output_dt_steps_list[k]  # 2

            train_input, x_y_theta_input = self.layer_input_history.get_state(state_index=k, delay=dt_input_output)
            train_output, x_y_theta_output = self.layer_input_history.get_state(state_index=k, delay=0)

            # here, train_input and train_output are I+C dists that have been scaled

            if k == self.num_layers - 1:
                train_context = None
            else:
                # train_context = self.layer_dists_history[k + 1][context_delay]
                # since context_delay == 0, instead of above, we can use last step dists:

                # why k + 2? because:
                #   layer context (k) == layer output (k + 1) == layer input (k + 2)
                train_context, x_y_theta_context = self.layer_input_history.get_state(state_index=k + 2, delay=dt_input_output)

                # assert that with zero delay from layer_input_history, same vector as above

            # here, train_input, train_output, and train_context are all scaled I+C dists, from different layers
            # (from feedforward sweep)

            self._learn_seq_nn(k=k,
                               cuda_table=self.cuda_tables_list[k],
                               train_input=train_input,
                               train_output=train_output,
                               train_context=train_context,
                               error_history=self.error_histories_list[k],
                               mean_error_history=self.mean_error_histories_list[k],
                               x_y_theta_input=x_y_theta_input,
                               x_y_theta_output=x_y_theta_output
                              )

        to_debug_print = {}
        return to_debug_print

    def _learn_seq_nn(self, k, cuda_table, train_input, train_output, train_context, error_history, mean_error_history, x_y_theta_input, x_y_theta_output):
        '''
        :param k: layer
        :param cuda_table: self.cuda_tables_list[k]
        :param train_input: I+C scaled dists
        :param train_output: I+C scaled dists
        :param train_context: I+C scaled dists
        :param error_history: self.error_histories_list[k]
        :param mean_error_history: self.mean_error_histories_list[k]
        :return:
        '''

        # sequential nn: replace "worst" pair in table (smallest distance), with new best pair, new best pair better than worst pair
        #   "worst" pair: smallest distance between rows in pair
        #   "better": larger distance between rows in pair

        do_random_init = self.do_random_init  # initialize with random entries

        # (1) get distance of new row to all current rows in table

        dists = cuda_table.query(query_input=train_input,
                                 query_output=train_output,
                                 query_context=train_context)

        # should we be scaling dists here?
        # no- this is only for the purpose of determining which rows to update or replace
        #   so, relative dist is all that matters, so scaling should not matter for learning.

        sorted_dist_indices = np.argsort(dists)
        new_min_ind = sorted_dist_indices[0]
        new_min_dist = dists[new_min_ind]
        ind2 = sorted_dist_indices[1]
        dist2 = dists[ind2]

        dist = new_min_dist
        error_step = self.error_steps_list[k]
        error_history[error_step] = dist
        mean_index_0 = max(0, error_step - self.error_average_steps)
        mean_index_1 = error_step
        new_mean_error_history = np.mean(error_history[mean_index_0:mean_index_1])
        mean_error_history[error_step] = new_mean_error_history
        self.error_steps_list[k] += 1

        best_second_diff_current = dist2 - dist
        self.effectiveness_num_list[k][new_min_ind] += 1
        self.effectiveness_sum_list[k][new_min_ind] += best_second_diff_current

        # (2) compare min of above, to current_min_row_to_row (min dist in table)
        # (3) if new min > current min:

        if do_random_init and self.tmp_ind_list[k] < cuda_table.get_num_rows():
            # necessary so dist matrix helper is not so slow at start

            # print('here!!!', train_input, train_output)
            dists[self.tmp_ind_list[k]] = np.inf
            cuda_table.set_matrix_row(row_index=self.tmp_ind_list[k],
                                      row_input=train_input,
                                      row_output=train_output,
                                      row_context=train_context,
                                      row_to_table_dists=dists,
                                      fast_init=True)

            if x_y_theta_output is not None:
                self.debug_x_y_theta_output[k][self.tmp_ind_list[k], :] = x_y_theta_output[:]
            if x_y_theta_input is not None:
                self.debug_x_y_theta_input[k][self.tmp_ind_list[k], :] = x_y_theta_input[:]

            self.tmp_ind_list[k] += 1
        else:
            if not self.post_init_done:
                print()
                print('Doing post init...')
                print()
                cuda_table.post_init()
                self.post_init_done = True
                print('Done')

            table_min_dist, table_min_dist_r, table_min_dist_c = cuda_table.get_min_dist()

            if new_min_dist > table_min_dist:
                # minimum distance of new row to current rows is greater than current minimum row-row distance
                # so: replace one row of current minimum, with new row

                # get one of the row indices of current minimum dist pair
                r_r_ind = table_min_dist_r  # could be table_min_dist_c

                # zero out its stats in preparation for replacement
                self.table_use_hist_list[k][r_r_ind] = 0
                self.row_ages_list[k][r_r_ind] = 0
                self.effectiveness_sum_list[k][r_r_ind] = 0.0
                self.effectiveness_num_list[k][r_r_ind] = 1

                dists[r_r_ind] = np.inf
                # replace the current min dist row, with the new row
                cuda_table.set_matrix_row(row_index=r_r_ind,
                                          row_input=train_input,
                                          row_output=train_output,
                                          row_context=train_context,
                                          row_to_table_dists=dists)  # optional arg- if None, cuda_table computes it internally as above

                self.debug_x_y_theta_output[k][r_r_ind, :] = x_y_theta_output[:]
                self.debug_x_y_theta_input[k][r_r_ind, :] = x_y_theta_input[:]

                if r_r_ind not in self.replacements_by_layer[k]:
                    self.replacements_by_layer[k][r_r_ind] = 1
                else:
                    self.replacements_by_layer[k][r_r_ind] = self.replacements_by_layer[k][r_r_ind] + 1

    def plan_and_get_debug_position_angle_list(self, goal_state, starting_state, visualizer, rays, topdown_info, current_goal_position_angle):
        plan_position_angle_list = None
        return plan_position_angle_list

    def save_to_pkl(self):
        f = open(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + '_PredictorEnsemble.pkl', 'wb')
        pickle.dump(self, f)
        f.close()

    def get_predictions_im(self, current_x_y_theta):
        '''
        get an image with possible predicted position + angle sequences from current position + angle
        :return:
        '''

        if False:
            return None

        predictions_im = None
        newest_states_list = self.layer_input_history.get_newest_states_list()

        N = len(newest_states_list)
        assert N == self.num_layers + 1
        assert len(self.cuda_tables_list) == self.num_layers

        print()
        print('**** prediction_im ****')

        input_pixel_values, x_y_theta = newest_states_list[0]

        # (1) make sure data stored with newest state 0 is actually current position / angle
        print('*** Check 1 ***')
        print('        Actual:',
              (x_y_theta[0], x_y_theta[1]),
              'Err: ',
              sqrt(pow(abs(x_y_theta[0] - current_x_y_theta[0]), 2) + pow(abs(x_y_theta[1] - current_x_y_theta[1]), 2)))

        # (2) get best row in table 0, given current state. get error in position/angle to that row's stored data
        first_layer_dists, _ = newest_states_list[1]

        best_row_index = np.argmax(first_layer_dists)
        row_input, row_output, _ = self.cuda_tables_list[0].get_matrix_row(row_index=best_row_index)
        stored_x_y_theta = self.debug_x_y_theta_output[0][best_row_index, :]

        print('*** Check 2 ***')
        print('        Stored:',
              (stored_x_y_theta[0], stored_x_y_theta[1]),
              'Err: ',
              sqrt(pow(abs(stored_x_y_theta[0] - current_x_y_theta[0]), 2) + pow(abs(stored_x_y_theta[1] - current_x_y_theta[1]), 2)))

        # (3) since above is not working, sanity check: do lookup here explicitly

        dists_tmp = self.cuda_tables_list[0].query(query_input=input_pixel_values,
                                                   query_output=None,
                                                   query_context=None)

        best_row_index = np.argmax(dists_tmp)
        row_input, row_output, _ = self.cuda_tables_list[0].get_matrix_row(row_index=best_row_index)
        stored_x_y_theta = self.debug_x_y_theta_output[0][best_row_index, :]

        print('*** Check 3 ***')
        print('        Stored:',
              (stored_x_y_theta[0], stored_x_y_theta[1]),
              'Err: ',
              sqrt(pow(abs(stored_x_y_theta[0] - current_x_y_theta[0]), 2) + pow(abs(stored_x_y_theta[1] - current_x_y_theta[1]), 2)))

        # TODO question: why are check 2, check 3 different above? should be same row?
        # TODO question: why error so large?

        # (4) cv2 imshow: input, best row input- for comparison
        #   i.e. row_input, input_pixel_values

        print()
        return predictions_im


        # newest_states_list: vectors with rank-values [0, 1.0] for each row of state vector
        for state_index in range(N - 1, 0, -1):
            print('     State: ', state_index)
            # does not include index 0: the input
            # i.e. for N==5: [4, 3, 2, 1]

            # starts in highest layer
            scaled_dists, extra_data = newest_states_list[state_index]

            # get argmax -> lowest dist -> highest confindence
            # TODO should use best "K", not just the one best

            best_row_index = np.argmax(scaled_dists)
            row_input, row_output, _ = self.cuda_tables_list[state_index - 1].get_matrix_row(row_index=best_row_index)

            # "output" column -> in space of scaled_dists for [l_n-1] -> prediction for this layer
            # "input" column -> in space of scaled_dists for [l_n-1] -> input for this layer from prev layer

            # correct:
            row_output_m = row_output.copy()
            row_input_m = row_input.copy()

            for layer_index in range(state_index - 2, -1, -1):
                # take output as dists for layer l_m
                # correct:
                # scaled_dists_m = row_output_m.copy()
                # incorrect:
                scaled_dists_m = row_input_m.copy()

                best_row_index_m = np.argmax(scaled_dists_m)
                row_input_m, row_output_m, _ = self.cuda_tables_list[layer_index].get_matrix_row(row_index=best_row_index_m)
                # print('        ', layer_index, '...')
                if layer_index == 0:
                    p_r = self.debug_x_y_theta_output[0][best_row_index_m, 0]
                    p_c = self.debug_x_y_theta_output[0][best_row_index_m, 1]
                    print('        p:', best_row_index_m, (p_r, p_c))
                    tmp_x = p_r
                    tmp_y = p_c

        # TODO now display actual position & angle corresponding to input:
        _, x_y_theta = newest_states_list[0]
        print('        Actual:', (x_y_theta[0], x_y_theta[1]), 'Err: ', sqrt(pow(abs(x_y_theta[0] - tmp_x), 2) + pow(abs(x_y_theta[1] - tmp_y), 2)))

        print()

        # now for each row, pair with position + angle, multiple by rank-value, and plot
        # remaining question: what position + angle is associated with each row?

        # ax.plot(self.debug_x_y_theta_input[k][:, 0], self.debug_x_y_theta_input[k][:, 1], 'go')
        # ax.plot(self.debug_x_y_theta_output[k][:, 0],  self.debug_x_y_theta_output[k][:, 1], 'ro')

        return predictions_im

    def get_table_im(self, layer_index):
        cuda_table = self.cuda_tables_list[layer_index]

        # layer 0: display as color images below
        # layer 1...N-1: display as grayscale [0, 1] values?

        input_dim = cuda_table.input_dim
        output_dim = cuda_table.output_dim
        context_dim = cuda_table.context_dim

        table = np.transpose(self.cuda_tables_list[layer_index].get_table_from_gpu())

        if layer_index == 0:
            entries = 40 * 2 * 3
        else:
            entries = table.shape[0]

        im_input = table[0:entries, 0:input_dim]
        im_prediction = table[0:entries, input_dim:input_dim + output_dim]
        im_context = table[0:entries, input_dim + output_dim::]

        if layer_index == 0:
            A = im_input
            B = im_prediction
            C = np.empty((A.shape[0] + B.shape[0], A.shape[1]))
            C[::2, :] = A
            C[1::2, :] = B

            im = np.reshape(C, (C.shape[0], C.shape[1] / 3, 3))

            im = cv2.resize(im, dsize=(0,0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        else:
            # A = im_input
            # B = im_prediction
            # C = im_context
            # D = np.hstack((A, B, C))
            D = table

            imscale = 0.2  # full table
            # imscale = 5.0
            im = cv2.resize(D, dsize=(0,0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        return im

    def plot_error(self, fig, ax, skip_slower_plots):

        for k in range(self.num_layers):
            mean_error_history = self.mean_error_histories_list[k]
            table_use_hist = self.table_use_hist_list[k]
            effectiveness_sum = self.effectiveness_sum_list[k]
            effectiveness_num = self.effectiveness_num_list[k]

            ax.cla()
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            #ax.set_ylim([0.0, 10.0])
            #ax.set_yticks(np.arange(0, 10, 0.5))
            #ax.axhline(y=1.5, color='g')
            if self.error_steps_list[k] > 10:
                thing_to_plot = mean_error_history[10:self.error_steps_list[k]]
                ax.plot(thing_to_plot, 'b-')
                fig.savefig(self.plots_save_folder + '/' + self.plots_prefix + '_' + 'error_history_' + str(k) + '.png', dpi=100)

            if not skip_slower_plots:
                # TODO find faster way to make these- ax.bar here is very slow
                ax.cla()
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                sorted_net_indices = np.argsort(table_use_hist)[::-1]
                ax.bar(np.arange(table_use_hist.shape[0]), table_use_hist[sorted_net_indices])
                fig.savefig(self.plots_save_folder + '/' + self.plots_prefix + '_' + 'table_use_hist_' + str(k) + '.png', dpi=100)

                ax.cla()
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                effectiveness_mean = np.divide(effectiveness_sum, effectiveness_num)
                sorted_effectiveness = np.argsort(effectiveness_mean)[::-1]
                ax.bar(np.arange(effectiveness_mean.shape[0]), effectiveness_mean[sorted_effectiveness])
                fig.savefig(self.plots_save_folder + '/' + self.plots_prefix + '_' + 'effectiveness_hist_' + str(k) + '.png', dpi=100)

            ax.cla()
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            ax.set_xlim([0 - 0.1, self.env_width_height + 0.1])
            ax.set_ylim([0 - 0.1, self.env_width_height + 0.1])

            ax.plot(self.debug_x_y_theta_input[k][:, 0],  self.debug_x_y_theta_input[k][:, 1], 'go')
            #ax.plot(self.debug_x_y_theta_output[k][:, 0],  self.debug_x_y_theta_output[k][:, 1], 'ro')

            ax.set_title(str(np.count_nonzero(self.debug_x_y_theta_input[k][:, 0]) * 1.0 / self.debug_x_y_theta_input[k].shape[0]))
            fig.savefig(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + 'debug_td_info_' + str(k) + '.png', dpi=100)
