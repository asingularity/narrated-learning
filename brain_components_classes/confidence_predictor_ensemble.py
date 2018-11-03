
from matrix_vector_dist_parallel import knn_query as knn_parallel_query
from cuda_dist_query import CudaTable
import time
import cv2
import pickle
import random
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
        input_context_dt = 4  # context future time

        # self.debug_x_y_theta_output = np.zeros((self.entries, 3))
        # self.debug_x_y_theta_input = np.zeros((self.entries, 3))

        # input (t), prediction (t+1), context (t+2)

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

        total_gb = 0

        layer_input_dim_list = []  # for initializing states_history object for input
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
                include_layers = ['ioc', 'ic_only']
            else:
                include_layers = ['io_only', 'i_only']

            layer_output_dim = layer_input_dim
            self.cuda_tables_list.append(CudaTable(num_entries=layer_entries,
                                                   input_dim=layer_input_dim,
                                                   output_dim=layer_output_dim,
                                                   context_dim=layer_context_dim,
                                                   include_layers=include_layers))
            layer_size_gb = self.cuda_tables_list[len(self.cuda_tables_list) - 1].get_size_gb()
            print( 'Init of layer', k, 'with rows X cols, size_GB,', '(', layer_entries, 'X', ('('+str(layer_input_dim) + ' + ' + str(layer_context_dim) + ' + ' + str(layer_output_dim)+')'), layer_size_gb)

            total_gb += layer_size_gb

            self.error_histories_list.append(np.zeros(self.max_history_length))
            self.mean_error_histories_list.append(np.zeros(self.max_history_length))
            self.error_steps_list.append(0)

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

        print ('finished initializing ensemble. total gb: ', total_gb)

        self.layer_input_history = StatesLimitedHistory(params={'max_delay': self.max_delay,
                                                                'states_dim_list': layer_input_dim_list})

        self.plots_save_folder = params['plots_save_folder']
        self.error_average_steps = params['error_average_steps']

        self.t = 0

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
            dists = np.tanh(dists * 0.001)  # TODO factor will be needed here, dependent on layer

            ind = np.argmin(dists)
            self.table_use_hist_list[k][ind] += 1
            self.row_ages_list[k][:] = self.row_ages_list[k][:] + 1

            self.last_step_layer_dists[k] = dists.copy()
            layer_input = dists.copy()
            if k < self.num_layers - 1:
                new_states_list.append(layer_input)

        self.layer_input_history.process_new_states(newest_states_list=new_states_list)
        # learning
        # should only happen if guaranteed enough history already

        for k in range(self.num_layers):
            dt_input_output = self.input_output_dt_steps_list[k]  # 2
            dt_input_context = self.input_context_dt_steps_list[k]  # 4
            assert dt_input_context > dt_input_output  # context always further future

            input_delay = dt_input_context  # 4
            output_delay = dt_input_context - dt_input_output  # 2

            train_input = self.layer_input_history.get_state(state_index=k, delay=input_delay)
            train_output = self.layer_input_history.get_state(state_index=k, delay=output_delay)

            if k == self.num_layers - 1:
                train_context = None
            else:
                # train_context = self.layer_dists_history[k + 1][context_delay]
                # since context_delay == 0, instead of above, we can use last step dists:
                train_context = self.last_step_layer_dists[k + 1]

            self._learn_seq_kmeans(k=k,
                                   cuda_table=self.cuda_tables_list[k],
                                   train_input=train_input,
                                   train_output=train_output,
                                   train_context=train_context,
                                   error_history=self.error_histories_list[k],
                                   mean_error_history=self.mean_error_histories_list[k])

        to_debug_print = {}
        return to_debug_print

    def _learn_seq_nn(self, k, cuda_table, train_input, train_output, train_context, error_history, mean_error_history):
        '''
        :param k: layer
        :param cuda_table: self.cuda_tables_list[k]
        :param train_input:
        :param train_output:
        :param train_context:
        :param error_history: self.error_histories_list[k]
        :param mean_error_history: self.mean_error_histories_list[k]
        :return:
        '''
        pass
        # can do this shortcut (and not store all row-to-row distances), because only this process is updating rows
        #   won't work... the algorithm (seq nn) assumes you *unconditionally* add new data point to your table
        #   you only *don't add* the data point if its min dist happens to be less than min dist in table
        #   (making it the minimum)
        #
        # (1) get distance of new row to all current rows in table
        # (2) compare min of above, to current_min_row_to_row (min dist in table)
        # (3) if new min > current min:
        #       replace one of current pair with new row, in table
        #       current_min_row_to_row = new_min (over whole table)- would have to be next closest from before
        #       also find next new next closest
        #       (can't avoid keeping track of all dists)
        #       i.e.
        #       end up adding new point and all its distances to existing points, somewhere in the ordered list of dists, and remove current min
        # what we need is a data structure ordered by distance
        # a queue that we can insert new data into in correct positions
        # answer: sortedcontainers: http://www.grantjenks.com/docs/sortedcontainers/
        #   we want SortedDict: bisect_left / bisect_right

        '''
        Idea:
            have:
                ordered list of dists
                dict: index in ordered dist list -> (index pair)
                OR
                sorted dictionary: dist -> (index pair)
        '''

        # sequential nn: replace "worst" pair in table (smallest distance), with new best pair, new best pair better than worst pair
        #   "worst" pair: smallest distance between rows in pair
        #   "better": larger distance between rows in pair

        # (1) get distance of new row to all current rows in table

        dists = cuda_table.query(query_input=train_input,
                                 query_output=train_output,
                                 query_context=train_context)
        sorted_dist_indices = np.argsort(dists)
        new_min_ind = sorted_dist_indices[0]
        new_min_dist = dists[new_min_ind]

        # (2) compare min of above, to current_min_row_to_row (min dist in table)
        # (3) if new min > current min:

        if new_min_dist > self.current_min_dists[k]:
            # minimum distance of new row to current rows is greater than current minimum row-row distance
            # so: replace one row of current minimum, with new row

            # get one of the row indices of current minimum dist pair
            r_r_ind = self.current_min_dist_indices[k][0]

            # zero out its stats in preparation for replacement
            self.table_use_hist_list[k][r_r_ind] = 0
            self.row_ages_list[k][r_r_ind] = 0
            self.effectiveness_sum_list[k][r_r_ind] = 0.0
            self.effectiveness_num_list[k][r_r_ind] = 0

            # replace the current min dist row, with the new row
            cuda_table.set_matrix_row(row_index=r_r_ind,
                                      row_input=train_input,
                                      row_output=train_output,
                                      row_context=train_context)

            # THIS IS WRONG: we need to get the new minimum (something else in table- what was second minimum before?)
            # now- using sorted dicts (?), get new worst row
            self.current_min_dists[k] = min_dist
            self.current_min_dist_indices[k][1] = r_r_ind
            self.current_min_dist_indices[k][0] = min_ind

        # OLD CODE:
        if False:
            row_replace_candidates = np.nonzero(self.row_ages_list[k] > min_replaced_row_age)[0]
            effectiveness_mean = np.divide(self.effectiveness_sum_list[k], self.effectiveness_num_list[k])

            if len(row_replace_candidates) > 0 and self.t > self.last_replacement_t_list[k] + replacement_every_k_steps:
                r_r_c_ind = np.argmin(effectiveness_mean[row_replace_candidates])

                r_r_ind = row_replace_candidates[r_r_c_ind]
                # if k == 0:
                # print('replacing layer, index:', k, r_r_ind)
                self.table_use_hist_list[k][r_r_ind] = 0
                self.row_ages_list[k][r_r_ind] = 0
                self.effectiveness_sum_list[k][r_r_ind] = 0.0
                self.effectiveness_num_list[k][r_r_ind] = 0

                cuda_table.set_matrix_row(row_index=r_r_ind,
                                          row_input=train_input,
                                          row_output=train_output,
                                          row_context=train_context)

                # self.debug_x_y_theta_output[r_r_ind, :] = x_y_theta[:]
                # self.debug_x_y_theta_input[r_r_ind, :] = last_x_y_theta[:]

                self.last_replacement_t_list[k] = self.t

    def _learn_seq_kmeans(self, k, cuda_table, train_input, train_output, train_context, error_history, mean_error_history):

        do_random_init = self.do_random_init  # initialize with random entries
        do_adaptation = self.do_adaptation  # WTA-based learning
        do_replacements = self.do_replacements  # replace low effectiveness over time

        dists = cuda_table.query(query_input=train_input,
                                 query_output=train_output,
                                 query_context=train_context)

        dists = np.tanh(dists * 0.001)  # TODO factor will be needed here, dependent on layer
        #print(k, np.unique(dists))
        sorted_dist_indices = np.argsort(dists)
        sorted_dists = dists[sorted_dist_indices]
        ind2 = sorted_dist_indices[1]
        dist2 = dists[ind2]
        ind = sorted_dist_indices[0]
        dist = dists[ind]

        error_step = self.error_steps_list[k]
        error_history[error_step] = dist
        mean_index_0 = max(0, error_step - self.error_average_steps)
        mean_index_1 = error_step
        new_mean_error_history = np.mean(error_history[mean_index_0:mean_index_1])
        mean_error_history[error_step] = new_mean_error_history
        self.error_steps_list[k] += 1

        if do_random_init and self.tmp_ind_list[k] < cuda_table.get_num_rows():
            # print('here!!!', train_input, train_output)
            cuda_table.set_matrix_row(row_index=self.tmp_ind_list[k],
                                      row_input=train_input,
                                      row_output=train_output,
                                      row_context=train_context)

            # self.debug_x_y_theta_output[self.tmp_ind, :] = x_y_theta[:]
            # self.debug_x_y_theta_input[self.tmp_ind, :] = last_x_y_theta[:]
            self.tmp_ind_list[k] += 1

        best_second_diff_current = dist2 - dist
        self.effectiveness_num_list[k][ind] += 1
        self.effectiveness_sum_list[k][ind] += best_second_diff_current

        if do_adaptation and self.tmp_ind_list[k] >= cuda_table.get_num_rows():
            #print('would adapt:', type(train_input), type(train_output), type(train_context))

            cuda_table.seq_kmeans_adapt(row_index=ind,
                                        rate=1.0 / self.effectiveness_num_list[k][ind],
                                        row_input=train_input,
                                        row_output=train_output,
                                        row_context=train_context)

            # for theta: could be weird... discontinuities
            # self.debug_x_y_theta_output[ind, :] = 0.9 * self.debug_x_y_theta_output[ind, :] + 0.1 * x_y_theta[:]
            # self.debug_x_y_theta_input[ind, :] = 0.9 * self.debug_x_y_theta_input[ind, :] + 0.1 * last_x_y_theta[:]

        min_replaced_row_age = 1000
        replacement_every_k_steps = self.replacement_every_k_steps

        if do_replacements:
            row_replace_candidates = np.nonzero(self.row_ages_list[k] > min_replaced_row_age)[0]
            effectiveness_mean = np.divide(self.effectiveness_sum_list[k], self.effectiveness_num_list[k])

            if len(row_replace_candidates) > 0 and self.t > self.last_replacement_t_list[k] + replacement_every_k_steps:

                r_r_c_ind = np.argmin(effectiveness_mean[row_replace_candidates])

                r_r_ind = row_replace_candidates[r_r_c_ind]
                # if k == 0:
                #print('replacing layer, index:', k, r_r_ind)
                self.table_use_hist_list[k][r_r_ind] = 0
                self.row_ages_list[k][r_r_ind] = 0
                self.effectiveness_sum_list[k][r_r_ind] = 0.0
                self.effectiveness_num_list[k][r_r_ind] = 1

                cuda_table.set_matrix_row(row_index=r_r_ind,
                                          row_input=train_input,
                                          row_output=train_output,
                                          row_context=train_context)

                # self.debug_x_y_theta_output[r_r_ind, :] = x_y_theta[:]
                # self.debug_x_y_theta_input[r_r_ind, :] = last_x_y_theta[:]

                self.last_replacement_t_list[k] = self.t

    def plan_and_get_debug_position_angle_list(self, goal_state, starting_state, visualizer, rays, topdown_info, current_goal_position_angle):
        plan_position_angle_list = None
        return plan_position_angle_list

    def save_to_pkl(self):
        f = open(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + '_PredictorEnsemble.pkl', 'wb')
        pickle.dump(self, f)
        f.close()

    def get_table_im(self):
        cuda_table = self.cuda_tables_list[0]

        input_dim = cuda_table.input_dim
        output_dim = cuda_table.output_dim
        context_dim = cuda_table.context_dim

        table = np.transpose(self.cuda_tables_list[0].get_table_from_gpu())

        entries = 40

        im_input = table[0:entries, 0:input_dim]
        im_prediction = table[0:entries, input_dim:input_dim+output_dim]
        im_context = table[0:entries, input_dim+output_dim::]

        if False:
            print('***')
            print(table.shape)
            print(im_input.shape, im_prediction.shape, im_context.shape)
            # interleave rows so context below prediction below input

            A = im_input
            B = im_prediction
            C = im_context
            D = np.empty((A.shape[0] + B.shape[0] + C.shape[0], A.shape[1]))
            print(D.shape)
            D[::3, :] = A
            D[1::3, :] = B
            D[2::3, :] = C
            D = np.reshape(D, (D.shape[0], D.shape[1] / 3, 3))

            im = D
        else:
            A = im_input
            B = im_prediction
            C = np.empty((A.shape[0] + B.shape[0], A.shape[1]))
            #im = np.concatenate((im_input, im_prediction))
            C[::2, :] = A
            C[1::2, :] = B

            im = np.reshape(C, (C.shape[0], C.shape[1] / 3, 3))

        #print('**', np.amin(im), np.amax(im), im.dtype, im.shape)

        im = cv2.resize(im, dsize=(0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)

        return im

    def plot_error(self, fig, ax):

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

            # TODO re-enable this for first layer!
            if False:
                ax.cla()
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                ax.set_xlim([0 - 0.1, self.env_width_height + 0.1])
                ax.set_ylim([0 - 0.1, self.env_width_height + 0.1])
                ax.plot(self.debug_x_y_theta_input[:, 0],  self.debug_x_y_theta_input[:, 1], 'go')
                #ax.plot(self.debug_x_y_theta_output[:, 0],  self.debug_x_y_theta_output[:, 1], 'ro')
                ax.set_title(str(np.count_nonzero(self.debug_x_y_theta_input[:, 0]) * 1.0 / self.debug_x_y_theta_input.shape[0]))
                fig.savefig(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + 'debug_td_info' + '.png', dpi=100)
