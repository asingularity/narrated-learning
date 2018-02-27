
from knn_parallel import knn_query as knn_parallel_query
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


class PredictorEnsemble(object):
    def __init__(self, params):
        print 'initializing ensemble...'

        self.do_random_init = params['do_random_init']  # False  # initialize with random (first) entries
        self.do_adaptation = params['do_adaptation']  # True  # WTA-based learning
        self.do_replacements = params['do_replacements']  # True  # replace low effectiveness over time

        self.max_history_length = params['max_history_length']
        self.error_history = np.zeros(params['max_history_length'])
        self.mean_error_history = np.zeros(params['max_history_length'])

        self.replacement_every_k_steps = params['replacement_every_k_steps']  # TODO put as parameter. 100 for 800 rows, 10 for 8000 rows

        self.use_context_in_knn_diff = params['use_context_in_knn_diff']

        self.plots_prefix = params['plots_prefix']
        self.env_width_height = params['env_width_height']
        self.error_step = 0

        self.entries = params['entries']
        self.dim = params['dim']

        self.debug_x_y_theta_output = np.zeros((self.entries, 3))
        self.debug_x_y_theta_input = np.zeros((self.entries, 3))

        # input (t), prediction (t+1), context (t+2)
        self.table = np.zeros((self.entries, self.dim * 3)).astype(np.float32)
        #self.table_dist = np.zeros((self.entries, self.entries))  # max 10k * 10k
        #self.table_dist[np.arange(self.entries), np.arange(self.entries)] = np.inf

        #self.table_dist = np.random.random((self.entries, self.entries))  # max 10k * 10k

        self.min_dist_pair = (0, 1)  # doesn't matter since they all start 0
        self.min_dist = 0

        self.temp_array = np.zeros(self.entries).astype(np.float32)
        print 'finished initializing ensemble.'

        self.plots_save_folder = params['plots_save_folder']
        self.error_average_steps = params['error_average_steps']

        self.error_history = np.zeros(self.max_history_length)
        self.mean_error_history = np.zeros(self.max_history_length)
        self.error_step = 0

        self.table_use_hist = np.zeros(self.entries)
        self.tmp_ind = 0

        self.effectiveness_sum = np.zeros(self.entries)
        self.effectiveness_num = np.ones(self.entries)

        self.row_ages = np.zeros(self.entries)

        self.t = 0
        self.last_replacement_t = 0
        self.radius = None

        # things to precompute
        self.output_input_distance = None
        self.output_context_distance = None

        self.use_cuda = True
        self.cuda_split_n = 1
        self.cuda_init_done = False

    def precompute(self):
        '''
        to be run after learning, on a learned table
        called after loading ensemble from file, in robot_brain.py
        :return: nothing. fills in self.input_output_distance, self.context_output_distance
        '''

        self.use_cuda = True
        self.cuda_split_n = 1
        self.cuda_init_done = False
        self.original_way = True

        # *** table is: | input | output | context | ***

        # dist, ind = knn_parallel_query(self.table, new_entry, self.temp_array, self.table.shape[0], self.dim * 3)

        table_input = np.ascontiguousarray(self.table[:, 0:self.dim])
        table_output = np.ascontiguousarray(self.table[:, self.dim:self.dim * 2])
        table_context = np.ascontiguousarray(self.table[:, self.dim * 2::])

        self.table_input = table_input.astype(np.float32)
        self.table_output = table_output.astype(np.float32)
        self.table_context = table_context.astype(np.float32)

        print table_input.shape, table_output.shape, table_context.shape

        precompute_dist = self.original_way
        skcuda.misc.init()

        self.o_i_dist_gpu = None
        self.o_c_dist_gpu = None

        if precompute_dist:
            self.output_input_distance = np.zeros((self.entries, self.entries))
            for k in range(self.entries):
                output_entry = table_output[k, :]
                dist, ind = knn_parallel_query(table_input, output_entry, self.temp_array, self.table.shape[0], self.dim)
                self.output_input_distance[k, range(0, self.entries)] = self.temp_array[:]

            self.output_context_distance = np.zeros((self.entries, self.entries))
            for k in range(self.entries):
                output_entry = table_output[k, :]
                dist, ind = knn_parallel_query(table_context, output_entry, self.temp_array, self.table.shape[0], self.dim)
                self.output_context_distance[k, range(0, self.entries)] = self.temp_array[:]

            if self.cuda_split_n == 1 and self.use_cuda:
                self.o_i_dist_gpu = gpuarray.to_gpu(self.output_input_distance)
                self.o_c_dist_gpu = gpuarray.to_gpu(self.output_context_distance)

    def step(self, last_input_state, input_state, next_input_state, last_x_y_theta, x_y_theta, learn=True):
        '''
        for learning mode

        :param last_input_state:
        :param input_state:
        :param next_input_state:
        :param last_x_y_theta:
        :param x_y_theta:
        :param learn:
        :return:
        '''
        self.t += 1

        predictor_context = next_input_state
        predictor_output = input_state
        predictor_input = last_input_state

        # 1. find input+output (IO) error
        new_entry = np.concatenate((predictor_input, predictor_output))
        new_entry = np.concatenate((new_entry, predictor_context))

        if self.use_context_in_knn_diff:
            dist, ind = knn_parallel_query(self.table, new_entry, self.temp_array, self.table.shape[0], self.dim * 3)
        else:
            dist, ind = knn_parallel_query(self.table, new_entry, self.temp_array, self.table.shape[0], self.dim * 2)

        self.table_use_hist[ind] += 1

        self.row_ages[:] = self.row_ages[:] + 1

        # find second best
        sorted_dist_indices = np.argsort(self.temp_array)
        sorted_dists = self.temp_array[sorted_dist_indices]
        assert self.temp_array[sorted_dist_indices[0]] == dist
        ind2 = sorted_dist_indices[1]
        ind3 = sorted_dist_indices[2]
        dist2 = self.temp_array[ind2]

        self.error_history[self.error_step] = dist
        mean_index_0 = max(0, self.error_step - self.error_average_steps)
        mean_index_1 = self.error_step
        self.mean_error_history[self.error_step] = np.mean(self.error_history[mean_index_0:mean_index_1])
        self.error_step += 1

        to_debug_print = {}

        if learn:

            do_random_init = self.do_random_init  # initialize with random entries
            do_adaptation = self.do_adaptation  # WTA-based learning
            do_replacements = self.do_replacements  # replace low effectiveness over time

            if do_random_init and self.tmp_ind < self.table.shape[0]:
                self.table[self.tmp_ind, :] = new_entry[:]
                self.debug_x_y_theta_output[self.tmp_ind, :] = x_y_theta[:]
                self.debug_x_y_theta_input[self.tmp_ind, :] = last_x_y_theta[:]
                self.tmp_ind += 1
            else:
                pass

            best_second_diff_current = dist2 - dist
            self.effectiveness_num[ind] += 1
            self.effectiveness_sum[ind] += best_second_diff_current

            # simple best learns:
            if do_adaptation:
                self.table[ind, :] = 0.9 * self.table[ind, :] + 0.1 * new_entry
                # for theta: could be weird... discontinuities
                self.debug_x_y_theta_output[ind, :] = 0.9 * self.debug_x_y_theta_output[ind, :] + 0.1 * x_y_theta[:]
                self.debug_x_y_theta_input[ind, :] = 0.9 * self.debug_x_y_theta_input[ind, :] + 0.1 * last_x_y_theta[:]

                if 0:  # try something like a SOM
                    if self.radius is None:
                        self.radius = np.inf

                    if self.radius == np.inf:
                        self.table[:, :] = 0.99 * self.table[:, :] + 0.01 * new_entry
                        self.radius = 10.0
                    else:
                        within_radius = sorted_dist_indices[sorted_dists < self.radius]
                        self.table[within_radius, :] = 0.99 * self.table[within_radius, :] + 0.01 * new_entry
                        self.radius *= 0.9999

                        to_debug_print['radius'] = self.radius
                        to_debug_print['num within'] = len(within_radius)

            min_replaced_row_age = 1000
            replacement_every_k_steps = self.replacement_every_k_steps

            if do_replacements:
                #to_debug_print['warning'] = 'doing replacements!'

                row_replace_candidates = np.nonzero(self.row_ages > min_replaced_row_age)[0]
                effectiveness_mean = np.divide(self.effectiveness_sum, self.effectiveness_num)

                if len(row_replace_candidates) > 0 and self.t > self.last_replacement_t + replacement_every_k_steps:
                    r_r_c_ind = np.argmin(effectiveness_mean[row_replace_candidates])
                    r_r_c_eff = effectiveness_mean[row_replace_candidates][r_r_c_ind]

                    #if r_r_c_eff < 0.2 or np.isnan(r_r_c_eff):
                    r_r_ind = row_replace_candidates[r_r_c_ind]
                    #print r_r_ind
                    self.table_use_hist[r_r_ind] = 0
                    self.row_ages[r_r_ind] = 0
                    self.effectiveness_sum[r_r_ind] = 0.0
                    self.effectiveness_num[r_r_ind] = 0

                    self.table[r_r_ind, :] = new_entry[:]
                    self.debug_x_y_theta_output[r_r_ind, :] = x_y_theta[:]
                    self.debug_x_y_theta_input[r_r_ind, :] = last_x_y_theta[:]

                    self.last_replacement_t = self.t
                    #    print 'replacing: ', r_r_c_eff
                    #else:
                    #    print 'NOT replacing: ', r_r_c_eff

        return to_debug_print

    def plan_and_get_debug_position_angle_list(self, goal_state, starting_state, visualizer, rays, topdown_info, current_goal_position_angle):
        print 'predictor_ensemble::plan_and_get_debug_position_angle_list'
        if self.original_way:
            return self.plan_and_get_debug_position_angle_list_OLD_ALL_TO_ALL(goal_state, starting_state, visualizer, rays, topdown_info, current_goal_position_angle)
        else:
            # planning:
            N = 5  # currently this is also number of planned steps
            iter_steps = 10
            num_to_propagate = 1

            # TODO normalization

            assert self.entries == self.table.shape[0]

            output_uncertainties = {}
            top_output_rows = {}
            top_output_row_indices = {}
            top_output_row_uncertainties = {}

            for k in range(N):
                output_uncertainties[k] = None
                top_output_rows[k] = None
                top_output_row_indices[k] = None
                top_output_row_uncertainties[k] = None

            for iter_step in range(iter_steps):
                print 'starting iter_step: ', iter_step

                for level in range(N):
                    print 'starting level:', level
                    if level == 0:
                        input_rows = starting_state[np.newaxis, :]
                        print 'input_rows.shape:', input_rows.shape
                    else:
                        input_rows = top_output_rows[level - 1]

                    if level == N - 1:
                        context_rows = goal_state[np.newaxis, :]
                        print 'context_rows.shape:', context_rows.shape
                    else:
                        context_rows = top_output_rows[level + 1]

                    # we have K input rows, L context rows coming to Table

                    # for each row in input_rows: calculate distance to self.table_input, update uncertainty totals

                    input_uncertainties = None
                    if input_rows is not None:
                        for r in range(input_rows.shape[0]):
                            dist, ind = knn_parallel_query(self.table_input, input_rows[r, :], self.temp_array, self.table.shape[0], self.dim)
                            if level == 0:
                                factor = 1.0
                            else:
                                factor = top_output_row_uncertainties[level - 1][r]

                            if input_uncertainties is None:
                                input_uncertainties = self.temp_array.copy() * factor    # TODO copy necessary?
                            else:
                                input_uncertainties = input_uncertainties + self.temp_array + factor

                    # for each row in context_rows: calculate distance to (all rows in ) self.table_context, update uncertainty totals

                    context_uncertainties = None
                    if context_rows is not None:
                        for r in range(context_rows.shape[0]):
                            dist, ind = knn_parallel_query(self.table_context, context_rows[r, :], self.temp_array,
                                                           self.table.shape[0], self.dim)
                            if level == N - 1:
                                factor = 1.0
                            else:
                                factor = top_output_row_uncertainties[level + 1][r]

                            if context_uncertainties is None:
                                context_uncertainties = self.temp_array.copy() * factor   # TODO copy necessary?
                            else:
                                context_uncertainties = context_uncertainties + self.temp_array + factor

                    # result should be an uncertainty value for every row (entire output column) of Table

                    assert not (input_uncertainties is None and context_uncertainties is None)
                    sum_uncertainties = np.zeros(self.entries)

                    if input_uncertainties is not None:
                        print '+inp: ', np.sum(input_uncertainties * 1.0 / input_rows.shape[0])
                        sum_uncertainties = sum_uncertainties + input_uncertainties * 1.0 / input_rows.shape[0]
                    if context_uncertainties is not None:
                        print '+ctx: ', np.sum(context_uncertainties * 1.0 / context_rows.shape[0])
                        sum_uncertainties = sum_uncertainties + context_uncertainties * 1.0 / context_rows.shape[0]

                    top_row_ind = np.argsort(sum_uncertainties)
                    top_row_ind = top_row_ind[0:num_to_propagate]

                    top_output_rows[level] = self.table_output[top_row_ind, :].copy()  # TODO copy necessary?
                    top_output_row_indices[level] = top_row_ind

                    tmp = sum_uncertainties[top_row_ind]
                    tmp = tmp - np.amin(tmp)
                    tmp = tmp * 1.0 / np.amax(tmp)
                    #print 'tmp', tmp

                    #top_output_row_uncertainties[level] = tmp
                    top_output_row_uncertainties[level] = np.ones(num_to_propagate)

                    #print top_output_row_uncertainties

                plan_position_angle_list = []

                for k in range(N):
                    # rand_entry = random.randint(0, self.entries - 1)
                    rand_entry = top_output_row_indices[k][0]

                    plan_position_angle_list.append((self.debug_x_y_theta_output[rand_entry, 0],
                                                     self.debug_x_y_theta_output[rand_entry, 1],
                                                     self.debug_x_y_theta_output[rand_entry, 2]))

                im = visualizer._get_topdown_map(rays, topdown_info, current_goal_position_angle, plan_position_angle_list)
                cv2.imshow('planned', im)
                cv2.waitKey(1)
                print 'finished iter_step: ', iter_step

                time.sleep(0.1)

                if 0:

                    out_uncert = self._run_level_0(input_starting_state=starting_state,
                                                   context={'rows': 'table_output', 'uncertainties': output_uncertainties[level + 1]})
                    #print 'setting output uncertainties, level:', level
                    output_uncertainties[level] = out_uncert.copy()

                    # run intermediate levels
                    for level in range(1, N - 1):
                        out_uncert = self._run_level_k(input={'rows': 'table_output', 'uncertainties': output_uncertainties[level - 1]},
                                                       context={'rows': 'table_output', 'uncertainties': output_uncertainties[level + 1]},
                                                       debug_print=False)#(level==2))
                        #print 'setting output uncertainties, level:', level
                        output_uncertainties[level] = out_uncert.copy()

                    # run highest level: N - 1
                    level = N - 1
                    out_uncert = self._run_level_N(input={'rows': 'table_output', 'uncertainties': output_uncertainties[level - 1]},
                                                   context_goal_state=goal_state)
                    output_uncertainties[level] = out_uncert.copy()

            return plan_position_angle_list


    def plan_and_get_debug_position_angle_list_OLD_ALL_TO_ALL(self, goal_state, starting_state, visualizer, rays, topdown_info, current_goal_position_angle):
        print 'predictor_ensemble::plan_and_get_debug_position_angle_list'

        # print goal_state.shape, starting_state.shape  # 48, 48
        # print self.table.shape # (800, 144)
        # print self.dim  # 48

        # planning:
        N = 5  # currently this is also number of planned steps
        iter_steps = 10
        # TODO should be able to display plan as it evolves, inside this function, each iteration
        # for now, just display intermediate states here

        # rows: implicit - table_output - output rows are always the same column of same table.
        #   currently all rows are implicitly propagated with some assigned uncertainty
        #   (will require all-to-all distance pre-compute between input & context, output & context, input & output)
        # uncertainties: inverse confidence (propagated error) of next or previous level's predictions.
        #   needs normalization?

        output_uncertainties = {}

        for k in range(N):
            output_uncertainties[k] = None

        for iter_step in range(iter_steps):
            # run level 0
            level = 0
            out_uncert = self._run_level_0(input_starting_state=starting_state,
                                           context={'rows': 'table_output', 'uncertainties': output_uncertainties[level + 1]})
            #print 'setting output uncertainties, level:', level
            output_uncertainties[level] = out_uncert.copy()

            # run intermediate levels
            for level in range(1, N - 1):
                out_uncert = self._run_level_k(input={'rows': 'table_output', 'uncertainties': output_uncertainties[level - 1]},
                                               context={'rows': 'table_output', 'uncertainties': output_uncertainties[level + 1]},
                                               debug_print=False)#(level==2))
                if 0: #level == 2:
                    print '**////'
                    print output_uncertainties[level - 1], output_uncertainties[level + 1]
                    print '**////'
                    print out_uncert
                    print '////***'
                #print 'setting output uncertainties, level:', level
                output_uncertainties[level] = out_uncert.copy()

            # run highest level: N - 1
            level = N - 1
            out_uncert = self._run_level_N(input={'rows': 'table_output', 'uncertainties': output_uncertainties[level - 1]},
                                           context_goal_state=goal_state)
            output_uncertainties[level] = out_uncert.copy()

            plan_position_angle_list = []

            for k in range(N):
                #rand_entry = random.randint(0, self.entries - 1)
                rand_entry = np.argmin(output_uncertainties[k])

                plan_position_angle_list.append((self.debug_x_y_theta_output[rand_entry, 0],
                                                 self.debug_x_y_theta_output[rand_entry, 1],
                                                 self.debug_x_y_theta_output[rand_entry, 2]))

            im = visualizer._get_topdown_map(rays, topdown_info, current_goal_position_angle, plan_position_angle_list)
            print 'iter_step: ', iter_step
            cv2.imshow('planned', im)
            cv2.waitKey(10)
            time.sleep(1)

        return plan_position_angle_list

    def _compute_uncertainty(self, output_to_column_distance, output_column_uncertainties, output_to_column_distance_gpu=None):
        '''
        :param dist_mat: ex. self.output_context_distance, or self.output_input_distance
        :param output_column_uncertainties: ex. uncertainty associated with each output-column row
        :return: uncertainty associated with each input or context row
        '''

        # TODO speed this up!

        #  compute context "uncertainties" by computing:
        #  for each row in input/context column of table:
        #       1. compute distance to each row in output column of table.
        #       2. weight by output_uncertainty of that row. (from "context" parameter)
        #       3. take min or max? for "sum" uncertainty for this input/context-column row
        #print 'starting slow part?'

        if 1:
            if output_column_uncertainties is None:
                uncert = np.amin(output_to_column_distance, axis=0)
            else:
                if self.use_cuda:
                    if not self.cuda_init_done:
                        print 'doing cuda init...'
                        #linalg.init()
                        #skcuda.misc.init()
                        self.cuda_init_done = True

                    entries = self.entries

                    split_n = self.cuda_split_n  #20

                    if split_n > 1:
                        v1 = np.reshape(output_column_uncertainties, (output_column_uncertainties.shape[0], 1))

                        uncert = np.ones(entries) * np.inf

                        entries_per_split = entries * 1.0 / split_n
                        assert int(entries_per_split) == entries_per_split
                        entries_per_split = int(entries_per_split)

                        indices_start_end_list = []
                        for k in range(split_n):
                            indices_start_end_list.append((k * entries_per_split, (k + 1) * entries_per_split))

                        for indices_start_end in indices_start_end_list:
                            i0 = indices_start_end[0]
                            i1 = indices_start_end[1]

                            o_i_dist_gpu = gpuarray.to_gpu(output_to_column_distance[i0:i1, :])
                            v1_gpu = gpuarray.to_gpu(v1[i0:i1, :])
                            sum_gpu = misc.add(o_i_dist_gpu, v1_gpu)
                            uncert_gpu = misc.min(sum_gpu, axis=0)  # , keepdims=False)
                            new_uncert = uncert_gpu.get()
                            uncert = np.minimum(uncert, new_uncert[:])
                    else:
                        v1 = output_column_uncertainties
                        v1_gpu = gpuarray.to_gpu(v1)
                        sum_gpu = misc.add_matvec(output_to_column_distance_gpu, v1_gpu, axis=0)
                        uncert_gpu = misc.min(sum_gpu, axis=0)  # , keepdims=False)
                        uncert = uncert_gpu.get()
                else:
                    tmp = np.reshape(output_column_uncertainties, (output_column_uncertainties.shape[0], 1))
                    tmp2 = output_to_column_distance + tmp
                    uncert = np.amin(tmp2, axis=0)

                #uncert = np.amin(np.multiply(output_to_column_distance, tmp), axis=0)
        else:
            uncert = []
            for k in range(self.entries):
                if output_column_uncertainties is None:
                    row_uncert = output_to_column_distance[:, k]
                else:
                    v1 = output_to_column_distance[:, k]
                    #if np.amin(v1) == 0.0:
                        #print '**@#$Q@#%$*&#%'
                        #print k
                        #print np.argmin(v1)
                    v2 = output_column_uncertainties
                    row_uncert = self._combine_uncertainties(v1, v2)  # np.multiply???

    #            print '*k', k, row_uncert
                row_uncert = np.amin(row_uncert)

                uncert.append(row_uncert)
            uncert = np.array(uncert)

        #print 'finished slow part'

        #print '*********************'
        #print '*** 1'
        #print np.amin(output_to_column_distance)
        #print '**** 2'
        #print np.sum(np.array(uncert))

        return uncert

    def _combine_uncertainties(self, uncert_1, uncert_2):
        #uncert_3 = np.multiply(uncert_1, uncert_2)

        uncert_3 = uncert_1 + uncert_2
        #uncert_3 = uncert_3 * 1.0 / np.amax(uncert_3)
        return uncert_3

    def _run_level_0(self, input_starting_state, context):
        '''

        out_uncert = self._run_level_0(input_starting_state=starting_state,
                                       context={'rows': 'table_output', 'uncertainties': output_uncertainties[level + 1]})

        :param input_starting_state: single array of size self.dim
        :param context: dict, ex: {'rows': 'table_output', 'uncertainties': output_uncertainties[level + 1]}
        :return: single array of size self.entries
        '''

        self.temp_array = np.zeros(self.entries).astype(np.float32)

        # receive: input vector of current input
        # receive: each output row of table, as context, and associated uncertainty
        # compute: uncertainty for all output rows

        # take distance of input_starting_state to all rows in input column to get input "uncertainties"
        dist, ind = knn_parallel_query(self.table, input_starting_state, self.temp_array, self.table.shape[0], self.dim)
        input_uncertainties = self.temp_array.copy()

        context_uncertainties = self._compute_uncertainty(output_to_column_distance=self.output_context_distance,
                                                          output_column_uncertainties=context['uncertainties'],
                                                          output_to_column_distance_gpu=self.o_c_dist_gpu)

        # for now define output_uncertainty as max or sum (input_uncertainty, context_uncertainty)
        out_uncert = self._combine_uncertainties(input_uncertainties, context_uncertainties)

        return out_uncert

    def _run_level_k(self, input, context, debug_print=False):

        input_uncertainties = self._compute_uncertainty(output_to_column_distance=self.output_input_distance,
                                                        output_column_uncertainties=input['uncertainties'],
                                                        output_to_column_distance_gpu=self.o_i_dist_gpu)

        context_uncertainties = self._compute_uncertainty(output_to_column_distance=self.output_context_distance,
                                                          output_column_uncertainties=context['uncertainties'],
                                                          output_to_column_distance_gpu=self.o_c_dist_gpu)

        out_uncert = self._combine_uncertainties(input_uncertainties, context_uncertainties)

        if debug_print:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print input_uncertainties
            print context_uncertainties
            print out_uncert
            print '``````````````````````````````````````'

        return out_uncert

    def _run_level_N(self, input, context_goal_state):

        input_uncertainties = self._compute_uncertainty(output_to_column_distance=self.output_input_distance,
                                                        output_column_uncertainties=input['uncertainties'],
                                                        output_to_column_distance_gpu=self.o_i_dist_gpu)

        dist, ind = knn_parallel_query(self.table_context, context_goal_state, self.temp_array, self.table.shape[0], self.dim)
        context_uncertainties = self.temp_array.copy()

        out_uncert = self._combine_uncertainties(input_uncertainties, context_uncertainties)

        return out_uncert

    def save_to_pkl(self):
        f = open(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + '_PredictorEnsemble.pkl', 'w')
        pickle.dump(self, f)
        f.close()

    def get_table_im(self):

        entries = 40

        im_input = self.table[0:entries, 0:self.dim]
        im_prediction = self.table[0:entries, self.dim:self.dim * 2]
        im_context = self.table[0:entries, self.dim * 2::]

        # interleave rows so context below prediction below input
        A = im_input
        B = im_prediction
        C = im_context
        D = np.empty((A.shape[0] + B.shape[0] + C.shape[0], A.shape[1]))

        D[::3, :] = A
        D[1::3, :] = B
        D[2::3, :] = C
        D = np.reshape(D, (D.shape[0], D.shape[1] / 3, 3))

        im = D
        im = cv2.resize(im, dsize=(0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)

        return im

    def plot_error(self, fig, ax):
        ax.cla()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        ax.set_ylim([0.0, 10.0])
        ax.set_yticks(np.arange(0, 10, 0.5))
        ax.axhline(y=1.5, color='g')
        thing_to_plot = self.mean_error_history[0:self.error_step]
        ax.plot(thing_to_plot, 'b-')
        fig.savefig(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + 'error_history' + '.png', dpi=100)

        ax.cla()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        sorted_net_indices = np.argsort(self.table_use_hist)[::-1]
        ax.bar(np.arange(self.table_use_hist.shape[0]), self.table_use_hist[sorted_net_indices])
        fig.savefig(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + 'table_use_hist' + '.png', dpi=100)

        ax.cla()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        effectiveness_mean = np.divide(self.effectiveness_sum, self.effectiveness_num)
        sorted_effectiveness = np.argsort(effectiveness_mean)[::-1]
        ax.bar(np.arange(effectiveness_mean.shape[0]), effectiveness_mean[sorted_effectiveness])
        fig.savefig(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + 'effectiveness_hist' + '.png', dpi=100)

        ax.cla()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        ax.set_xlim([0 - 0.1, self.env_width_height + 0.1])
        ax.set_ylim([0 - 0.1, self.env_width_height + 0.1])
        ax.plot(self.debug_x_y_theta_input[:, 0],  self.debug_x_y_theta_input[:, 1], 'go')
        #ax.plot(self.debug_x_y_theta_output[:, 0],  self.debug_x_y_theta_output[:, 1], 'ro')
        ax.set_title(str(np.count_nonzero(self.debug_x_y_theta_input[:, 0]) * 1.0 / self.debug_x_y_theta_input.shape[0]))
        fig.savefig(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + 'debug_td_info' + '.png', dpi=100)
