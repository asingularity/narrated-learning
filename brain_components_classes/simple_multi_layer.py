import time
import cv2
import numpy as np
import random
import pickle
from tabulate import tabulate

from math import sin, cos
from cuda_dist_query import CudaTable
from utils.load_sim_history import load_states_history, load_td_info_history
from utils.fps_counter import FPSCounter
from brain_components_classes.states_history import StatesLimitedHistory

NL_SIM_DIR = '/srv/projects/NL-sim/'


class SimpleMultiLayer(object):
    '''
        This is a simplification of MultiLayerEnsemble
        It assumes there is only one table, and does planning given multiple "virtual" layers of that one table
        i.e. notes from 8/4/19
    '''
    def __init__(self, params):
        '''

        :param params:
        {
            input_dim
            goal_context_dim
            entries
            predict_time_per_layer
            enable_learning
            table_learn_time
            prediction_learn_time
            pre_init_goal_contexts
            max_history_length
        }
        '''

        self.input_dim = params['input_dim']
        goal_context_dim = params['goal_context_dim']
        self.goal_context_dim = goal_context_dim
        self.entries = params['entries']
        predict_time_per_layer = params['predict_time_per_layer']
        self.predict_time_per_layer = predict_time_per_layer
        self.learning_enabled = params['enable_learning']
        table_learn_time = params['table_learn_time']
        prediction_learn_time = params['prediction_learn_time']
        pre_init_goal_contexts = params['pre_init_goal_contexts']
        max_history_length = params['max_history_length']

        self.n_layers = len(predict_time_per_layer)

        self.start_table_learn_t = 2  # why not 0? to have a valid last motor command in the system?
        self.stop_table_learn_t = self.start_table_learn_t + table_learn_time

        # why + 100 below? so that recent states history (for predict learn) includes any last table row replaces
        self.start_prediction_learn_t = self.stop_table_learn_t + 100
        self.stop_prediction_learn_t = self.start_prediction_learn_t + prediction_learn_time

        print()
        print('table learn [start, stop]:', '[' + str(self.start_table_learn_t) + ', ' + str(self.stop_table_learn_t) + ']')
        print('prediction learn [start, stop]:', '[' + str(self.start_prediction_learn_t) + ', ' + str(self.stop_prediction_learn_t) + ']')
        print()

        assert self.stop_prediction_learn_t < max_history_length, 'max_history_length too short!, learn stop time. '\
                                                                  + str((max_history_length,
                                                                         self.stop_prediction_learn_t))

        self.cuda_table_I = CudaTable(num_entries=self.entries,
                                      input_dim=self.input_dim,
                                      output_dim=0,
                                      context_dim=0,
                                      include_layers=['i_only'])

        self.init_I_row_num = None

        self.W_by_layer = {}
        for k in range(self.n_layers - 1):  # why -1? because goal context W counts as a separate "layer"
            # TODO these need to be better represented in a sparse way, on cuda
            self.W_by_layer[k] = np.zeros((self.entries, self.entries), np.int)

        self.W_goal = np.zeros((int(np.amax(pre_init_goal_contexts)), self.entries), np.int)
        self.goal_context_values = pre_init_goal_contexts

        # print(goal_context_dim, self.goal_context_values)
        # 1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # since we are using cuda table: need to define history of input and context also
        # might this be easier by using a wrapper class like single_multi_context_layer?
        # really its just: history of input and one-hot encoded input
        self._init_histories()

        # why * 2? store linear velocity, angular velocity - so two values per step
        self.store_motor_steps = predict_time_per_layer[0]
        self.motor_table = np.zeros((self.entries, self.entries, self.store_motor_steps, 2), np.float)

        # stats
        self.stat_IO_row_replaces = np.zeros(self.n_layers, np.int)
        self.stat_C_row_replaces = np.zeros(self.n_layers, np.int)

        self.stat_disp_last_time = time.time()
        self.stat_disp_interval = 10

        self.entries_x_y_theta_input = np.zeros((self.entries, 3), np.float)

        self.t = 0

        self.predict_time_per_layer = np.array(params['predict_time_per_layer'])

        self.plan_I_seq = None

    def _init_histories(self):
        # TODO must determine max delay here

        max_predict_time = np.amax(self.predict_time_per_layer)

        self.input_history = StatesLimitedHistory(params={'max_delay': max_predict_time,
                                                          'states_dim_list': [self.input_dim]})

        self.I_history = StatesLimitedHistory(params={'max_delay': max_predict_time,
                                                      'states_dim_list': [self.entries]})

        self.motor_history = StatesLimitedHistory(params={'max_delay': max_predict_time,  # TODO fix here
                                                          'states_dim_list': [2]})

        self.goal_context_history = StatesLimitedHistory(params={'max_delay': max_predict_time,
                                                            'states_dim_list': [self.goal_context_dim]})

    def disable_learning(self):
        self.learning_enabled = False

    def _lookup_and_learn_table(self, input_state, learn_table, input_x_y_theta):
        '''

        why is lookup and learn one function?
            if they were separate, we would have to do same lookup twice, or pass info in an awkward way

        :param input_state:
        :param learn_table:
        :return: I (sparse vector, one-hot)
        '''

        dists = self.cuda_table_I.query(query_input=input_state,
                                        query_output=None,
                                        query_context=None)

        I = np.zeros(self.entries, np.int)
        I[np.argmin(dists)] = 1

        if learn_table:
            self._learn_table(dists=dists, input_state=input_state, input_x_y_theta=input_x_y_theta)

        return I  # learning might have invalidated this; doesn't matter for now because for now we are not using lookup if learning

    def _learn_table(self, dists, input_state, input_x_y_theta):
        '''

        :param dists:
        :param input_state:
        :return:
        '''

        # TODO on replacement or learning: should zero out W from that row for all predictions
        #   why don't we do this now?
        #   because, for now, we learn table, then we leave it alone when learning predictions later

        row_replaced = False

        assert input_state.shape[0] == self.input_dim

        sorted_dist_indices = np.argsort(dists)
        new_min_ind = sorted_dist_indices[0]
        new_min_dist = dists[new_min_ind]

        if self.init_I_row_num is None:
            self.init_I_row_num = 0

        if self.init_I_row_num < self.cuda_table_I.get_num_rows():
            # necessary so dist matrix helper is not so slow at start
            dists[self.init_I_row_num] = np.inf
            try:
                self.cuda_table_I.set_matrix_row(row_index=self.init_I_row_num,
                                                 row_input=input_state,
                                                 row_output=None,
                                                 row_context=None,
                                                 row_to_table_dists=dists,
                                                 fast_init=True)
            except AssertionError:
                print('\nError! Invalid GPU data type. input_state.dtype: ' + str(input_state.dtype) + '\n')
                raise

            # This only really makes sense for the first layer
            if input_x_y_theta is not None:
                self.entries_x_y_theta_input[self.init_I_row_num, :] = input_x_y_theta[:]

            row_replaced = True
            self.init_I_row_num += 1
        else:
            if not self.cuda_table_I.post_init_done:
                self.cuda_table_I.post_init()

            table_min_dist, table_min_dist_r, table_min_dist_c = self.cuda_table_I.get_min_dist()

            if new_min_dist > table_min_dist:
                # minimum distance of new row to current rows is greater than current minimum row-row distance
                # so: replace one row of current minimum, with new row

                # get one of the row indices of current minimum dist pair
                r_r_ind = table_min_dist_r  # could be table_min_dist_c

                dists[r_r_ind] = np.inf

                # replace the current min dist row, with the new row
                self.cuda_table_I.set_matrix_row(row_index=r_r_ind,
                                                 row_input=input_state,
                                                 row_output=None,
                                                 row_context=None,
                                                 row_to_table_dists=dists)

                if input_x_y_theta is not None:
                    self.entries_x_y_theta_input[r_r_ind, :] = input_x_y_theta[:]

                row_replaced = True

        return row_replaced

    def _learn_predictions_and_motor(self):
        '''

        updates:
            self.W_by_layer
            self.W_goal
        uses:
            self.predict_time_by_layer
            self.I_history
            self.motor_history
            self.goal_context_history
            self.goal_context_values

        :param at_goal_context_state_now:
        :return:
        '''

        for k in range(self.n_layers - 1):  # why -1? because goal context W counts as a separate "layer"
            predict_time = self.predict_time_per_layer[k]

            train_I_now, train_input_x_y_theta_now = self.I_history.get_state(state_index=0, delay=predict_time)
            train_I_future, train_input_x_y_theta_future = self.I_history.get_state(state_index=0, delay=0)

            in_entries = np.nonzero(train_I_now)
            out_entries = np.nonzero(train_I_future)

            self.W_by_layer[k][out_entries, in_entries] = 1

            if k == 0:
                # motor learning
                # motor table is:
                # [entries (now) X entries (future) X motor commands width X prediction length (first layer)]

                # TODO verify this is correct time alignment. i.e. delay_long, delay_short set correctly below?
                #   should it be predict_time + 1 instead?
                #   from earlier code:
                #   since motor is "last motor command" i.e. what got us to same time's input state,
                #   get "last_motor_command" corresponding to output state (after predict_time)
                # we want motor table to store: what got us from delay=predict_time, to delay=0

                motor_seq = self.motor_history.get_state_sequence(state_index=0,
                                                                  delay_long=predict_time - 1,
                                                                  delay_short=0)
                try:
                    self.motor_table[in_entries, out_entries, :, :] = motor_seq
                except:
                    '''
                        self.store_motor_steps = predict_time_per_layer[0]
                        self.motor_table = np.zeros((self.entries, self.entries, self.store_motor_steps, self.store_motor_steps), np.float)

                        ERROR!

                        motor_seq (1, 2) motor_table (400, 400, 1, 1)

                        fix:
                        self.motor_table = np.zeros((self.entries, self.entries, self.store_motor_steps, 2), np.float)

                    '''
                    print()
                    print('ERROR!')
                    print()
                    print('motor_seq', motor_seq.shape, 'motor_table', self.motor_table.shape)
                    print()
                    raise

        goal_predict_time = self.predict_time_per_layer[self.n_layers - 1]

        train_I_now, _ = self.I_history.get_state(state_index=0, delay=goal_predict_time)
        goal_context_future, _ = self.goal_context_history.get_state(state_index=0, delay=0)

        in_entries = np.nonzero(train_I_now)
        assert len(goal_context_future) == 1
        assert goal_context_future.shape[0] == 1

        # TODO init W_goal to be appropriate: max goal value
        self.W_goal[goal_context_future[0], in_entries] = 1

    def _get_motor_for_task(self, goal_context_state_task):
        '''

        n_layers = 4
        list(range(n_layers - 1))
            [0, 1, 2]
        list(range(n_layers - 2, -1, -1))
            [2, 1, 0]

        :param goal_context_state_task:
        :return:
        '''

        debug_print = False

        # print('goal_context_state_task:', goal_context_state_task)  # i.e. [3]
        # do planning here, and display plan as it is being iteratively planned

        # (1) do forward pass from current I, and show "matched rows" number per layer

        I_seq = []  # store sequence of I
        x_y_theta_seq = []  # store sequence of x_y_theta

        (I, x_y_theta) = self.I_history.get_state(state_index=0, delay=0)

        I_in = I.copy()
        I_seq.append(I)  # I_seq[0]

        for k in range(self.n_layers - 1):  # why -1? because goal context W counts as a separate "layer"
            W_layer = self.W_by_layer[k]  # W[future, past]

            # get predicted I (with more than one nonzero) from W_layer, and I
            I_next = np.dot(W_layer, I_in)
            I_next[I_next > 1.0] = 1.0  # there may be multiple ways to predict a particular row

            # print(k, np.count_nonzero(I_next), len(I_next), np.amax(I_next))

            # then, set I to prediction for next layer
            I_in = I_next.copy()
            I_seq.append(I_in)  # I_seq[k + 1]

        I_next = np.dot(self.W_goal, I_in)  # goal
        I_next[I_next > 1.0] = 1.0  # there may be multiple ways to predict a particular row
        I_seq.append(I_next)  # I_seq[n_layers]

        # print('INFO', len(I_seq), self.n_layers)  # INFO 5 4 -
        # why 5 > 4? sequence includes input, and then output of each predictive layer

        if debug_print:
            print()
            print('PLAN')
            print('fwd predictions:')
            plan_str = ''
            for k in range(len(I_seq)):
                plan_str += '[' + str(np.count_nonzero(I_seq[k])) + '] '
            print(plan_str)

        # (2) do backward pass from goal state and compute AND, and show new filtered "matched rows" number per layer
        goal_I = np.zeros_like(I_next)
        goal_I[goal_context_state_task[0]] = 1.0
        I_seq[self.n_layers] = np.logical_and(goal_I, I_seq[self.n_layers])
        goal_I = I_seq[self.n_layers]

        I_in = np.dot(goal_I, self.W_goal)  # is this right? what I_in's predict goal_I

        for k in range(self.n_layers - 2, -1, -1):
            # compute AND of I_in, corresponding I_seq that's already stored
            I_seq[k+1] = np.logical_and(I_seq[k+1], I_in)
            I_in = I_seq[k+1]

            W_layer = self.W_by_layer[k]  # W[future, past]
            I_in = np.dot(I_in, W_layer)

        # don't really need this- this is current input? we care about next prediction onward only
        I_seq[0] = np.logical_and(I_seq[0], I_in)

        # (3) show plan on map (position and angle sequence)
        if debug_print:
            print('after planning:')
            plan_str = ''
            for k in range(len(I_seq)):
                plan_str += '[' + str(np.count_nonzero(I_seq[k])) + '] '
            print(plan_str)
            print()

        if np.count_nonzero(I_seq[1]) == 0:
            self.plan_I_seq = None  # no plan found!
            in_entries = None  # just for display
            out_entries = None  # just for display
            # TODO if no plan found: should it return previous motor command? This means random motor.
            motor_out = None
        else:
            self.plan_I_seq = I_seq

            in_entries = np.nonzero(I_seq[0])[0]
            # i0 = 0
            i0 = random.randint(0, len(in_entries) - 1)
            in_entries = in_entries[i0]

            out_entries = np.nonzero(I_seq[1])[0]
            # i1 = 0
            i1 = random.randint(0, len(out_entries) - 1)
            out_entries = out_entries[i1]

            motor_seq = self.motor_table[in_entries, out_entries, :, :].flatten()
            # print(motor_seq.shape, motor_seq)
            motor_out = motor_seq

        print()
        print('in_entries:', in_entries, 'out_entries:', out_entries)
        print('motor_out:', motor_out)
        print()

        return motor_out

    def get_current_plan(self):
        return self.plan_I_seq, self.entries_x_y_theta_input

    def step(self, input_state, input_x_y_theta, goal_context_state_learning, goal_context_state_task, last_motor_command):
        '''

        :param input_state:
        :param input_x_y_theta:
        :param goal_context_state_learning:
        :param goal_context_state_task:
        :param last_motor_command:
        :return:
        '''

        # get current table lookup (sparse, one-hot) and learn table if learning is on

        learn_table = self.learning_enabled and (self.start_table_learn_t < self.t < self.stop_table_learn_t)

        I_t = self._lookup_and_learn_table(input_state=input_state, learn_table=learn_table, input_x_y_theta=input_x_y_theta)

        # store states in history

        self.input_history.store_new_states(newest_states_list=[input_state], extra_data_list=[input_x_y_theta])
        self.I_history.store_new_states(newest_states_list=[I_t], extra_data_list=[input_x_y_theta])
        self.goal_context_history.store_new_states(newest_states_list=[goal_context_state_learning], extra_data_list=[None])
        self.motor_history.store_new_states(newest_states_list=[last_motor_command], extra_data_list=[None])

        # handle prediction learning

        if self.learning_enabled and (self.start_prediction_learn_t < self.t < self.stop_prediction_learn_t):
            assert goal_context_state_learning is not None

            # uses self.input_history, self.I_history, self.goal_context_history, self.motor_history
            # learns all prediction matrices W, motor_table (from single_multi_context_layer.py)
            self._learn_predictions_and_motor()

        # handle prediction for task
        # we are in task mode if goal_context_state_task is not None
        if goal_context_state_task is not None:
            # does prediction across layers, computing overlap, getting motor from predicted state
            # could still return None if cannot coompute a plan!
            #   in which case, agent should apply last motor, or random motor?
            motor_out = self._get_motor_for_task(goal_context_state_task=goal_context_state_task)
        else:
            motor_out = None

        self.t += 1
        return motor_out

    def get_table_ims(self):
        # TODO in task mode: show planned trajectories, decisions per layer during settling process
        # TODO allow multiple planning steps per task step

        W_im_list = []
        for k in range(self.n_layers - 1):  # why -1? because goal context W counts as a separate "layer"
            W_im_list.append(self._get_W_im(W=self.W_by_layer[k]))

        W_im_list.append(self._get_W_im(W=self.W_goal))

        return self._get_I_im(), W_im_list

    def save_to_pkl(self, file_path, file_name):

        pass

    def init_after_load(self):
        self.cuda_table_I.init_after_load()

    def _get_table_im(self, cuda_table, layer_index=0):

        # layer 0: display as color images below
        # layer 1...N-1: display as grayscale [0, 1] values?

        input_dim = cuda_table.input_dim

        table = np.transpose(cuda_table.get_table_from_gpu())

        if layer_index == 0:
            entries = 40 * 1 * 3
        else:
            entries = table.shape[0]

        im_input = table[0:entries, 0:input_dim]

        if layer_index == 0:
            A = im_input
            C = A

            im = np.reshape(C, (C.shape[0], C.shape[1] / 3, 3))

            im = cv2.resize(im, dsize=(0,0), fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        else:
            # A = im_input
            # B = im_prediction
            # C = im_context
            # D = np.hstack((A, B, C))
            D = table

            max_dim = max(D.shape[0], D.shape[1])

            imscale = 500. / max_dim  # 0.2: full table, 2.0
            # imscale = 5.0
            im = cv2.resize(D, dsize=(0,0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        return im

    def _get_I_im(self):
        return self._get_table_im(cuda_table=self.cuda_table_I, layer_index=0)

    def _get_W_im(self, W):
        if W is None:
            return None

        D = W.astype(np.uint8) * 255

        max_dim = max(D.shape[0], D.shape[1])

        imscale = 500. / max_dim  # 0.2: full table, 2.0
        # imscale = 5.0
        im = cv2.resize(D, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        return im









































