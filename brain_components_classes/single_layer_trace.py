import time
import cv2
import numpy as np
import random
import pickle

from math import sin, cos
from cuda_dist_query import CudaTable
from utils.load_sim_history import load_states_history, load_td_info_history
from utils.fps_counter import FPSCounter
from brain_components_classes.states_history import StatesLimitedHistory

NL_SIM_DIR = '/srv/projects/NL-sim/'


class SingleLayerTrace(object):
    def __init__(self, params):
        self.input_dim = params['input_dim']
        goal_context_dim = params['goal_context_dim']
        self.goal_context_dim = goal_context_dim
        self.entries = params['entries']
        predict_time_per_layer = params['predict_time_per_layer']
        self.predict_time = predict_time_per_layer[0]
        self.learning_enabled = params['enable_learning']
        table_learn_time = params['table_learn_time']
        prediction_learn_time = params['prediction_learn_time']
        pre_init_goal_contexts = params['pre_init_goal_contexts']
        max_history_length = params['max_history_length']

        n_layers = len(predict_time_per_layer)
        assert n_layers == 1, 'this is a single layer predictor!'

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
                                      input_dim=self.input_dim)

        self.init_I_row_num = None

        self.trace_tau = 0.9
        self.max_trace_time = 100  # length of keeping track of trace for W.
        # so, minimal possible trace value in self.W is = pow(self.trace_tau, self.max_trace_time)

        self.goal_context_values = pre_init_goal_contexts

        # print(goal_context_dim, self.goal_context_values)
        # 1 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # need a place to store: the set of input indices, corresponding to each goal index (include null set?)
        # assume that its possible to assign more than one label to a particular state

        self.goal_table = np.zeros((self.entries, len(self.goal_context_values))).astype(np.int)

        # since we are using cuda table: need to define history of input and context also
        # might this be easier by using a wrapper class like single_multi_context_layer?
        # really its just: history of input and one-hot encoded input
        self._init_histories()

        # why * 2? store linear velocity, angular velocity - so two values per step
        self.store_motor_steps = predict_time_per_layer[0]
        self.motor_table = np.zeros((self.entries, self.entries, self.store_motor_steps, 2), np.float)

        self.entries_x_y_theta_input = np.zeros((self.entries, 3), np.float)

        self.t = 0

        # matrix that stores the trace value (current, future), 1.0 if it is next step, 0.0 if beyond max timescale
        self.W = np.zeros((self.entries, self.entries), np.float)

    def _init_histories(self):

        max_trace_time = self.max_trace_time
        max_predict_time = self.predict_time

        # to do later - this is not used right now at all - remove?
        self.input_history = StatesLimitedHistory(params={'max_delay': max_predict_time,
                                                          'states_dim_list': [self.input_dim]})

        # we don't need this- only need a 1D vector to store actual input index from [0, self.entries - 1] on each time step
        # self.I_history = StatesLimitedHistory(params={'max_delay': max_predict_time,
        #                                               'states_dim_list': [self.entries]})

        self.I_history = StatesLimitedHistory(params={'max_delay': max_trace_time,
                                                      'states_dim_list': [1]})

        self.motor_history = StatesLimitedHistory(params={'max_delay': max_predict_time,
                                                          'states_dim_list': [2]})

        self.goal_context_history = StatesLimitedHistory(params={'max_delay': max_predict_time,
                                                                 'states_dim_list': [self.goal_context_dim]})

    def disable_learning(self):
        self.learning_enabled = False

    def get_current_plan(self):
        # to do later what for this predictor is the current plan sequence we should display?
        self.plan_I_seq = []
        return self.plan_I_seq, self.entries_x_y_theta_input

    def _lookup_and_learn_table(self, input_state, learn_table, input_x_y_theta):
        '''

        why is lookup and learn one function?
            if they were separate, we would have to do same lookup twice, or pass info in an awkward way

        :param input_state:
        :param learn_table:
        :return: I (sparse vector, one-hot)
        '''

        dists = self.cuda_table_I.query(query_input=input_state)

        I_index = np.argmin(dists)
        if learn_table:
            self._learn_table(dists=dists, input_state=input_state, input_x_y_theta=input_x_y_theta)

        return I_index  # learning might have invalidated this; doesn't matter for now because for now we are not using lookup if learning

    def _learn_table(self, dists, input_state, input_x_y_theta):
        '''

        :param dists:
        :param input_state:
        :return:
        '''

        # to do later on replacement or learning: should zero out W from that row for all predictions
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
                                                 row_to_table_dists=dists)

                if input_x_y_theta is not None:
                    self.entries_x_y_theta_input[r_r_ind, :] = input_x_y_theta[:]

                row_replaced = True

        return row_replaced

    def _learn_predictions_and_motor(self):
        current_goal_context, _ = self.goal_context_history.get_state(state_index=0, delay=0)
        assert len(current_goal_context) == 1
        assert current_goal_context.shape[0] == 1

        # print('current_goal_context:', current_goal_context)
        # current_goal_context: [0.]
        # current_goal_context: [3.]

        # *** self.W learning ***

        current_I_index_arr, _ = self.I_history.get_state(state_index=0, delay=0)
        current_I_index = int(current_I_index_arr[0])

        # minimal possible trace value in self.W is = pow(self.trace_tau, self.max_trace_time)
        to_index = current_I_index

        # to do later speed up this function
        from_index_arr = self.I_history.get_state_sequence(state_index=0,
                                                           delay_long=self.max_trace_time,
                                                           delay_short=1).flatten().astype(np.int)

        trace_value_arr = np.power(self.trace_tau, np.arange(self.max_trace_time - 1, -1, -1))

        self.W[to_index, from_index_arr] = np.maximum(trace_value_arr, self.W[to_index, from_index_arr])

        # self.goal_table = np.zeros((self.entries, len(self.goal_context_values))).astype(np.int)

        self.goal_table[current_I_index, current_goal_context[0]] = 1

        # *** motor learning ***
        # motor table is:
        # [entries (now) X entries (future) X motor commands width X prediction length (first layer)]

        #   since motor is "last motor command" i.e. what got us to same time's input state,
        #   get "last_motor_command" corresponding to output state (after predict_time)
        # we want motor table to store: what got us from delay=predict_time, to delay=0

        from_index = from_index_arr[-1]

        motor_seq, _ = self.motor_history.get_state(state_index=0,
                                                    delay=0)

        self.motor_table[from_index, to_index, :, :] = motor_seq

    def _get_motor_for_task(self, goal_context_state_task):
        '''

        :param goal_context_state_task: i.e. [3.]
        :return: motor_out
        '''

        goal_index = int(goal_context_state_task[0])

        current_I_index_arr, _ = self.I_history.get_state(state_index=0, delay=0)
        current_I_index = int(current_I_index_arr[0])

        # (1) look up all states possible in next step (W=1 from current state index) -> array I_next

        I_next = np.nonzero(self.W[:, current_I_index] == 1)[0]
        # print('I_next', I_next)

        if len(I_next) == 0:
            print('no path found!')
            motor_out = None
            return motor_out

        # (2) look up all states corresponding to current goal context -> array I_goal

        I_goal = np.nonzero(self.goal_table[:, goal_index] == 1)[0]
        # print('I_goal', I_goal)

        # (3) find W from every I_next to every I_goal -> W_next_to_goal

        W_next_to_goal = np.zeros((self.entries, self.entries), np.float)
        # need to meshgrid ...
        gv, nv = np.meshgrid(I_goal, I_next)
        W_next_to_goal[gv, nv] = self.W[gv, nv]

        # (4) take max W -> select I_next_step corresponding to I_next for argmax(W_next_to_goal)
        I_goal_max, I_next_max = np.unravel_index(np.argmax(W_next_to_goal), W_next_to_goal.shape)

        # *** PROBLEM *** for 200 entries, this W is always 1.0 *** PROBLEM ***
        print('I_goal_max:', I_goal_max, ', I_next_max:', I_next_max, ', W:', W_next_to_goal[I_goal_max, I_next_max])

        # (4) motor lookup based on current_I_index, I_next_step
        motor_seq = self.motor_table[current_I_index, I_next_max, :, :].flatten()
        motor_out = motor_seq

        if I_goal_max == 0 and I_next_max == 0 and W_next_to_goal[I_goal_max, I_next_max] == 0.0:
            print()
            print('******* Bad State ******')
            print('    no path found?')
            print()
            print('len(I_goal)', len(I_goal), 'len(I_next)', len(I_next))
            print()
            print('np.count_nonzero(W_next_to_goal)', np.count_nonzero(W_next_to_goal))
            print()
            print('motor_out', motor_out)
            print()
            print('    setting motor_out to None')
            print('********************')
            print()

            motor_out = None

        return motor_out

    def step(self, raycast_image, input_state, input_x_y_theta, goal_context_state_learning, goal_context_state_task, last_motor_command):
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

        I_index_t = self._lookup_and_learn_table(input_state=input_state, learn_table=learn_table, input_x_y_theta=input_x_y_theta)

        # store states in history

        self.input_history.store_new_states(newest_states_list=[input_state], extra_data_list=[input_x_y_theta])
        self.I_history.store_new_states(newest_states_list=[np.array([I_index_t])], extra_data_list=[input_x_y_theta])
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
            # could still return None if cannot compute a plan!
            #   in which case, agent should apply last motor, or random motor?
            motor_out = self._get_motor_for_task(goal_context_state_task=goal_context_state_task)
        else:
            motor_out = None

        self.t += 1
        return motor_out

    def get_table_ims(self):
        W_im_list = [self._get_W_im(W=self.W)]
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

        D = (W * 255.0).astype(np.uint8)

        max_dim = max(D.shape[0], D.shape[1])

        imscale = 500. / max_dim  # 0.2: full table, 2.0
        # imscale = 5.0
        im = cv2.resize(D, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        return im



































