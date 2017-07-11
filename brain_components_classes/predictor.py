import pickle
from PVM.PVM_framework import MLP
import numpy as np
np.set_printoptions(suppress=True)

from fast_save_matrix import savetxt
from knn_parallel import knn_query as knn_parallel_query


class Predictor(object):
    def __init__(self, params):
        self.dt_output = params['dt_output']
        self.dt_context = params['dt_context']
        self.state_index_input = params['state_index_input']
        self.state_index_context = params['state_index_context']
        self.state_index_output = params['state_index_output']

        self.input_history = None
        self.output_history = None
        self.input_history_t = None

        self.debug_input_td_info_history = None
        self.debug_context_td_info_history = None
        self.debug_output_td_info_history = None

        self.error_t = None
        self.mean_error_t = None

        self.error_history = None
        self.mean_error_history = None
        self.error_average_steps = None
        self.max_history_length = None

        self.temp_array = None

    def initialize_but_keep_nets(self, simulation_params):

        self.error_t = None
        self.mean_error_t = None

        # TODO fix conflict between max_history_length (for error history here) vs. same variable for input_history
        self.error_average_steps = simulation_params['error_average_steps']
        self.max_history_length = simulation_params['max_history_length']

        self.error_history = np.zeros(self.max_history_length)
        self.mean_error_history = np.zeros(self.max_history_length)

        # TODO fix conflict between max_history_length (for error history here) vs. same variable for input_history
        self.temp_array = np.zeros(self.max_history_length).astype(np.float32)

        if self.input_history is not None:
            self.input_history = self.input_history.astype(np.float32)
        if self.output_history is not None:
            self.output_history = self.output_history.astype(np.float32)

    def _get_input_context_output(self, states_history, training_delay):
        input_state = states_history.get_state(state_index=self.state_index_input, delay=training_delay + self.dt_context)
        context_state = states_history.get_state(state_index=self.state_index_context, delay=training_delay + 0)
        output_state = states_history.get_state(state_index=self.state_index_output, delay=training_delay + self.dt_context - self.dt_output)
        return input_state, context_state, output_state

    def _get_td_info_input_context_output(self, debug_topdown_info_history, training_delay):
        input_td_info = debug_topdown_info_history.get_td_info(delay=training_delay + self.dt_context)
        context_td_info = debug_topdown_info_history.get_td_info(delay=training_delay + 0)
        output_td_info = debug_topdown_info_history.get_td_info(delay=training_delay + self.dt_context - self.dt_output)
        return input_td_info, context_td_info, output_td_info

    def train(self, states_history, training_delay, debug_topdown_info_history):
        '''
        predictor must decide if it has enough history to train
        training: sets part of proper states_history array to its own knn data
        :param states_history:
        :param training_delay:
        :return:
        '''

        assert training_delay > self.dt_context, \
            'training_delay must be greater than self.dt_context ' + str(training_delay) + str(self.dt_context)
        assert self.dt_context > self.dt_output, 'dt_context must be > dt_output'

        input_state, context_state, output_state = self._get_input_context_output(states_history=states_history,
                                                                                  training_delay=training_delay)

        if input_state is not None:
            net_input = np.concatenate((input_state, context_state))
            net_output = output_state

            if self.input_history is None:
                self.input_history = np.zeros((self.max_history_length, net_input.shape[0])).astype(np.float32)
                self.output_history = np.zeros((self.max_history_length, net_output.shape[0])).astype(np.float32)

                # hard-coded 3: robot_x, robot_y, robot_z
                self.debug_input_td_info_history = np.zeros((self.max_history_length, 3)).astype(np.float32)
                self.debug_context_td_info_history = np.zeros((self.max_history_length, 3)).astype(np.float32)
                self.debug_output_td_info_history = np.zeros((self.max_history_length, 3)).astype(np.float32)

                self.input_history_t = 0


            self.input_history[self.input_history_t, :] = net_input
            self.output_history[self.input_history_t, :] = net_output

            # TODO get robot position & angle for input, context, and output
            input_td_info, context_td_info, output_td_info = self._get_td_info_input_context_output(debug_topdown_info_history=debug_topdown_info_history,
                                                                                                    training_delay=training_delay)
            # TODO store here
            self.debug_input_td_info_history[self.input_history_t, :] = input_td_info
            self.debug_context_td_info_history[self.input_history_t, :] = context_td_info
            self.debug_output_td_info_history[self.input_history_t, :] = output_td_info

            self.input_history_t += 1

    #@profile
    def optimize_train(self, states_history, training_delay):
        '''
        Overwrites output state for closest entry to new (input, context), if output state is more "efficient"
        :param states_history:
        :param training_delay:
        :return:
        '''

        assert training_delay > self.dt_context, \
            'training_delay must be greater than self.dt_context ' + str(training_delay) + str(self.dt_context)
        assert self.dt_context > self.dt_output, 'dt_context must be > dt_output'

        input_state, context_state, output_state = self._get_input_context_output(states_history=states_history,
                                                                                  training_delay=training_delay)

        if input_state is not None:

            net_output_predicted, dist, index = self.predict(input_state=input_state,
                                                             context_state=context_state)

            # replace based on minimum distance:
            #   |(output_predicted) - input_state| + |(output_predicted) - context_state|
            #       vs.
            #   |(output_actual) - input_state| + |(output_actual) - context_state|
            # problem: input state, or context state, not same dimensionality (necessarily) as output state...
            #   1. we do it anyway for now since it can be the same
            #   2. we only take distance vs. input_state, assuming predictor output is always same dim as input
            #           (always predicting its own input)

            distance_predicted = np.mean(np.fabs(net_output_predicted - input_state)) + np.mean(np.fabs(net_output_predicted - context_state))
            distance_actual = np.mean(np.fabs(output_state - input_state)) + np.mean(np.fabs(output_state - context_state))

            if distance_actual < distance_predicted:

                print 'replacing: (actual < predicted) ', distance_actual, ' < ', distance_predicted
                # replace only output (near term prediction)
                net_output = output_state
                self.output_history[index, :] = net_output
            else:
                pass
                print 'not replacing: (actual > predicted) ', distance_actual, ' > ', distance_predicted

    def predict(self, input_state, context_state):
        net_input = np.concatenate((input_state, context_state)).astype(np.float32)
        data_set = self.input_history
        data_frames = self.input_history_t
        dim = data_set.shape[1]
        tmp = self.temp_array

        dist, ind = knn_parallel_query(data_set, net_input, tmp, data_frames, dim)
        net_output_predicted = self.output_history[ind, :]
        return net_output_predicted, dist, ind

    def predict_and_get_debug_td_info(self, input_state, context_state):
        net_output_predicted, dist, ind = self.predict(input_state, context_state)
        input_td_info = self.debug_input_td_info_history[ind, :]
        context_td_info = self.debug_context_td_info_history[ind, :]
        output_td_info = self.debug_output_td_info_history[ind, :]
        return net_output_predicted, dist, ind, input_td_info, context_td_info, output_td_info

    def test_newest_point_and_store_error(self, states_history):
        '''
        predictor must decide if it has enough history to test

        :param states_history:
        :param training_delay:
        :return:
        '''

        if self.input_history_t is not None:
            data_set = self.input_history
            data_frames = self.input_history_t
            dim = data_set.shape[1]
            tmp = self.temp_array

            input_state, context_state, output_state = self._get_input_context_output(states_history=states_history,
                                                                                      training_delay=0)

            if input_state is not None:
                net_input = np.concatenate((input_state, context_state))
                net_output_actual = output_state

                dist, ind = knn_parallel_query(data_set, net_input, tmp, data_frames, dim)
                net_output_predicted = self.output_history[ind, :]

                error = net_output_actual - net_output_predicted
                error = np.mean(np.fabs(error))

                if self.error_t is None:
                    self.error_t = 0

                self.error_history[self.error_t] = error
                self.error_t += 1

                if self.error_t > self.error_average_steps:
                    if self.mean_error_t is None:
                        self.mean_error_t = 0
                    mean_error = np.mean(self.error_history[self.error_t - self.error_average_steps:self.error_t])
                    self.mean_error_history[self.mean_error_t] = mean_error
                    self.mean_error_t += 1

    def get_mean_error_history(self):
        return self.mean_error_history[0:self.mean_error_t]

