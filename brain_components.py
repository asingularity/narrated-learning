import pickle
from PVM.PVM_framework import MLP
import numpy as np
np.set_printoptions(suppress=True)

from fast_save_matrix import savetxt
from knn_parallel import knn_query as knn_parallel_query


def _get_mlp(num_inputs, num_hidden, num_outputs, learning_rate):
    state = {}
    state['layers'] = [
        {'activation': np.zeros((num_inputs + 1,)), 'error': np.zeros((num_inputs + 1,)), 'delta': np.zeros((num_inputs + 1,))},
        {'activation': np.zeros((num_hidden + 1,)), 'error': np.zeros((num_hidden + 1,)), 'delta': np.zeros((num_hidden + 1,))},
        {'activation': np.zeros((num_outputs + 1,)), 'error': np.zeros((num_outputs + 1,)), 'delta': np.zeros((num_outputs + 1,))},
    ]
    state['weights'] = [
        MLP.initialize_weights(np.zeros((num_inputs + 1, num_hidden)), False),
        MLP.initialize_weights(np.zeros((num_hidden + 1, num_outputs)), False)
    ]
    state['beta'] = np.array([1.0])
    state['learning_rate'] = np.array([learning_rate])
    state['momentum'] = np.array([0.5])
    state['mse'] = np.array([0.0])

    return MLP.MLP(state)


class Autoencoder(object):
    def __init__(self, params):

        self.net = _get_mlp(num_inputs=params['num_inputs'],
                            num_hidden=params['num_hidden'],
                            num_outputs=params['num_inputs'],
                            learning_rate=params['learning_rate'])

        self.error_t = None
        self.mean_error_t = None

        self.error_history = None
        self.mean_error_history = None
        self.error_average_steps = None
        self.max_history_length = None

    def initialize_but_keep_nets(self, simulation_params):
        '''
        initializes variables, but keeps networks as they are
        :return:
        '''

        self.error_t = None
        self.mean_error_t = None

        self.error_average_steps = simulation_params['error_average_steps']
        self.max_history_length = simulation_params['max_history_length']

        self.error_history = np.zeros(self.max_history_length)
        self.mean_error_history = np.zeros(self.max_history_length)

    def evaluate_and_store_error(self, net_input, store_error):
        '''
        must evaluate net_output against net_input and store that error
        returns hidden
        :param net_input:
        :return:
        '''

        net_output = self.net.evaluate(net_input)
        hidden = self.net.layers[1]['activation'][:-1].copy()

        if store_error:
            error = net_output - net_input
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

        return hidden

    def train(self, net_input):
        self.net.train(net_input, net_input)

    def get_mean_error_history(self):
        return self.mean_error_history[0:self.mean_error_t]


class StatesHistory(object):
    def __init__(self, params):
        self.max_history_length = params['max_history_length']
        self.states_dim_list = params['states_dim_list']
        self.state_arrays_list = []
        for k in range(len(self.states_dim_list)):
            self.state_arrays_list.append(np.zeros((self.max_history_length, self.states_dim_list[k])).astype(np.float32))
        self.t = 0

    def process_new_states(self, newest_states_list):

        assert len(newest_states_list) == len(self.state_arrays_list), 'Error: invalid states length!'

        state_index = 0
        for state in newest_states_list:
            self.state_arrays_list[state_index][self.t, :] = state[:]
            state_index += 1
        self.t += 1

    def get_state(self, state_index, delay):
        if self.t - 1 - delay >= 0:
            #print 'StatesHistory.get_state: self.t: ', self.t, ', index: ', self.t - 1 - delay
            state = self.state_arrays_list[state_index][self.t - 1 - delay, :]
            return state
        else:
            return None


class MotorHistory(object):
    def __init__(self, params):
        self.max_history_length = params['max_history_length']
        self.dim = params['dim']

        self.motor_array = np.zeros((self.max_history_length, self.dim)).astype(np.float32)
        self.t = 0

    def process_new_motor_command(self, motor_command):
        self.motor_array[self.t, :] = motor_command[:]
        self.t += 1

    def get_sequence(self, delay_start, delay_end):
        assert delay_start >= delay_end, 'delay_start must be >= delay_end ' + str(delay_start) + ', ' + str(delay_end)
        #print 'MotorHistory.get_sequence: self.t: ', self.t, ', range: [', self.t - 1 - delay_start, ', ', self.t - delay_end, ')'
        return_arr = self.motor_array[self.t - 1 - delay_start:self.t - delay_end, :].flatten()
        return return_arr


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

    def train(self, states_history, training_delay):
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
                self.input_history_t = 0

            self.input_history[self.input_history_t, :] = net_input
            self.output_history[self.input_history_t, :] = net_output
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


class InverseModel(object):
    def __init__(self, params):
        self.state_index_current = params['state_index_current']
        self.state_index_future = params['state_index_future']
        self.dt = params['dt']

        self.input_history = None
        self.output_history = None
        self.input_history_t = None

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

    def _get_current_future_motor(self, states_history, motor_history, training_delay):
        #print '********* START _get_current_future_motor ***********'
        state_add_delay = 1
        # because:
        #   latest motor_command (T) in motor_history was initiated at T-1, applied [T-1, T],
        #   latest state (T) in states_history is at time T

        current_state = states_history.get_state(state_index=self.state_index_current,
                                                 delay=state_add_delay + training_delay + self.dt)
        future_state = states_history.get_state(state_index=self.state_index_future,
                                                delay=state_add_delay + training_delay + 0)
        motor_sequence = motor_history.get_sequence(delay_start=training_delay + self.dt,
                                                    delay_end=training_delay + 1)

        #print 'current state: delay: ', training_delay + self.dt
        #print 'future state: delay: ', training_delay + 0
        #print 'motor_sequence: delay_start: ', training_delay + self.dt, ', delay_end: ', training_delay + 1

        #print '********* END _get_current_future_motor ***********'
        # current state: delay:  129
        # future state: delay:  128
        # motor_sequence: delay_start:  129, delay_end:  129

        return current_state, future_state, motor_sequence

    def train(self, states_history, motor_history, training_delay):
        assert training_delay > self.dt, \
            'training_delay must be greater than self.dt ' + str(training_delay) + str(self.dt)

        current_state, future_state, motor_sequence = self._get_current_future_motor(states_history=states_history,
                                                                                     motor_history=motor_history,
                                                                                     training_delay=training_delay)

        if current_state is not None:
            net_input = np.concatenate((current_state, future_state))
            net_output = motor_sequence

            if self.input_history is None:
                self.input_history = np.zeros((self.max_history_length, net_input.shape[0])).astype(np.float32)
                self.output_history = np.zeros((self.max_history_length, net_output.shape[0])).astype(np.float32)
                self.input_history_t = 0

            self.input_history[self.input_history_t, :] = net_input
            self.output_history[self.input_history_t, :] = net_output
            self.input_history_t += 1

    def lookup_motor_to_goal(self, goal_states, states_history):

        data_set = self.input_history
        data_frames = self.input_history_t
        dim = data_set.shape[1]
        tmp = self.temp_array

        current_state = states_history.get_state(state_index=self.state_index_current,
                                                 delay=0)

        net_input = np.concatenate((current_state, goal_states[self.state_index_future].astype(np.float32)))
        dist, ind = knn_parallel_query(data_set, net_input, tmp, data_frames, dim)
        net_output_predicted = self.output_history[ind, :]
        #print 'lookup_motor_to_goal: net_output_predicted ', net_output_predicted
        return net_output_predicted

    def test_newest_point_and_store_error(self, states_history, motor_history):
        if self.input_history_t is not None:
            data_set = self.input_history
            data_frames = self.input_history_t
            dim = data_set.shape[1]
            tmp = self.temp_array

            current_state, future_state, motor_sequence = self._get_current_future_motor(states_history=states_history,
                                                                                         motor_history=motor_history,
                                                                                         training_delay=0)

            if current_state is not None:
                net_input = np.concatenate((current_state, future_state))
                net_output_actual = motor_sequence

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

    # TODO for inverse model, mean error histories, add to plots!

