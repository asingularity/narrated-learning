import pickle
from PVM.PVM_framework import MLP
import numpy as np
np.set_printoptions(suppress=True)

from fast_save_matrix import savetxt
from knn_parallel import knn_query as knn_parallel_query


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

