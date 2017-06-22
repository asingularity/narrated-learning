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
