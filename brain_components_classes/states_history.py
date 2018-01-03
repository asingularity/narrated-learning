
import pickle
from PVM.PVM_framework import MLP
import numpy as np
np.set_printoptions(suppress=True)

from fast_save_matrix import savetxt
from knn_parallel import knn_query as knn_parallel_query


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

    def save_states(self, plots_save_folder, state_indices_list):
        for state_index in state_indices_list:
            f = open(plots_save_folder + '/states_history_' + str(state_index) + '.pkl', 'w')
            pickle.dump(self.state_arrays_list[state_index], f)
            f.close()
