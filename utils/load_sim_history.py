
import pickle
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_states_history(sim_load_name):
    data_file = sim_load_name + '/states_history_0.pkl'

    print('loading states history...')
    f = open(data_file, 'rb')
    states_history = pickle.load(f, encoding='latin1')  # latin1 needed for python 2 pickle
    f.close()
    print (states_history.shape ) # (4000001, 48)
    return states_history


def load_td_info_history(sim_load_name):
    data_file = sim_load_name + '/debug_td_info_history.pkl'

    print ('loading td info history...')
    f = open(data_file, 'rb')
    td_info_history = pickle.load(f, encoding='latin1')  # latin1 needed for python 2 pickle
    f.close()
    print (td_info_history.shape ) # (4000001, 48)
    return td_info_history
