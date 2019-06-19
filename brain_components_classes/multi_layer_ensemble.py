import time
import cv2
import numpy as np
from math import sin, cos
from cuda_dist_query import CudaTable
from utils.load_sim_history import load_states_history, load_td_info_history
from utils.fps_counter import FPSCounter
from brain_components_classes.states_history import StatesLimitedHistory
from brain_components_classes.single_table_layer import SingleTableLayer

NL_SIM_DIR = '/srv/projects/NL-sim/'


class MultiLayerEnsemble(object):
    def __init__(self, params):
        '''

        :param params:
            {
                'predictor_ensemble_load_from_file': False,
                'predictor_ensemble_filename': None,
                'input_dim': dim,
                'entries_per_layer': [200, 200],
                'predict_time_per_layer': [8, 8],
            }
        '''

        self.n_layers = len(params['entries_per_layer'])

        self.tables = []
        self.last_i_c_dists = []

        for layer_index in range(self.n_layers):
            if layer_index == 0:
                input_dim = params['input_dim']
            else:
                input_dim = params['entries_per_layer'][layer_index - 1]

            if layer_index == self.n_layers - 1:
                context_dim = 0
                include_layers = ['io_only', 'i_only']  # last layer: io: learning, i: running
            else:
                context_dim = params['entries_per_layer'][layer_index + 1]
                include_layers = ['ioc', 'ic_only']  # ioc: learning, ic: running

            self.tables.append(SingleTableLayer(params={
                'input_dim': input_dim,
                'output_dim': input_dim,
                'context_dim': context_dim,
                'num_entries': params['entries_per_layer'][layer_index],
                'predict_time': params['predict_time_per_layer'][layer_index],
                'include_layers': include_layers
            }))

            self.last_i_c_dists.append(np.zeros(params['entries_per_layer'][layer_index], np.float32))

    def step(self, input_state, input_x_y_theta, learn):
        '''

        :param input_state:
        :param input_x_y_theta:
        :param learn:
        :return:
        '''

        scaled_i_c_dists = None

        for layer_index in range(self.n_layers):

            if layer_index == 0:
                layer_input_state = input_state
                layer_input_x_y_theta = input_x_y_theta
            else:
                layer_input_state = scaled_i_c_dists
                layer_input_x_y_theta = None

            if layer_index == self.n_layers - 1:
                layer_context_state = None
            else:
                layer_context_state = self.last_i_c_dists[layer_index + 1]

            scaled_i_c_dists = self.tables[layer_index].step(input_state=layer_input_state,
                                                             context_state=layer_context_state,
                                                             input_x_y_theta=layer_input_x_y_theta,
                                                             learn=learn)

            self.last_i_c_dists[layer_index] = scaled_i_c_dists.copy()

    def get_table_ims(self):
        im_list = []

        for layer_index in range(self.n_layers):
            im_list.append(self.tables[layer_index].get_table_im(layer_index=layer_index))

        return im_list


def test_run_multi_layer_ensemble():
    sim_load_name = '48DIMx1M_states_positions_saved_2018-01-31T13:09:15.676826'
    plots_save_folder = NL_SIM_DIR + sim_load_name + '/'
    states_history = load_states_history(plots_save_folder)
    td_info_history = load_td_info_history(plots_save_folder)

    dim = states_history.shape[1]
    max_history_length = states_history.shape[0]

    ensemble = MultiLayerEnsemble(params={
        'predictor_ensemble_load_from_file': False,
        'predictor_ensemble_filename': None,
        'input_dim': dim,
        'entries_per_layer': [400, 200, 200, 200],
        'predict_time_per_layer': [8, 8, 8, 8],
    })

    fps = FPSCounter(params={'display_every_k_seconds': 5})

    # show image
    last_imshow_time = time.time()
    imshow_every_k_seconds = 2.0

    for t in range(max_history_length):

        input_state = states_history[t, :]
        x_y_theta = td_info_history[t, :]

        ensemble.step(input_state=input_state,
                      input_x_y_theta=x_y_theta,
                      learn=True)

        if time.time() > last_imshow_time + imshow_every_k_seconds:
            last_imshow_time = time.time()

            im_list = ensemble.get_table_ims()
            k = 0
            for im in im_list:
                if im is not None:
                    cv2.imshow('table_' + str(k), im)
                k += 1

        cv2.waitKey(1)
        fps.update()
