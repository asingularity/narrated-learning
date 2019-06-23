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

        # stats
        self.stat_row_replaces = np.zeros(self.n_layers, np.int)
        self.stat_disp_last_time = time.time()
        self.stat_disp_interval = 10

    def step(self, input_state, input_x_y_theta, learn):
        '''

        :param input_state:
        :param input_x_y_theta:
        :param learn:
        :return:
        '''

        scaled_i_c_dists = None
        invalidate_stats = np.zeros((self.n_layers, 2))

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

            # ODO need to add in: invalidate other layers' indices on replace (more difficult)
            # ODO need to add: refresh if needed (easy)
            # TODO also need to ignore invalid entries in lookups without biasing based on number of invalid
            # ODO also refresh valid_table wherever a row is replaced
            # ODO question- refreshing a row should count as replacement? will then trigger more invalidate?

            #   need to visualize / debug how many are invalidated (what percent of table of each layer over time)

            scaled_i_c_dists, replaced_row_index = self.tables[layer_index].step(input_state=layer_input_state,
                                                                                 context_state=layer_context_state,
                                                                                 input_x_y_theta=layer_input_x_y_theta,
                                                                                 learn=learn,
                                                                                 debug_print=layer_index==1
                                                                                 )

            if replaced_row_index is not None:

                # invalidate this index for layer below in context
                if layer_index > 0:
                    # invalidate whole "column" of table
                    self.tables[layer_index - 1].invalidate_index_context(column_index=replaced_row_index)

                # invalidate this index for layer above in input
                if layer_index < self.n_layers - 1:
                    # invalidate whole "column" of table
                    self.tables[layer_index + 1].invalidate_index_input(column_index=replaced_row_index)

                self.stat_row_replaces[layer_index] += 1

            self.last_i_c_dists[layer_index] = scaled_i_c_dists.copy()

            prop_table_valid, prop_rows_refreshed = self.tables[layer_index].get_invalidate_stats()
            invalidate_stats[layer_index, 0] = prop_table_valid
            invalidate_stats[layer_index, 1] = prop_rows_refreshed

        if time.time() - self.stat_disp_last_time > self.stat_disp_interval:
            print('row_replaces:')
            print(self.stat_row_replaces)
            print('table_valid, rows_refreshed:')
            print(invalidate_stats)

            self.stat_row_replaces = np.zeros(self.n_layers, np.int)
            self.stat_disp_last_time = time.time()

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
    imshow_every_k_seconds = 1.0

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
