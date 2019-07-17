import time
import cv2
import numpy as np
import pickle
from tabulate import tabulate

from math import sin, cos
from cuda_dist_query import CudaTable
from utils.load_sim_history import load_states_history, load_td_info_history
from utils.fps_counter import FPSCounter
from brain_components_classes.states_history import StatesLimitedHistory
from brain_components_classes.single_multi_context_layer import SingleMultiContextLayer

NL_SIM_DIR = '/srv/projects/NL-sim/'


class MultiLayerEnsemble(object):
    def __init__(self, params):
        '''

        :param params:
            {
                'input_dim': dim,
                'IO_entries_per_layer': [200, 200],
                'C_entries_per_layer': [800, 800],
                'predict_time_per_layer': [8, 8],
            }
        '''

        self.n_layers = len(params['IO_entries_per_layer'])

        self.tables = []
        self.last_i_c_dists = []

        IO_entries_per_layer = params['IO_entries_per_layer']
        C_entries_per_layer = params['C_entries_per_layer']

        layer_IO_learn_times = params['layer_IO_learn_times']
        layer_C_learn_times = params['layer_C_learn_times']

        assert self.n_layers == len(IO_entries_per_layer) and self.n_layers == len(C_entries_per_layer)
        assert self.n_layers == len(layer_IO_learn_times) and self.n_layers == len(layer_C_learn_times)
        assert self.n_layers == len(params['predict_time_per_layer'])

        self.layer_IO_learn_time_ranges = []
        self.layer_C_learn_time_ranges = []

        for layer in range(self.n_layers):
            self.layer_IO_learn_time_ranges.append((None, None))
            self.layer_C_learn_time_ranges.append((None, None))

        step_start_t = 0
        for step in range(1, self.n_layers + 3):
            IO_learn_layer = step - 1
            IO_C_learn_layer = step - 3

            if IO_learn_layer < 0 or IO_learn_layer > self.n_layers - 1:
                IO_learn_layer = None
            if IO_C_learn_layer < 0 or IO_C_learn_layer > self.n_layers - 1:
                IO_C_learn_layer = None

            if IO_learn_layer is not None:
                step_IO_learn_time = layer_IO_learn_times[IO_learn_layer]
            else:
                step_IO_learn_time = 0

            if IO_C_learn_layer is not None:
                step_IO_C_learn_time = layer_C_learn_times[IO_C_learn_layer]
            else:
                step_IO_C_learn_time = 0

            # take max, with assumption that learning "too long" is not a problem
            step_learn_time = max(step_IO_learn_time, step_IO_C_learn_time)
            step_stop_t = step_start_t + step_learn_time

            if IO_learn_layer is not None:
                self.layer_IO_learn_time_ranges[IO_learn_layer] = (step_start_t, step_stop_t)
            if IO_C_learn_layer is not None:
                self.layer_C_learn_time_ranges[IO_C_learn_layer] = (step_start_t, step_stop_t)

            step_start_t += step_learn_time

            #print(step, 'IO:', IO_learn_layer, 'IO+C:', IO_C_learn_layer)

        print()
        print('IO learn:', self.layer_IO_learn_time_ranges)
        print('IO_C learn:', self.layer_C_learn_time_ranges)
        print()

        disp_table = []

        for layer_index in range(self.n_layers):
            num_io_entries = params['IO_entries_per_layer'][layer_index]
            num_c_entries = params['C_entries_per_layer'][layer_index]

            if layer_index == 0:
                input_dim = params['input_dim']
            else:
                input_dim = params['IO_entries_per_layer'][layer_index - 1]

            if layer_index == self.n_layers - 1:
                context_dim = params['goal_context_dim']
                include_context = True
            else:
                context_dim = params['IO_entries_per_layer'][layer_index + 1]
                include_context = True

            disp_table.append([layer_index,
                               '(' + str(num_io_entries) + ' X ' + str(input_dim) + '+' + str(input_dim) + ')',
                               '(' + str(num_c_entries) + ' X ' + str(context_dim) + ')'])

            self.tables.append(SingleMultiContextLayer(params={
                'input_dim': input_dim,
                'output_dim': input_dim,
                'context_dim': context_dim,
                'num_IO_entries': num_io_entries,
                'num_C_entries': num_c_entries,
                'predict_time': params['predict_time_per_layer'][layer_index],
                'include_context': include_context
            }))

            self.last_i_c_dists.append(np.zeros(params['IO_entries_per_layer'][layer_index], np.float32))

        print()
        print(tabulate(disp_table, headers=['layer', 'input-output', 'context']))
        print()

        # stats
        self.stat_IO_row_replaces = np.zeros(self.n_layers, np.int)
        self.stat_C_row_replaces = np.zeros(self.n_layers, np.int)

        self.stat_disp_last_time = time.time()
        self.stat_disp_interval = 10

        self.t = 0

        self.predict_time_per_layer = np.array(params['predict_time_per_layer'])

    def step(self, input_state, input_x_y_theta, goal_context_state):
        '''

        :param input_state:
        :param input_x_y_theta:
        :param learn:
        :param goal_context_state: None or a context state for a currently reached goal
        :return:
        '''

        # TODO actually in this function: goal_context_state should never be None
        #   because: even a none-goal state has a context

        assert goal_context_state is not None

        scaled_i_c_dists = None

        for layer_index in range(self.n_layers):

            if layer_index == 0:
                layer_input_state = input_state
                layer_input_x_y_theta = input_x_y_theta
            else:
                layer_input_state = scaled_i_c_dists
                layer_input_x_y_theta = None

            if layer_index == self.n_layers - 1:

                # TODO how to properly introduce goal state? passing correct "future" layer_context state here? or everything delayed?
                # here we either need goal-context from the future, or we need to apply it with a separate function (not passed in step)

                '''
                what is context delay here?

                say current time is 't'

                say this layer's output is "in the future" by a total of 'tau2'
                and this layer's input is "in the future" by a total of 'tau1'
                i.e. overall, it is meant to be predicting tau2 into the future, accounting for all
                previous layers and their prediction times (tau1) as well as last layer's prediction time (tau2 - tau1).

                means in layer.step, to last layer, you are passing:
                    input: at (t + tau1)
                    output: at (t + tau2)

                what should context be? at minimum,
                    context: at (t + tau2)  -   but, could be at a later time

                say currently, agent encounters a goal state (now at time t) - what should we do with it?
                    let's say in task mode we always want to say:
                    "when you get this goal-context, be at the goal at time: (t + tau2)"

                * any time we encounter the goal right now at time t: *
                * give learning_context_delay as tau2 *

                '''

                # TODO define goal_encountered, goal_context_state
                # TODO make sure single_multi_context_layer can deal with both scenarios below
                # TODO for task mode- we need to add a "task_context_state" which will be same for most layers

                if goal_context_state is not None:
                    layer_context_state = goal_context_state
                    layer_context_delay = np.sum(self.predict_time_per_layer)
                else:
                    layer_context_state = None
                    layer_context_delay = 0
            else:
                layer_context_state = self.last_i_c_dists[layer_index + 1]
                layer_context_delay = 0

            IO_learn_t_range = self.layer_IO_learn_time_ranges[layer_index]
            C_learn_t_range = self.layer_C_learn_time_ranges[layer_index]

            learn_IO = IO_learn_t_range[0] <= self.t < IO_learn_t_range[1]
            learn_C = C_learn_t_range[0] <= self.t < C_learn_t_range[1]

            # if context
            scaled_i_c_dists, IO_replaced, C_replaced = self.tables[layer_index].step(input_state=layer_input_state,
                                                                                      learning_context_state=layer_context_state,
                                                                                      learning_context_delay=layer_context_delay,  # how much is this context delayed compared to I/O? normally zero, but for goal-context, it is delayed
                                                                                      input_x_y_theta=layer_input_x_y_theta,
                                                                                      learn_IO=learn_IO,
                                                                                      learn_C=learn_C)

            if IO_replaced:
                self.stat_IO_row_replaces[layer_index] += 1
            if C_replaced:
                self.stat_C_row_replaces[layer_index] += 1

            self.last_i_c_dists[layer_index] = scaled_i_c_dists.copy()

        if time.time() - self.stat_disp_last_time > self.stat_disp_interval:
            print('Time:', self.t)
            print('IO_row_replaces:', self.stat_IO_row_replaces)
            print('C_row_replaces:', self.stat_C_row_replaces)

            self.stat_IO_row_replaces = np.zeros(self.n_layers, np.int)
            self.stat_C_row_replaces = np.zeros(self.n_layers, np.int)
            self.stat_disp_last_time = time.time()

        self.t += 1

    def get_table_ims(self):
        IO_im_list = []
        C_im_list = []
        W_im_list = []

        for layer_index in range(self.n_layers):
            IO_im_list.append(self.tables[layer_index].get_IO_im(layer_index=layer_index))
            C_im_list.append(self.tables[layer_index].get_C_im())
            W_im_list.append(self.tables[layer_index].get_W_im())

        return IO_im_list, C_im_list, W_im_list

    def save_to_pkl(self, file_path, file_name):

        for table in self.tables:
            table.prepare_for_save()

        assert '.pkl' in file_name, 'invalid pickle filename: ' + file_name
        f = open(file_path + '/' + file_name, 'wb')
        pickle.dump(self, f)
        f.close()

    def init_after_load(self):
        for table in self.tables:
            table.init_after_load()
        pass

def test_run_multi_layer_ensemble():
    sim_load_name = '48DIMx1M_states_positions_saved_2018-01-31T13:09:15.676826'
    plots_save_folder = NL_SIM_DIR + sim_load_name + '/'
    states_history = load_states_history(plots_save_folder)
    td_info_history = load_td_info_history(plots_save_folder)

    dim = states_history.shape[1]
    max_history_length = states_history.shape[0]

    # TODO these ratios need a systematic way to adjust!
    IO_entries_per_layer = [400, 200, 200]

    learn_time_factor = 200

    C_entries_per_layer = [IO_entries_per_layer[0] * 4, IO_entries_per_layer[1] * 4, IO_entries_per_layer[2] * 4]
    layer_IO_learn_times = [IO_entries_per_layer[0] * learn_time_factor, IO_entries_per_layer[1] * learn_time_factor, IO_entries_per_layer[2] * learn_time_factor]
    layer_C_learn_times = [IO_entries_per_layer[0] * learn_time_factor, IO_entries_per_layer[1] * learn_time_factor, 0]

    ensemble = MultiLayerEnsemble(params={
        'input_dim': dim,
        'IO_entries_per_layer': IO_entries_per_layer,
        'C_entries_per_layer': C_entries_per_layer,
        'predict_time_per_layer': [4] * len(IO_entries_per_layer),
        'layer_IO_learn_times': layer_IO_learn_times,
        'layer_C_learn_times': layer_C_learn_times,
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
                      goal_context_state=None)

        if time.time() > last_imshow_time + imshow_every_k_seconds:
            last_imshow_time = time.time()

            IO_im_list, C_im_list, W_im_list = ensemble.get_table_ims()

            k = 0
            # print('*** IO ***')
            for im in IO_im_list:
                if im is not None:
                    cv2.imshow('IO_' + str(k), im)
                    # print(k, im.shape, im.dtype, np.amin(im), np.amax(im))
                else:
                    pass
                    # print(k, None)
                k += 1

            k = 0
            # print('*** C ***')
            for im in C_im_list:
                if im is not None:
                    cv2.imshow('C_' + str(k), im)
                    # print(k,im.shape, im.dtype, np.amin(im), np.amax(im))
                else:
                    pass
                    # print(k, None)
                k += 1

            k = 0
            # print('*** W ***')
            for im in W_im_list:
                if im is not None:
                    cv2.imshow('W_' + str(k), im)
                    # print(k,im.shape, im.dtype, np.amin(im), np.amax(im))
                else:
                    pass
                    # print(k, None)
                k += 1

        cv2.waitKey(1)
        fps.update()

    while True:
        cv2.waitKey(1)