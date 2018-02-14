
import time
import cv2
import pickle
import numpy as np
np.set_printoptions(suppress=True)
#from PVM.PVM_framework import MLP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brain_components_classes.predictor_ensemble import PredictorEnsemble


def _load_states_history(sim_load_name):
    data_file = '/home/intec/NL-sim/' + sim_load_name + '/states_history_0.pkl'

    print 'loading states history...'
    f = open(data_file, 'r')
    states_history = pickle.load(f)
    f.close()
    print states_history.shape  # (4000001, 48)
    return states_history


def _load_td_info_history(sim_load_name):
    data_file = '/home/intec/NL-sim/' + sim_load_name + '/debug_td_info_history.pkl'

    print 'loading td info history...'
    f = open(data_file, 'r')
    td_info_history = pickle.load(f)
    f.close()
    print td_info_history.shape  # (4000001, 48)
    return td_info_history


def _do_display(input_state, dim, scale_camera_factor):
    input_im_color = np.zeros((1, dim / 3, 3))
    input_im_color[0, :, :] = input_state.reshape((dim / 3, 3))

    resized_input_im = cv2.resize(src=input_im_color, dsize=(0, 0), fx=scale_camera_factor,
                                  fy=scale_camera_factor, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('actual', resized_input_im)
    cv2.waitKey(1)


def _compute_sparse_inputs(input_state, n_bins):

    io_bounds = (0.0, 1.0)

    input_sparse = np.zeros(input_state.shape[0] * n_bins) + io_bounds[0]
    input_indices = (input_state * n_bins).astype(np.int)
    input_indices[input_indices == n_bins] = n_bins - 1
    offset = np.arange(0, input_indices.shape[0] * n_bins, n_bins)
    input_indices = input_indices + offset
    input_sparse[input_indices] = io_bounds[1]

    return input_sparse.astype(np.float)


def print_debug_info(debug_info):
    if len(debug_info) > 0:
        print
        for name in debug_info:
            print name + ':', debug_info[name]

        print

def run_experiment():
    sim_load_name = '48DIMx1M_states_positions_saved_2018-01-31T13:09:15.676826'
    plots_save_folder = '/home/intec/NL-sim/' + sim_load_name + '/'
    #plots_save_folder = '/home/intec/NL-tmp/'

    states_history = _load_states_history(sim_load_name)
    td_info_history = _load_td_info_history(sim_load_name)

    dim = states_history.shape[1]

    max_history_length = states_history.shape[0]
    scale_camera_factor = 32
    do_display = False
    plot_error_every_k_seconds = 60 * 5
    imshow_every_k_seconds = 1
    start_step_offset = 0
    env_width_height = 30

    do_random_permute_train = True
    #learning_off_time = np.inf
    learning_off_time = 900000
    #sim_off_time = np.inf
    sim_off_time = 1000000

    ensemble = PredictorEnsemble(params={'max_history_length': max_history_length,
                                         'plots_save_folder': plots_save_folder,
                                         'error_average_steps': 500,
                                         'entries': 40000,
                                         'replacement_every_k_steps': 1,  # 100 for 800 rows, 10 for 8000 rows
                                         'dim': dim,
                                         'use_context_in_knn_diff': True,  # if False, input+output only. no context.
                                         'do_random_init': True,
                                         'do_adaptation': True,
                                         'do_replacements': True,
                                         'env_width_height': env_width_height,  # for plotting positions
                                         'plots_prefix': 'EXPR2_init_1_adapt_1_repl_1_40K_entries_learn_off_900K'})

    k_to_train = np.arange(3, max_history_length - 1)[start_step_offset::]

    if do_random_permute_train:
        k_to_train = np.random.permutation(k_to_train)


    fps_frames = 0
    last_fps_time = time.time()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    t = 0
    last_plot_time = time.time()
    last_imshow_time = time.time()
    to_debug_print = {}

    for k in k_to_train.tolist():
        fps_frames += 1

        if time.time() > last_fps_time + 5:
            print 'FPS: ', fps_frames / (time.time() - last_fps_time)
            print_debug_info(to_debug_print)
            fps_frames = 0
            last_fps_time = time.time()

        last_input_state = states_history[k - 1, :]
        last_x_y_theta = td_info_history[k - 1, :]

        input_state = states_history[k, :]
        x_y_theta = td_info_history[k, :]

        next_input_state = states_history[k + 1, :]

        if do_display:
            _do_display(input_state, dim, scale_camera_factor)

        to_debug_print = ensemble.step(last_input_state=last_input_state,
                                      input_state=input_state,
                                      next_input_state=next_input_state,
                                      last_x_y_theta=last_x_y_theta,
                                      x_y_theta=x_y_theta,
                                      learn=(t < learning_off_time))

        if t == learning_off_time:
            print 'SAVING ENSEMBLE TO PKL'
            ensemble.save_to_pkl()

        if t > sim_off_time:
            print 'Sim Complete:', t, sim_off_time
            return

        if time.time() > last_imshow_time + imshow_every_k_seconds:
            im = ensemble.get_table_im()
            cv2.imshow('im', im)
            cv2.waitKey(1)
            last_imshow_time = time.time()

        if time.time() > last_plot_time + plot_error_every_k_seconds:
            print 'plotting error', time.time()
            ensemble.plot_error(fig, ax)
            last_plot_time = time.time()
        t += 1


if __name__ == '__main__':
    run_experiment()
















