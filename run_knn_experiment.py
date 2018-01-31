
import time
import cv2
import pickle
import numpy as np
np.set_printoptions(suppress=True)
from PVM.PVM_framework import MLP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from knn_parallel import knn_query as knn_parallel_query


def _load_states_history():
    data_file = '/home/intec/NL-sim/48DIMx1M_states_positions_saved_2018-01-31T13:09:15.676826/states_history_0.pkl'

    print 'loading states history...'
    f = open(data_file, 'r')
    states_history = pickle.load(f)
    f.close()
    print states_history.shape  # (4000001, 48)
    return states_history


def _load_td_info_history():
    data_file = '/home/intec/NL-sim/48DIMx1M_states_positions_saved_2018-01-31T13:09:15.676826/debug_td_info_history.pkl'

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


class PredictorEnsemble(object):
    def __init__(self, params):
        print 'initializing ensemble...'

        self.do_random_init = params['do_random_init']  # False  # initialize with random (first) entries
        self.do_adaptation = params['do_adaptation']  # True  # WTA-based learning
        self.do_replacements = params['do_replacements']  # True  # replace low effectiveness over time

        self.max_history_length = params['max_history_length']
        self.error_history = np.zeros(params['max_history_length'])
        self.mean_error_history = np.zeros(params['max_history_length'])

        self.plots_prefix = params['plots_prefix']
        self.env_width_height = params['env_width_height']
        self.error_step = 0

        self.entries = params['entries']
        self.dim = params['dim']

        self.debug_x_y_theta_output = np.zeros((self.entries, 3))
        self.debug_x_y_theta_input = np.zeros((self.entries, 3))

        self.table = np.zeros((self.entries, self.dim * 2)).astype(np.float32)
        #self.table_dist = np.zeros((self.entries, self.entries))  # max 10k * 10k
        #self.table_dist[np.arange(self.entries), np.arange(self.entries)] = np.inf

        #self.table_dist = np.random.random((self.entries, self.entries))  # max 10k * 10k

        self.min_dist_pair = (0, 1)  # doesn't matter since they all start 0
        self.min_dist = 0

        self.temp_array = np.zeros(self.entries).astype(np.float32)
        print 'finished initializing ensemble.'

        self.plots_save_folder = params['plots_save_folder']
        self.error_average_steps = params['error_average_steps']

        self.error_history = np.zeros(self.max_history_length)
        self.mean_error_history = np.zeros(self.max_history_length)
        self.error_step = 0

        self.table_use_hist = np.zeros(self.entries)
        self.tmp_ind = 0

        self.effectiveness_sum = np.zeros(self.entries)
        self.effectiveness_num = np.ones(self.entries)

        self.row_ages = np.zeros(self.entries)

        self.t = 0
        self.last_replacement_t = 0

    def step(self, last_input_state, input_state, last_x_y_theta, x_y_theta, learn=True):
        self.t += 1

        predictor_output = input_state
        predictor_input = last_input_state
        # 1. find input+output (IO) error
        new_entry = np.concatenate((predictor_input, predictor_output))

        dist, ind = knn_parallel_query(self.table, new_entry, self.temp_array, self.table.shape[0], self.table.shape[1])
        self.table_use_hist[ind] += 1

        self.row_ages[:] = self.row_ages[:] + 1

        # find second best
        sorted_dist_indices = np.argsort(self.temp_array)
        assert self.temp_array[sorted_dist_indices[0]] == dist
        ind2 = sorted_dist_indices[1]
        ind3 = sorted_dist_indices[2]
        dist2 = self.temp_array[ind2]

        self.error_history[self.error_step] = dist
        mean_index_0 = max(0, self.error_step - self.error_average_steps)
        mean_index_1 = self.error_step
        self.mean_error_history[self.error_step] = np.mean(self.error_history[mean_index_0:mean_index_1])
        self.error_step += 1

        if learn:

            do_random_init = self.do_random_init  # initialize with random entries
            do_adaptation = self.do_adaptation  # WTA-based learning
            do_replacements = self.do_replacements  # replace low effectiveness over time

            if do_random_init and self.tmp_ind < self.table.shape[0]:
                self.table[self.tmp_ind, :] = new_entry[:]
                self.debug_x_y_theta_output[self.tmp_ind, :] = x_y_theta[:]
                self.debug_x_y_theta_input[self.tmp_ind, :] = last_x_y_theta[:]
                self.tmp_ind += 1
            else:
                pass

            best_second_diff_current = dist2 - dist
            self.effectiveness_num[ind] += 1
            self.effectiveness_sum[ind] += best_second_diff_current

            # simple best learns:
            if do_adaptation:
                self.table[ind, :] = 0.9 * self.table[ind, :] + 0.1 * new_entry
                # for theta: could be weird... discontinuities
                self.debug_x_y_theta_output[ind, :] = 0.9 * self.debug_x_y_theta_output[ind, :] + 0.1 * x_y_theta[:]
                self.debug_x_y_theta_input[ind, :] = 0.9 * self.debug_x_y_theta_input[ind, :] + 0.1 * last_x_y_theta[:]

            min_replaced_row_age = 1000
            replacement_every_k_steps = 100

            if do_replacements:
                row_replace_candidates = np.nonzero(self.row_ages > min_replaced_row_age)[0]
                effectiveness_mean = np.divide(self.effectiveness_sum, self.effectiveness_num)

                if len(row_replace_candidates) > 0 and self.t > self.last_replacement_t + replacement_every_k_steps:
                    r_r_c_ind = np.argmin(effectiveness_mean[row_replace_candidates])
                    r_r_c_eff = effectiveness_mean[row_replace_candidates][r_r_c_ind]

                    #if r_r_c_eff < 0.2 or np.isnan(r_r_c_eff):
                    r_r_ind = row_replace_candidates[r_r_c_ind]
                    #print r_r_ind
                    self.table_use_hist[r_r_ind] = 0
                    self.row_ages[r_r_ind] = 0
                    self.effectiveness_sum[r_r_ind] = 0.0
                    self.effectiveness_num[r_r_ind] = 0

                    self.table[r_r_ind, :] = new_entry[:]
                    self.debug_x_y_theta_output[r_r_ind, :] = x_y_theta[:]
                    self.debug_x_y_theta_input[r_r_ind, :] = last_x_y_theta[:]

                    self.last_replacement_t = self.t
                    #    print 'replacing: ', r_r_c_eff
                    #else:
                    #    print 'NOT replacing: ', r_r_c_eff

    def get_table_im(self):

        entries = 30

        im_left = self.table[0:entries, 0:self.dim]
        im_right = self.table[0:entries, self.dim::]

        # interleave rows so prediction below input
        A = im_left
        B = im_right
        C = np.empty((A.shape[0] + B.shape[0], A.shape[1]))
        C[::2, :] = A
        C[1::2, :] = B
        C = np.reshape(C, (C.shape[0], C.shape[1] / 3, 3))

        im = C
        im = cv2.resize(im, dsize=(0,0), fx=20, fy=20, interpolation=cv2.INTER_NEAREST)

        return im

    def plot_error(self, fig, ax):
        ax.cla()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        ax.set_ylim([0.0, 10.0])
        ax.set_yticks(np.arange(0, 10, 0.5))
        ax.axhline(y=1.5, color='g')
        thing_to_plot = self.mean_error_history[0:self.error_step]
        ax.plot(thing_to_plot, 'b-')
        fig.savefig(self.plots_save_folder + '/' + self.plots_prefix + '_' + 'error_history' + '.png', dpi=100)

        ax.cla()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        sorted_net_indices = np.argsort(self.table_use_hist)[::-1]
        ax.bar(np.arange(self.table_use_hist.shape[0]), self.table_use_hist[sorted_net_indices])
        fig.savefig(self.plots_save_folder + '/' + self.plots_prefix + '_' + 'table_use_hist' + '.png', dpi=100)

        ax.cla()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        effectiveness_mean = np.divide(self.effectiveness_sum, self.effectiveness_num)
        sorted_effectiveness = np.argsort(effectiveness_mean)[::-1]
        ax.bar(np.arange(effectiveness_mean.shape[0]), effectiveness_mean[sorted_effectiveness])
        fig.savefig(self.plots_save_folder + '/' + self.plots_prefix + '_' + 'effectiveness_hist' + '.png', dpi=100)

        ax.cla()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        ax.set_xlim([0 - 0.1, self.env_width_height + 0.1])
        ax.set_ylim([0 - 0.1, self.env_width_height + 0.1])
        ax.plot(self.debug_x_y_theta_input[:, 0],  self.debug_x_y_theta_input[:, 1], 'go')
        #ax.plot(self.debug_x_y_theta_output[:, 0],  self.debug_x_y_theta_output[:, 1], 'ro')
        ax.set_title(str(np.count_nonzero(self.debug_x_y_theta_input[:, 0]) * 1.0 / self.debug_x_y_theta_input.shape[0]))
        fig.savefig(self.plots_save_folder + '/' + self.plots_prefix + '_' + 'debug_td_info' + '.png', dpi=100)


def run_experiment():
    plots_save_folder = '/home/intec/NL-tmp/'

    states_history = _load_states_history()
    td_info_history = _load_td_info_history()

    dim = states_history.shape[1]

    max_history_length = states_history.shape[0]
    scale_camera_factor = 32
    do_display = False
    plot_error_every_k_seconds = 10
    imshow_every_k_seconds = 1
    start_step_offset = 0
    env_width_height = 30

    do_random_permute_train = True
    #learning_off_time = np.inf
    learning_off_time = 300000

    ensemble = PredictorEnsemble(params={'max_history_length': max_history_length,
                                         'plots_save_folder': plots_save_folder,
                                         'error_average_steps': 500,
                                         'entries': 800,
                                         'dim': dim,
                                         'do_random_init': False,
                                         'do_adaptation': True,
                                         'do_replacements': True,
                                         'env_width_height': env_width_height,  # for plotting positions
                                         'plots_prefix': 'init_False_adapt_True_repl_True_800_entries_learn_off_300K'})

    k_to_train = np.arange(3, max_history_length)[start_step_offset::]

    if do_random_permute_train:
        k_to_train = np.random.permutation(k_to_train)


    fps_frames = 0
    last_fps_time = time.time()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    t = 0
    last_plot_time = time.time()
    last_imshow_time = time.time()

    for k in k_to_train.tolist():
        fps_frames += 1

        if time.time() > last_fps_time + 5:
            print 'FPS: ', fps_frames / (time.time() - last_fps_time)
            fps_frames = 0
            last_fps_time = time.time()

        last_input_state = states_history[k - 1, :]
        last_x_y_theta = td_info_history[k - 1, :]

        input_state = states_history[k, :]
        x_y_theta = td_info_history[k, :]

        if do_display:
            _do_display(input_state, dim, scale_camera_factor)

        ensemble.step(last_input_state=last_input_state,
                      input_state=input_state,
                      last_x_y_theta=last_x_y_theta,
                      x_y_theta=x_y_theta,
                      learn=(t < learning_off_time))

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
















