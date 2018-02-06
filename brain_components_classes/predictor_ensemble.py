
from knn_parallel import knn_query as knn_parallel_query
import time
import cv2
import pickle
import random
import numpy as np
np.set_printoptions(suppress=True)
#from PVM.PVM_framework import MLP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PredictorEnsemble(object):
    def __init__(self, params):
        print 'initializing ensemble...'

        self.do_random_init = params['do_random_init']  # False  # initialize with random (first) entries
        self.do_adaptation = params['do_adaptation']  # True  # WTA-based learning
        self.do_replacements = params['do_replacements']  # True  # replace low effectiveness over time

        self.max_history_length = params['max_history_length']
        self.error_history = np.zeros(params['max_history_length'])
        self.mean_error_history = np.zeros(params['max_history_length'])

        self.use_context_in_knn_diff = params['use_context_in_knn_diff']

        self.plots_prefix = params['plots_prefix']
        self.env_width_height = params['env_width_height']
        self.error_step = 0

        self.entries = params['entries']
        self.dim = params['dim']

        self.debug_x_y_theta_output = np.zeros((self.entries, 3))
        self.debug_x_y_theta_input = np.zeros((self.entries, 3))

        # input (t), prediction (t+1), context (t+2)
        self.table = np.zeros((self.entries, self.dim * 3)).astype(np.float32)
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

    def step(self, last_input_state, input_state, next_input_state, last_x_y_theta, x_y_theta, learn=True):
        '''
        for learning mode

        :param last_input_state:
        :param input_state:
        :param next_input_state:
        :param last_x_y_theta:
        :param x_y_theta:
        :param learn:
        :return:
        '''
        self.t += 1

        predictor_context = next_input_state
        predictor_output = input_state
        predictor_input = last_input_state

        # 1. find input+output (IO) error
        new_entry = np.concatenate((predictor_input, predictor_output))
        new_entry = np.concatenate((new_entry, predictor_context))

        if self.use_context_in_knn_diff:
            dist, ind = knn_parallel_query(self.table, new_entry, self.temp_array, self.table.shape[0], self.dim * 3)
        else:
            dist, ind = knn_parallel_query(self.table, new_entry, self.temp_array, self.table.shape[0], self.dim * 2)

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

    def plan_and_get_debug_position_angle_list(self, goal_state, starting_state):
        print 'predictor_ensemble::plan_and_get_debug_position_angle_list'

        # print goal_state.shape, starting_state.shape  # 48, 48
        # print self.table.shape # (800, 144)
        # print self.dim  # 48

        # planning:
        N = 5  # currently this is also number of planned steps
        iter_steps = 100
        # TODO should be able to display plan as it evolves, inside this function, each iteration
        # for now, just display intermediate states here

        # rows: implicit - table_output - output rows are always the same column of same table.
        #   currently all rows are implicitly propagated with some assigned uncertainty
        #   (will require all-to-all distance pre-compute between input & context, output & context, input & output)
        # uncertainties: inverse confidence (propagated error) of next or previous level's predictions.
        #   needs normalization?

        output_uncertainties = {}

        for k in range(N):
            output_uncertainties[k] = None

        for iter_step in range(iter_steps):
            # run level 0
            level = 0
            out_uncert = self._run_level_0(input_starting_state=starting_state,
                                           context={'rows': 'table_output', 'uncertainties': output_uncertainties[level + 1]})
            output_uncertainties[level] = out_uncert.copy()

            # run intermediate levels
            for level in range(1, N - 1):
                out_uncert = self._run_level_k(input={'rows': 'table_output', 'uncertainties': output_uncertainties[level - 1]},
                                               context={'rows': 'table_output', 'uncertainties': output_uncertainties[level + 1]})
                output_uncertainties[level] = out_uncert.copy()

            # run highest level: N - 1
            level = N - 1
            out_uncert = self._run_level_N(input={'rows': 'table_output', 'uncertainties': output_uncertainties[level - 1]},
                                           context_goal_state=goal_state)
            output_uncertainties[level] = out_uncert.copy()

        # return random positions & angles
        # TODO fix this to return real position angle list!
        plan_position_angle_list = []

        for k in range(N):
            rand_entry = random.randint(0, self.entries - 1)
            plan_position_angle_list.append((self.debug_x_y_theta_output[rand_entry, 0],
                                             self.debug_x_y_theta_output[rand_entry, 1],
                                             self.debug_x_y_theta_output[rand_entry, 2]))

        return plan_position_angle_list

    def _run_level_0(self, input_starting_state, context):
        '''
        :param input_starting_state: single array of size self.dim
        :param context: dict, ex: {'rows': 'table_output', 'uncertainties': output_uncertainties[level + 1]}
        :return: single array of size self.entries
        '''

        out_uncert = np.ones(1)
        return out_uncert

    def _run_level_k(self, input, context):
        out_uncert = np.ones(1)
        return out_uncert

    def _run_level_N(self, input, context_goal_state):
        out_uncert = np.ones(1)
        return out_uncert

    def save_to_pkl(self):
        f = open(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + '_PredictorEnsemble.pkl', 'w')
        pickle.dump(self, f)
        f.close()

    def get_table_im(self):

        entries = 40

        im_input = self.table[0:entries, 0:self.dim]
        im_prediction = self.table[0:entries, self.dim:self.dim * 2]
        im_context = self.table[0:entries, self.dim * 2::]

        # interleave rows so context below prediction below input
        A = im_input
        B = im_prediction
        C = im_context
        D = np.empty((A.shape[0] + B.shape[0] + C.shape[0], A.shape[1]))

        D[::3, :] = A
        D[1::3, :] = B
        D[2::3, :] = C
        D = np.reshape(D, (D.shape[0], D.shape[1] / 3, 3))

        im = D
        im = cv2.resize(im, dsize=(0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)

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
        fig.savefig(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + 'error_history' + '.png', dpi=100)

        ax.cla()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        sorted_net_indices = np.argsort(self.table_use_hist)[::-1]
        ax.bar(np.arange(self.table_use_hist.shape[0]), self.table_use_hist[sorted_net_indices])
        fig.savefig(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + 'table_use_hist' + '.png', dpi=100)

        ax.cla()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        effectiveness_mean = np.divide(self.effectiveness_sum, self.effectiveness_num)
        sorted_effectiveness = np.argsort(effectiveness_mean)[::-1]
        ax.bar(np.arange(effectiveness_mean.shape[0]), effectiveness_mean[sorted_effectiveness])
        fig.savefig(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + 'effectiveness_hist' + '.png', dpi=100)

        ax.cla()
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        ax.set_xlim([0 - 0.1, self.env_width_height + 0.1])
        ax.set_ylim([0 - 0.1, self.env_width_height + 0.1])
        ax.plot(self.debug_x_y_theta_input[:, 0],  self.debug_x_y_theta_input[:, 1], 'go')
        #ax.plot(self.debug_x_y_theta_output[:, 0],  self.debug_x_y_theta_output[:, 1], 'ro')
        ax.set_title(str(np.count_nonzero(self.debug_x_y_theta_input[:, 0]) * 1.0 / self.debug_x_y_theta_input.shape[0]))
        fig.savefig(self.plots_save_folder + '/offline_trained_' + self.plots_prefix + '_' + 'debug_td_info' + '.png', dpi=100)
