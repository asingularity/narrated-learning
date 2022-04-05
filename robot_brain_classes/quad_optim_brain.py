import time
import cv2
from gekko import GEKKO

import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt
from brain_components_classes.states_history import StatesLimitedHistory
#from cython_eff import compute_eff
from offline_analyses.try_sparsify_3 import get_sparse_features, make_im

from utils.one_time_messages import OneTimeMessages

# WHYI IS THIS BROKEN NOW
#from cuda_dist_query import CudaTable

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log

np.set_printoptions(suppress=True, precision=2)

random.seed(0)
np.random.seed(0)


# TODO install Gekko to docker


class QuadOptimBrain(object):
    def __init__(self, params):

        self.input_im_dim = params['input_im_dim']
        self.num_rfs = params['num_rfs']
        self.lr = params['lr']
        self.max_time = params['max_time']

        self.do_raster_plots_every_k_im = params['do_raster_plots_every_k_im']
        self.ims_since_raster = 0

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

        self.plots_folder = "."
        self.input_concat_timesteps = 1
        self.input_history = StatesLimitedHistory(params={'max_delay': self.input_concat_timesteps,
                                                          'states_dim_list': [self.input_state_dim],
                                                          'store_extra_data': False})

        self.raster_steps = 200
        self.raster_t = 0  # circular; draw vertical line on plot here
        self.input_raster_history = np.zeros((self.input_state_dim, self.raster_steps), np.uint8)
        self.rfs_raster_history = np.zeros((self.num_rfs, self.raster_steps), np.uint8)

        self.fig_bar = plt.figure(figsize=(40, 20))
        self.ax_bar = self.fig_bar.add_subplot(1, 1, 1)
        self.ax_bar.cla()
        self.ax_bar.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_bar.get_yaxis().get_major_formatter().set_scientific(False)

        self.error_mean_time = 50000

        self.rf_weights = np.random.random((self.num_rfs, self.input_state_dim)) * 1e-2  # 1e-12

        self.optim_time_steps = 1000
        self.cardinality_K = 40
        self.optim_every_k_steps = self.optim_time_steps

        self.Q_size = self.optim_time_steps
        self.Q = np.zeros((self.Q_size, self.Q_size))

        self.input_history_for_Q = StatesLimitedHistory(params={'max_delay': self.optim_time_steps,
                                                                'states_dim_list': [self.input_state_dim],
                                                                'store_extra_data': False})

        self.t = 0

    def optim_rf(self, input_history, explained):
        state_seq = self.input_history_for_Q.get_state_sequence(delay_long=self.optim_time_steps - 1, delay_short=0)
        # print(state_seq.shape)  # (10000, 128): (M, L)

        M = self.optim_time_steps
        L = self.input_state_dim

        # TODO
        #   THIS ISNT RIGHT!!!
        #   TODO should apply thresholding!!! Based on activation!
        state_seq_remainder = state_seq - explained
        state_seq_remainder[state_seq_remainder < 0] = 0

        # TODO apply this if zeroing out -1*-1 below!
        # state_seq_remainder[state_seq_remainder == 0] = -1

        # compute all dot products

        # (1000, L) (L, 1000)   ->   (1000, 1000)
        all_dot_products = np.matmul(state_seq_remainder, np.transpose(state_seq_remainder))
        # this is already the quadtratic matrix, but need to zero out the diagonal:
        all_dot_products[np.arange(M), np.arange(M)] = 0

        # TODO
        #   need to zero out wherever it was (-1) * (-1)

        # tmp = state_seq_remainder.copy()
        # tmp[tmp > 0] = 0
        # neg_dot_products = np.matmul(tmp, np.transpose(tmp))
        # neg_dot_products[np.arange(M), np.arange(M)] = 0

        # all_dot_products = all_dot_products - neg_dot_products

        # all_dot_products = all_dot_products * 1.0 / (abs(np.amax(all_dot_products)))

        print('min', np.amin(all_dot_products), 'max', np.amax(all_dot_products))
        print(all_dot_products)

        self.Q = all_dot_products

        assert self.Q_size == M
        assert self.Q.shape[0] == self.Q.shape[1] == self.Q_size

        V = np.zeros(self.Q_size)

        new_rf_weights = np.zeros(self.input_state_dim)
        new_explained = explained.copy()

        m = GEKKO(remote=False)
        m.options.MAX_MEMORY = 5
        # m.options.SOLVER = 2  # 2

        num_vars = V.shape[0]
        x_lower = 0
        x_upper = 1

        x_select = [m.Var(value=0, lb=x_lower, ub=x_upper, integer=True) for i in range(num_vars)]

        # cardinality constraint
        enable_cardinality_constraint = True
        if enable_cardinality_constraint:
            print('initializing cardinality equation...')
            # compare Type 2 vs. Type 3 constraints: solution speed and value
            m.Equation((m.sum([x_select[i] for i in range(num_vars)])) == self.cardinality_K)

        print('    initializing q objective...')
        _ = m.qobj(b=V, A=self.Q, x=x_select, otype='max')

        print('    NONZERO Q', np.count_nonzero(self.Q))

        print('    solving...')

        try:
            m.solve(disp=True)
            print('finished solving.')

            x_arr = np.zeros(num_vars)
            for i in range(num_vars):
                select_val = x_select[i].value[0]
                x_arr[i] = select_val  # x_value[i].value[0]

            count_nnz = np.count_nonzero(x_arr)
            print('nnz x_arr:', count_nnz)
            if count_nnz > 0:
                nnz = np.nonzero(x_arr)[0]
                new_rf_weights[:] = np.mean(state_seq_remainder[nnz, :], axis=0)

                # TODO re-enable after applying thresholding to activation!!!
                #new_rf_weights[new_rf_weights >= 0.5] = 1
                #new_rf_weights[new_rf_weights < 0.5] = 0

                # new_rf_weights[:] = x_arr[0:L]

                # TODO explained is per frame, not overall; need to apply it per frame based on activation!

                new_explained = new_explained + new_rf_weights
                new_explained[new_explained > 1] = 1
            else:
                print('ERROR: zero chosen!!!')
        except:
            print('not solved')
            raise

        return new_rf_weights, new_explained


    def optim_rf_no_work(self, input_history, explained):
        # build Q matrix

        state_seq = self.input_history_for_Q.get_state_sequence(delay_long=self.optim_time_steps - 1, delay_short=0)
        #print('daeg 111', np.count_nonzero(state_seq), state_seq.size)
        # print(state_seq.shape)  # (10000, 128): (M, L)

        L = self.input_state_dim
        M = self.optim_time_steps

        #         self.Q_size = self.input_state_dim + self.optim_time_steps

        assert self.Q_size == L + M
        assert self.Q.shape[0] == self.Q.shape[1] == self.Q_size

        # TODO these need thought!
        state_seq_remainder = state_seq - explained
        state_seq_remainder[state_seq_remainder < 0] = 0

        state_seq_remainder_with_inv = state_seq_remainder.copy()
        state_seq_remainder_with_inv[state_seq_remainder_with_inv == 0] = -1

        #print('daeg 222', np.count_nonzero(state_seq_remainder_with_inv > 0), state_seq.size)

        self.Q[L:L+M, 0:L] = state_seq_remainder_with_inv
        self.Q[0:L, L:L+M] = np.transpose(state_seq_remainder_with_inv)

        # why 0.5? because symmetric
        self.Q = 0.5 * self.Q

        print('count nnz', np.count_nonzero(self.Q), 2 * state_seq.size)

        # optimize: constrain to integer!
        V = np.zeros(self.Q_size)

        new_rf_weights = np.zeros(self.input_state_dim)
        new_explained = explained.copy()

        m = GEKKO(remote=False)
        m.options.MAX_MEMORY = 5
        # m.options.SOLVER = 2  # 2

        num_vars = V.shape[0]
        x_lower = 0
        x_upper = 1

        x_select = [m.Var(value=0, lb=x_lower, ub=x_upper, integer=True) for i in range(num_vars)]

        # cardinality constraint
        enable_cardinality_constraint = False

        if enable_cardinality_constraint:
            print('initializing cardinality equation...')

            # compare Type 2 vs. Type 3 constraints: solution speed and value
            m.Equation((m.sum([x_select[i] for i in range(num_vars)])) == 2)

        print('    initializing q objective...')
        _ = m.qobj(b=V, A=self.Q, x=x_select, otype='max')

        print('    solving...')

        try:
            m.solve(disp=True)
            print('finished solving.')

            x_arr = np.zeros(num_vars)
            for i in range(num_vars):
                select_val = x_select[i].value[0]
                x_arr[i] = select_val  # x_value[i].value[0]

            print('nnz x_arr:', np.count_nonzero(x_arr))

            new_rf_weights[:] = x_arr[0:L]
            new_explained = new_explained + new_rf_weights
            new_explained[new_explained > 1] = 1

        except:
            print('not solved')

        return new_rf_weights, new_explained

    def process_input(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):

        if self.t >= self.max_time:
            return

        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            return

        # TODO for prediction, Don't skip zeros!!!

        self.input_history_for_Q.process_new_states(newest_states_list=[input_state])

        self.rfs_raster_history[:, self.raster_t] = 0

        if self.t > self.optim_time_steps and self.t % self.optim_every_k_steps == 0:
            print()
            print('Doing full optimization!')
            print()

            explained = np.zeros_like(input_state)

            for rf in range(self.num_rfs):
                print()
                print('    Doing RF: ', rf, 'of', self.num_rfs)
                print()
                # remainder is over whole states history!
                # better way: accruing a "thing to subtract out" instead
                # i.e. excess, explained

                self.rf_weights[rf, :], new_explained = self.optim_rf(input_history=self.input_history_for_Q,
                                                                      explained=explained)
                explained = new_explained

        # self.rfs_raster_history[activated_rf, self.raster_t] = 1

        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0

    def get_final_errors_dict(self):
        d = {}
        return d

    def get_table_ims(self):

        if self.do_raster_plots_every_k_im is not None:
            self.ims_since_raster += 1
            if self.ims_since_raster > self.do_raster_plots_every_k_im:
                self.do_plots()
                self.ims_since_raster = 0

        ims_list = []
        ims_names_list = []

        rfs_im, rf_ims_dict = make_im(self.rf_weights,
                                      num_bins_per_pixel=1,
                                      input_im_dim=self.input_im_dim,
                                      im_final_dim=int(50 * 3000 / 800),  # /800 for two-im per rf display
                                      mod_for_disp=int(sqrt(self.num_rfs)),
                                      normalize_weights=True)

        ims_list.append(rfs_im)
        ims_names_list.append('prob_weights')

        return ims_list, ims_names_list

    def set_plots_folder(self, folder):
        self.plots_folder = folder
        print()
        print('setting plots folder: ', self.plots_folder)
        print()

    def do_plots(self):

        # time plots:
        self.ax_bar.cla()
        num_rf = self.rfs_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.rfs_raster_history, np.arange(num_rf)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.ax_bar.axvline(x=self.raster_t, color='g')
        self.fig_bar.savefig(self.plots_folder + "/raster_rfs.png", dpi=100)




class QuadOptimBrainStepByStep(object):
    def __init__(self, params):
        self.input_im_dim = params['input_im_dim']
        self.num_rfs = params['num_rfs']
        self.lr = params['lr']
        self.max_time = params['max_time']

        self.do_raster_plots_every_k_im = params['do_raster_plots_every_k_im']
        self.ims_since_raster = 0

        self.input_state_dim = 2 * self.input_im_dim * self.input_im_dim  # why 2? + and - changes

        self.plots_folder = "."
        self.input_concat_timesteps = 1
        self.input_history = StatesLimitedHistory(params={'max_delay': self.input_concat_timesteps,
                                                          'states_dim_list': [self.input_state_dim],
                                                          'store_extra_data': False})

        self.raster_steps = 200
        self.raster_t = 0  # circular; draw vertical line on plot here
        self.input_raster_history = np.zeros((self.input_state_dim, self.raster_steps), np.uint8)
        self.rfs_raster_history = np.zeros((self.num_rfs, self.raster_steps), np.uint8)

        self.fig_bar = plt.figure(figsize=(40, 20))
        self.ax_bar = self.fig_bar.add_subplot(1, 1, 1)
        self.ax_bar.cla()
        self.ax_bar.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax_bar.get_yaxis().get_major_formatter().set_scientific(False)

        self.error_mean_time = 50000

        self.rf_weights = np.random.random((self.num_rfs, self.input_state_dim)) * 1e-2  # 1e-12

        Q_size = self.num_rfs + self.num_rfs * self.input_state_dim

        self.Q = np.zeros((Q_size, Q_size))

        self.t = 0

    def _make_matrix(self, input_state):

        n_in = input_state.shape[0]
        assert n_in == self.input_state_dim

        input_state_with_inv = input_state.copy()
        input_state_with_inv[input_state_with_inv == 0] = -1

        Q_size = self.num_rfs + self.num_rfs * n_in
        Q = self.Q

        for rf_ind in range(self.num_rfs):
            row = rf_ind * (n_in + 1)
            for input_ind in range(n_in):
                col = row + 1 + input_ind
                #print(rf_ind, row, col)

                Q[row, col] = input_state_with_inv[input_ind]
                Q[col, row] = Q[row, col]

        Q = Q * 0.5  # because symmetric
        return Q

    def _make_vector(self, alpha, rfs_prev):

        n_in = self.input_state_dim
        V_size = self.num_rfs + self.num_rfs * n_in

        V = np.zeros(V_size)

        for rf_ind in range(self.num_rfs):
            offset = rf_ind * (n_in + 1)
            V[offset] = 0
            for input_ind in range(n_in):
                V[offset + 1 + input_ind] = alpha * rfs_prev[rf_ind, input_ind]

        return V

    def process_input(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):

        if self.t >= self.max_time:
            return

        input_state = np.concatenate((input_events_p, input_events_n))

        self.input_raster_history[:, self.raster_t] = input_state[:]

        if input_state is None:
            return

        if np.count_nonzero(input_state) == 0:
            return

        self.rfs_raster_history[:, self.raster_t] = 0

        Q = self._make_matrix(input_state=input_state)
        print('nnz Q', np.count_nonzero(Q))
        V = self._make_vector(alpha=0.1, rfs_prev=self.rf_weights)
        print('nnz V', np.count_nonzero(V))
        assert Q.shape[0] == V.shape[0]
        assert Q.shape[1] == V.shape[0]

        m = GEKKO(remote=False)
        m.options.MAX_MEMORY = 5
        m.options.SOLVER = 2  # 2

        print('initializing variables...')
        num_vars = V.shape[0]
        print('    num_vars:', num_vars)
        # vars
        x_lower = 0
        x_upper = 1


        # TODO constraints! RFS must be [0 -> 1] continuous!
        #   but a-values [0, 1] binary!
        #   for now all same one way or other

        x_select = [m.Var(value=0, lb=x_lower, ub=x_upper, integer=False) for i in range(num_vars)]
        # x_select = [m.Var(value=0, integer=False) for i in range(num_vars)]

        print('initializing q objective...')
        _ = m.qobj(b=V, A=Q, x=x_select, otype='max')

        print('solving...')
        t0_solve = time.time()

        try:
            m.solve(disp=True)
            print('finished solving.')

            x_arr = np.zeros(num_vars)
            for i in range(num_vars):
                select_val = x_select[i].value[0]
                x_arr[i] = select_val  # x_value[i].value[0]

            print(x_arr)
            n_in = self.input_state_dim

            for rf_ind in range(self.num_rfs):
                offset = rf_ind * (n_in + 1)
                for input_ind in range(n_in):
                    self.rf_weights[rf_ind, input_ind] = x_arr[offset + 1 + input_ind]

        except:
            print('not solved')

        # self.rfs_raster_history[activated_rf, self.raster_t] = 1

        self.t += 1
        self.raster_t += 1
        if self.raster_t >= self.raster_steps:
            self.raster_t = 0

    def get_final_errors_dict(self):
        d = {}
        return d

    def get_table_ims(self):

        if self.do_raster_plots_every_k_im is not None:
            self.ims_since_raster += 1
            if self.ims_since_raster > self.do_raster_plots_every_k_im:
                self.do_plots()
                self.ims_since_raster = 0

        ims_list = []
        ims_names_list = []

        rfs_im, rf_ims_dict = make_im(self.rf_weights,
                                      num_bins_per_pixel=1,
                                      input_im_dim=self.input_im_dim,
                                      im_final_dim=int(200 * 3000 / 800),  # /800 for two-im per rf display
                                      mod_for_disp=int(sqrt(self.num_rfs)),
                                      normalize_weights=True)

        ims_list.append(rfs_im)
        ims_names_list.append('prob_weights')

        return ims_list, ims_names_list

    def set_plots_folder(self, folder):
        self.plots_folder = folder
        print()
        print('setting plots folder: ', self.plots_folder)
        print()

    def do_plots(self):

        # time plots:
        self.ax_bar.cla()
        num_rf = self.rfs_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.rfs_raster_history, np.arange(num_rf)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax_bar.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.ax_bar.axvline(x=self.raster_t, color='g')
        self.fig_bar.savefig(self.plots_folder + "/raster_rfs.png", dpi=100)
