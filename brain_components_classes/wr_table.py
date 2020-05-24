
import time
import cv2
from math import sqrt
import random
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg
import skcuda.misc as misc
from cuda_dist_query import CudaTable
from utils.fps_counter import FPSCounter
import SimpSOM as sps


np.set_printoptions(threshold=np.inf, linewidth=400)


def debug_print(vars_dict):
    print()
    for name in vars_dict:
        print(name)
        print()
        print(vars_dict[name])
        print()


class WRTableLimitOneIn(object):
    def __init__(self, num_entries, input_dim, num_q=1, disable_row_row_dists=False):

        self.use_gpu = False

        self.num_entries = num_entries
        self.input_dim = input_dim

        assert num_q == 1
        self.num_q = num_q  # number of query input rows

        table_i = np.random.random((num_entries, input_dim)).astype(np.float32)
        #table_i = 0.01 * np.ones((num_entries, input_dim), np.float32)

        self.table_i = table_i

        self.num_r_then_c = np.zeros((num_entries, num_entries), np.float32)  # number of times sequence [r -> c] occurred over two frames
        self.num_r_c_simul = np.zeros((num_entries, num_entries), np.float32)  # number of times [r, c] occurred on same frame
        self.num_r = np.zeros(num_entries, np.float32)  # number of times r has occurred overall

        self.t = 0
        self.active_last = None

        self.tmp_i = 0  # for _simple_add_to_table


    def query_multiple_rows(self, query_inputs):
        assert query_inputs.shape[0] == 1, 'only one input row supported currently'
        input_row = query_inputs.flatten()  # one input row only

        self._simple_add_to_table(input_row=input_row)

    def query_multiple_rows_SOM(self, query_inputs):
        # try implementing a SOM directly
        assert query_inputs.shape[0] == 1, 'only one input row supported currently'
        input_row = query_inputs.flatten()  # one input row only


        if 1:
            # PUT THIS STUFF IN INIT!!!!!!
            # som training
            self.som_train_frames = 2000
            self.som_dataset = np.zeros((self.som_train_frames, self.input_dim))
            self.som_data_t = 0  # how many frames we have collected

            self.som_learn_rate = 0.01
            self.som_epochs = 5000

        if 0:
            if self.som_data_t < self.som_train_frames:
                self.som_dataset[self.som_data_t, :] = input_row[:]
                if self.som_data_t < self.num_entries:
                    self.table_i[self.som_data_t, :] = input_row[:]  # just for info debug
            elif self.som_data_t == self.som_train_frames:
                self.som_net = sps.somNet(int(sqrt(self.num_entries)), int(sqrt(self.num_entries)), self.som_dataset, PBC=True, n_jobs=4)
                self.som_net.train(self.som_learn_rate, self.som_epochs)
                self.som_net.save()
        else:
            self.table_i = np.load('somNet_trained.npy')

        self.som_data_t += 1

        # first use table to collect sample data to train som

        # then train som, save weights

        # then, load weights into table


    def _save_som_to_file(self):
        '''

        :return:
        '''
        pass

    def _load_som_from_file(self):
        '''

        :return:
        '''
        pass

    def query_multiple_rows_342(self, query_inputs):
        assert query_inputs.shape[0] == 1, 'only one input row supported currently'
        input_row = query_inputs.flatten()  # one input row only

        tmp1 = np.dot(self.table_i, input_row)
        tmp2 = 1.0 / np.sum(self.table_i, axis=1)

        match_val = np.multiply(tmp1, tmp2)

        ind_by_match = np.argsort(match_val)[::-1]

        # force that first four matches should sum together to give image
        top_k = self.num_entries
        ind_top_k = ind_by_match[0:top_k]

        # input was binary, 0 or 1
        # if input was 0 for an element:
        #   all rows in top k unlearn, relative to their row value
        # if input was 1 for an element:
        #   largest row learns per element, all other unlearn

        input_zero_ind = np.nonzero(input_row < 0.5)[0]
        input_one_ind = np.nonzero(input_row > 0.5)[0]

        assert len(input_zero_ind) + len(input_one_ind) == len(input_row)

        ind_top_k = ind_top_k[:, np.newaxis]
        input_zero_ind = input_zero_ind[np.newaxis, :]
        input_one_ind = input_one_ind[np.newaxis, :]

        lr = 0.1 # 0.01

        #inv_match = 1.0 - match_val
        inv_match = np.ones(self.num_entries)
        #inv_match = match_val

        self.table_i[ind_top_k, input_zero_ind] = self.table_i[ind_top_k, input_zero_ind] - lr * np.multiply(self.table_i[ind_top_k, input_zero_ind], inv_match[ind_top_k])

        top_k_rows_one_in = self.table_i[ind_top_k, input_one_ind]
        max_row_per_element_of_one_in = ind_top_k[np.argmax(top_k_rows_one_in, axis=0)].flatten()  # column vector

        self.table_i[ind_top_k, input_one_ind] = self.table_i[ind_top_k, input_one_ind] - lr * np.multiply(self.table_i[ind_top_k, input_one_ind], inv_match[ind_top_k])

        # BELOW WOULD NEED UPDATE TO USE inv_match correctly! print(self.table_i[max_row_per_element_of_one_in, input_one_ind].shape)

        self.table_i[max_row_per_element_of_one_in, input_one_ind] = self.table_i[max_row_per_element_of_one_in, input_one_ind] + 2 * lr * self.table_i[max_row_per_element_of_one_in, input_one_ind]

        self.table_i[self.table_i < 0] = 0
        self.table_i[self.table_i > 1] = 1

        #drift = 0.0000001 * np.random.random((self.table_i.shape[0], self.table_i.shape[1]))
        #self.table_i = self.table_i + drift


    def query_multiple_rows_asd(self, query_inputs):
        # sum_appr = np.sum(self.table_i[ind_by_match[0:top_k], :], axis=0)
        # print(sum_appr.shape, np.amax(sum_appr), np.amin(sum_appr))


        # active decorrelate, continuous

        assert query_inputs.shape[0] == 1, 'only one input row supported currently'
        input_row = query_inputs.flatten()  # one input row only

        tmp1 = np.dot(self.table_i, input_row)
        tmp2 = 1.0 / np.sum(self.table_i, axis=1)

        match_val = np.multiply(tmp1, tmp2)

        print(match_val)

        if np.amin(match_val) < 0.0:
            print('!! under min!', np.amin(match_val))
        if np.amax(match_val) > 1.0:
            print('!! over max!', np.amax(match_val))

        xc = self.num_r_c_simul

        #print()
        #print(xc)
        #print()

        ind_by_match = np.argsort(match_val)[::-1]

        inhibit_amt = np.zeros(self.num_entries)
        TAU_LEARN = 0.01
        TAU_ESTIM = 0.01

        for row_i in list(ind_by_match):
            learn_rate = TAU_LEARN * (1.0 - inhibit_amt[row_i]) * (1.0 - match_val[row_i])  # this is not good

            #self.table_i[row_i, :] = learn_rate * input_row + keep_rate * self.table_i[row_i, :]
            self.table_i[row_i, :] = self.table_i[row_i, :] + learn_rate * (input_row > 0) - learn_rate * (input_row < 0)

            xc_row = xc[row_i, :] #- 0.5
            inhibit_amt = np.maximum(inhibit_amt, match_val[row_i] * xc_row)

            TAU_LEARN = TAU_LEARN * 0.9

        self.table_i[self.table_i < 0.0] = 0.0
        self.table_i[self.table_i > 1.0] = 1.0

        if 0:
            bias = np.multiply(xc, match_val[:, np.newaxis])  # matrix, column vector
            # bias[r, c] is: xc(r, c) * match(r)
            #   which means unit "c" should be inhibited by the max of bias[:, c]

            inhibit_amt = np.amax(bias, axis=0)  # max of each column


            learning_rate = TAU_LEARN * np.multiply((1.0 - inhibit_amt), match_val)  # num_entries
            keep_rate = (1.0 - learning_rate)  # num_entries

            tmp1 = learning_rate[:, np.newaxis]
            tmp2 = input_row[np.newaxis, :]
            term_1 = np.dot(tmp1, tmp2)  # (1200, 16384)
            term_2 = np.multiply(keep_rate[:, np.newaxis], self.table_i)  # (1200, 16384)

            self.table_i = term_1 + term_2

            #self.table_i[self.table_i < 0] = 0
            #self.table_i[self.table_i > 1] = 1

            # have to break symmetry!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        drift = 0.00001 * np.random.random((self.table_i.shape[0], self.table_i.shape[1]))
        self.table_i = self.table_i + drift

        self.num_r_c_simul[:, :] = TAU_ESTIM * np.dot(match_val[:, np.newaxis], match_val[np.newaxis, :]) + (1.0 - TAU_ESTIM) * self.num_r_c_simul[:, :]
        self.num_r_c_simul[range(self.num_entries), range(self.num_entries)] = 0.0  # self is zero

        self.t += 1

    def aquery_multiple_rows(self, query_inputs):
        '''

        # compute feedforward match
        # compute lateral + context from: past step events, last-frame-predictive-correlation
        # compute lateral - context from: current step feedforward match, current-frame-correlation
        # generate events from top k based on: feedforward match, lateral -, lateral +
        # learn ff RF

        :param query_inputs:
        :return:
        '''

        assert query_inputs.shape[0] == 1, 'only one input row supported currently'
        input_row = query_inputs.flatten()  # one input row only

        tmp1 = np.dot(self.table_i, input_row)
        tmp2 = 1.0 / np.sum(self.table_i, axis=1)

        match_val = np.multiply(tmp1, tmp2)

        if np.amin(match_val) < 0.0:
            print('!! under min!', np.amin(match_val))
        if np.amax(match_val) > 1.0:
            print('!! over max!', np.amax(match_val))

        xc = np.divide(self.num_r_c_simul, self.num_r[:, np.newaxis] + 1e-9)  # num_r_c_simul is symmetric

        # TODO suppress match_val or activation by xc and match_val
        # TODO bias becomes measure of: match_val * cross corr, that is, how much to suppress one that this is correlated with (not how much to suppress this one)

        bias = np.multiply(xc, match_val[:, np.newaxis])  # matrix, column vector
        # bias[r, c] is: xc(r, c) * match(r)
        #   which means unit "c" should be inhibited by the max of bias[:, c]

        inhibit_amt = np.amax(bias, axis=0)  # max of each column

        active_val = np.multiply(match_val, 1.0 - inhibit_amt)

        #print()
        #print(xc)

        if 0:
            debug_print({'num_r_c_simul': self.num_r_c_simul,
                         'xc': xc,
                         'match_val': match_val,
                         'bias': bias
                         #'max_xc_by_row': max_xc_by_row,
                         #'active_val': active_val
                         })

        #print()
        #print('num_r_c_simul')
        #print()
        #print(self.num_r_c_simul)
        #print()
        #print('xc')
        #print()
        #print(xc)
        #print()

        top_k = 5
        ind_by_match = np.argsort(active_val)[::-1]
        active_now = ind_by_match[0:top_k]

        learn_rate = 0.01 * (1.0 - inhibit_amt)
        keep_rate = 1.0 - learn_rate

        #self.table_i[active_now, :] = np.multiply(learn_rate[:, np.newaxis] * query_inputs[:] + 0.99 * self.table_i[active_now, :]

        self.num_r[active_now] += 1

        r_tmp = np.repeat(active_now, len(active_now))
        c_tmp = np.tile(active_now, len(active_now))

        self.num_r_c_simul[r_tmp, c_tmp] = self.num_r_c_simul[r_tmp, c_tmp] + 1

        self.num_r_c_simul[range(self.num_entries), range(self.num_entries)] = 0.0  # self is zero

        self.active_last = active_now.copy()
        self.t += 1


    def old__init__(self, num_entries, input_dim, num_q=1, disable_row_row_dists=False):

        self.use_gpu = False

        self.num_entries = num_entries
        self.input_dim = input_dim

        assert num_q == 1
        self.num_q = num_q  # number of query input rows

        table_i = np.random.random((num_entries, input_dim)).astype(np.float32)
        # table_i = 0.01 * np.ones((num_entries, input_dim), np.float32)

        self.table_i = table_i
        self.all_diffs = np.zeros_like(self.table_i)

        self.context_mat = np.zeros((num_entries, num_entries), np.float32)

        self.from_to_num = np.zeros((num_entries, num_entries), np.float32)

        self.frames = 0

        self.last_match_val = np.zeros(num_entries, np.float32)

        # for now, hard-coded p_prob
        # p_sum_in_per_row = 1.0
        # p_num_per_row = 20
        # n_sum_in_per_row = -1.0
        # n_num_per_row = 20

        # for k in range(num_entries):
        #     p_per_in = p_sum_in_per_row / p_num_per_row  # assume evenly distributed
        #     n_per_in = n_sum_in_per_row / n_num_per_row  # assume evenly distributed
        #
        #     rand_indices = np.random.permutation(num_entries)
        #     p_in_indices = rand_indices[0:p_num_per_row]
        #     n_in_indces = rand_indices[p_num_per_row:p_num_per_row+n_num_per_row]
        #     self.context_mat[k, p_in_indices] = p_per_in  # pre: column indices, post: row index
        #     self.context_mat[k, n_in_indces] = n_per_in  # pre: column indices, post: row index
        # print(np.amin(self.context_mat), np.amax(self.context_mat))
        # exit(1)

        self.last_input_row = None
        self.active_last = None

        # for simple_add_to_table
        self.tmp_i = 0

    def query_multiple_rows_OLD234(self, query_inputs):
        assert query_inputs.shape[0] == 1, 'only one input row supported currently'
        input_row = query_inputs.flatten()  # one input row only

        tmp1 = np.dot(self.table_i, input_row)
        tmp2 = 1.0 / np.sum(self.table_i, axis=1)

        match_val = np.multiply(tmp1, tmp2)
        if np.amin(match_val) < 0.0:
            print('!! under min!', np.amin(match_val))
        if np.amax(match_val) > 1.0:
            print('!! over max!', np.amax(match_val))

        ind_by_match = np.argsort(match_val)[::-1]
        top_k = 5
        # maybe instead of forward activity driving match
        # ... it should be driven by lateral activity
        # and forward should learn it

        active_now = ind_by_match[0:top_k]

        if self.active_last is not None:
            self.from_to_num[self.active_last, active_now] += 1  # NOT CORRECT
            # each row builds lateral connections in attempting to predict its own forward input
            # must maintain a set sum

        if self.frames > 0:
            print('***')
            # p_prob = self.from_to_num / self.frames   # this is prob (from -> to) sequence occuring overall
            # want prob ( to active | from active )
            # print(self.from_to_num[4, :])  # 4 predicting N happened how many times

            # want to compute prob of each row on current frame from previous frame
            #   then they learn feedforward to match the prediction
            #   probably need some constraints on prediction...

            prob_to_given_from = np.multiply(self.from_to_num, 1.0 / (np.sum(self.from_to_num, axis=1) + 1e-9))
            print('prob_to_given_from', prob_to_given_from)
            prob_from_given_to = np.multiply(self.from_to_num, 1.0 / (np.sum(self.from_to_num, axis=0) + 1e-9))
            print('prob_from_given_to', prob_from_given_to)

            # right constraint might be to force prediction probs towards being sparse per row, and to sum to 1.0 (outgoing) or incoming
            #   sum outgoing to one means: on average, this row being active predicts one row being active on next frame
            #   sum incoming to one means: on average, one row on previous frame predicted this row being active on current frame

            # would be interesting to learn feedforward that matches a particular specific distribution (linear, power law) of randomly set prediction probabilities
            # compare desired distribution, as a check, to computed distribution as above

            # first, just see what network is doing without any learning, what prediction stats it is gathering
            # print('lateral context: ', np.sum())


        self.active_last = active_now.copy()
        self.frames += 1

    def query_multiple_rows_bla(self, query_inputs):
        assert query_inputs.shape[0] == 1, 'only one input row supported currently'
        input_row = query_inputs.flatten()  # one input row only


        target_total_match_per_im = 4.0  # after ranking in order of match, when learning prediction

        # can we locally apply backprop?
        # i.e. a "fragmented" backprop learning


        # prediction pairs that sharpen their prediction?
        #   they all must start as an actual image pair, pre->post
        #   the only operation that is allowed is removing weight, no modifying/adding/adapting
        #   goal is, remove all but subset that meets some criteria
        #       might involve tagging/ labeling all units of the stored rows with multiple tags
        #

        # general cross-correlation of input across time steps?
        # i.e.
        #   slide current input vs. previous input, and find points of greatest correlation
        #   some idea of "space" is necessary for this to work
        #   i.e. a distance metric must be present, cannot be a bag
        #       implies that, this class needs to know how many bins per pixel and how pixels are arranged
        #       and, you need to do it by row/column

        pass

    def query_multiple_rows_REMAINDER(self, query_inputs):
        assert query_inputs.shape[0] == 1, 'only one input row supported currently'
        input_row = query_inputs.flatten()  # one input row only

        remainder = input_row.copy()
        invalidated_rows = []

        self.tmp_im = []

        table_i_new = self.table_i.copy()

        # while sum remainder > something
        for k in range(10):  # hard coding for the moment
            tmp1 = np.dot(self.table_i, remainder)
            tmp2 = 1.0 / np.sum(self.table_i, axis=1)
            match_val = np.multiply(tmp1, tmp2)

            #print('invalidated rows', invalidated_rows)

            #assert np.amin(match_val) >= 0.0, (np.amin(remainder), np.amax(remainder), np.amin(match_val), match_val)
            #assert np.amax(match_val) <= 1.0, (np.amin(remainder), np.amax(remainder), np.amax(match_val), match_val)
            if np.amin(match_val) < 0.0:
                print('min', np.amin(match_val))
            if np.amax(match_val) > 1.0:
                print('max', np.amax(match_val))

            # find best match for remainder
            row_i = None
            while row_i is None or row_i in invalidated_rows:
                if row_i is not None:
                    match_val[row_i] = -np.inf
                row_i = np.argmax(match_val)

            # learn on remainder
            row_before_learn = self.table_i[row_i, :].copy()
            #print(np.amin(remainder), np.amax(remainder)) # if max is also zero, TODO PROBLEM

            table_i_new[row_i, :] = 0.01 * remainder[:] + 0.99 * self.table_i[row_i, :]
            # table_i_new[row_i, :] = remainder[:]  #np.minimum(1.0, self.table_i[row_i] + 0.05 * remainder)

            #self.table_i[row_i, np.nonzero(remainder==0)[0]] = np.maximum(self.table_i[row_i, np.nonzero(remainder==0)[0]] - 0.05, 0.0)
            # find new remainder based on row before learning
            remainder = np.maximum(0.0, remainder - row_before_learn)

            if np.amax(remainder) == 0.0:
                break

            # invalidate current row for rest of this loop
            invalidated_rows.append(row_i)
            self.tmp_im.append(remainder.copy())

        self.table_i = table_i_new.copy()

    def old_code_4(self):
        assert query_inputs.shape[0] == 1, 'only one input row supported currently'
        input_row = query_inputs.flatten()  # one input row only
        # print('****************')
        # find match for all rows
        # print(self.table_i.shape, input_row.shape)
        tmp1 = np.dot(self.table_i, input_row)
        tmp2 = 1.0 / np.sum(self.table_i, axis=1)
        # print(tmp1.shape, tmp2.shape)
        match_val = np.multiply(tmp1, tmp2)
        #print(match_val)
        assert np.amin(match_val) >= 0.0
        assert np.amax(match_val) <= 1.0

        Mt = np.nonzero(match_val > 0.9)[0]
        Mn = np.nonzero(match_val <= 0.9)[0]

        mp = np.multiply(self.table_i, input_row[np.newaxis, :])
        self.table_i[Mn, :] += 0.01 * mp[Mn, :]
        self.table_i[Mt, :] += 0.1 * mp[Mt, :]
        print(match_val)

    def old_code_3(self):
        a = np.argsort(match_val)[::-1]

        mp = np.multiply(self.table_i, input_row[np.newaxis, :])

        self.table_i -= 0.01 * mp
        self.table_i[a[0:3], :] += 0.1 * mp[a[0:3], :]

        #self.table_i += 0.0001
        self.table_i[self.table_i > 1.0] = 1.0
        self.table_i[self.table_i < 0.0] = 0.0

    def old_code2(self):

        a = np.argsort(match_val)[::-1]

        top_k = 3
        learning_rate = 0.001 * np.ones(self.num_entries, np.float32)
        learning_rate[a[0:top_k]] = 0.01
        keep_rate = 1.0 - learning_rate

        tmp1 = learning_rate[:, np.newaxis]
        tmp2 = input_row[np.newaxis, :]
        term_1 = np.dot(tmp1, tmp2)  # (1200, 16384)

        term_2 = np.multiply(keep_rate[:, np.newaxis], self.table_i)  # (1200, 16384)

        self.table_i = term_1 + term_2

        self.last_match_val[:] = match_val[:]

        # context_in = np.dot(self.context_mat, self.last_match_val)  # range: [-1.0, 1.0]
        # match_prod = np.dot(match_val[:, np.newaxis], self.last_match_val[np.newaxis, :])

        # self._simple_add_to_table(input_row=input_row)

    def old_code(self):
        learning_rate = 0.2 * context_in
        # print('****', np.amin(context_in), np.amax(context_in))
        keep_rate = 1.0 - learning_rate

        tmp1 = learning_rate[:, np.newaxis]
        tmp2 = input_row[np.newaxis, :]
        term_1 = np.dot(tmp1, tmp2)  # (1200, 16384)
        term_2 = np.multiply(keep_rate[:, np.newaxis], self.table_i)  # (1200, 16384)

        self.table_i = term_1 + term_2
        self.table_i[self.table_i < 0.0] = 0.0
        self.table_i[self.table_i > 1.0] = 1.0

        # tmp3 = np.sum(self.table_i, 1)[:, np.newaxis] + 1e-9
        # self.table_i = self.table_i * 1.0 / tmp3

        self.last_match_val[:] = match_val[:]


    def query_mulitple_rows_test(self, query_inputs):
        assert query_inputs.shape[0] == 1, 'only one input row supported currently'
        input_row = query_inputs.flatten()  # one input row only

        self._simple_add_to_table(input_row=input_row)

    def _simple_add_to_table(self, input_row):
        if self.tmp_i < self.table_i.shape[0]:
            self.table_i[self.tmp_i, :] = input_row[:]
            self.tmp_i += 1
        elif self.tmp_i == self.table_i.shape[0]:
            pass
            #self.tmp_i = 0  # reset to begin

    def _init_num_query_inputs(self, num_query_inputs):
        '''

        use as:
            at start:
            self.num_query_inputs = None

            later:
            if self.num_query_inputs is None:
                self._init_num_query_inputs(num_query_inputs=query_inputs.shape[0])

        :param num_query_inputs:
        :return:
        '''

        assert False, 'unused right now'

        self.num_query_inputs = num_query_inputs
        # for 1200 X 256 X 1024, this takes 2 GB

        self.all_diffs = np.zeros((self.num_entries, self.input_dim), np.float32)
        # (1200, 256, 1024)

    def query_multiple_rows_OLD(self, query_inputs):
        '''

        :param query_inputs:
        :return:
        '''

        # self.table i:  (1200, 1024):  entries X pixels
        # query_inputs: (256, 1024):   tiles X pixels

        # calculate and store diff per pixel across all query_inputs and all rows

        # diff = self.table_i - query_inputs
        # print()
        # print(diff.shape)
        # print()

        # dists: (1, 1200)
        # argmin_dists: (1,)

        query_inputs_flatten = query_inputs.flatten()  # one input row only

        self.all_diffs[:, :] = self.table_i - query_inputs_flatten  # half the time
        self.all_diffs[:, :] = np.abs(self.all_diffs[:, :])  # half the time

        do_adapt = True

        if do_adapt:
            use_wr = True

            # TODO try that it only learns inside the win region?

            if use_wr:
                # compute win regions per table row, using all_diffs
                #   self.all_diffs.shape:  # (1200, 16384)
                winning_row_per_pixel = np.argmin(self.all_diffs, 0)  # (16384,)
                win_regions = np.zeros((self.num_entries, self.input_dim), np.int)
                win_regions[winning_row_per_pixel, np.arange(win_regions.shape[1])] = 1

                # try here: only learn inside win region, all learn equal learning rate
                learn_rate = 0.01
                keep_rate = 1.0 - learn_rate

                term_1 = keep_rate * self.table_i

                query_inputs_tiled = np.tile(query_inputs_flatten, (win_regions.shape[0], 1))
                term_2 = learn_rate * query_inputs_tiled

                nnz_win = np.nonzero(win_regions)
                self.table_i[nnz_win] = term_1[nnz_win] + term_2[nnz_win]

                if 0:
                    # converge towards mean
                    learning_rate =  0.0 * np.ones(self.table_i.shape[0], np.float32)
                    keep_rate = 1.0 - learning_rate

                    tmp1 = learning_rate[:, np.newaxis]
                    tmp2 = query_inputs_flatten[np.newaxis, :]
                    term_1 = np.dot(tmp1, tmp2)  # (1200, 16384)

                    term_2 = np.multiply(keep_rate[:, np.newaxis], self.table_i)  # (1200, 16384)

                    self.table_i = term_1 + term_2

            else:
                win_regions = np.ones((self.num_entries, self.input_dim), np.int)

                # for each input row: for all rows: compute average pixel error inside win regions

                wr_diff_sums = np.sum(np.multiply(self.all_diffs, win_regions), axis=1)  # length: num entries
                wr_diff_means = np.multiply(wr_diff_sums, 1.0 / np.sum(win_regions, axis=1))

                # for each input row: compute rank of table rows
                ranks = np.zeros(wr_diff_means.shape[0], np.int)
                argsort_diff_means = np.argsort(wr_diff_means)
                ranks[argsort_diff_means] = np.arange(wr_diff_means.shape[0])  # lower rank is smaller dist

                # for each input row: adapt proportionate to rank

                if 1:
                    base_rate = 0.01
                    learning_rate = base_rate * 0.01 * np.ones(wr_diff_means.shape[0], np.float32)
                    learning_rate[argsort_diff_means[0]] = base_rate

                if 0:
                    learn_tmp = ranks * 1.0 / ranks.shape[0]
                    learn_tmp = np.power(1.0 - learn_tmp, 4)

                    learning_rate_base = 0.001
                    learning_rate = learning_rate_base * learn_tmp

                keep_rate = 1.0 - learning_rate

                # learning rate: (1200,)
                # keep_rate: (1200,)
                # query_inputs: (16384,)
                # table_i: (1200, 16384)

                tmp1 = learning_rate[:, np.newaxis]
                tmp2 = query_inputs_flatten[np.newaxis, :]
                term_1 = np.dot(tmp1, tmp2)  # (1200, 16384)

                term_2 = np.multiply(keep_rate[:, np.newaxis], self.table_i)  # (1200, 16384)

                self.table_i = term_1 + term_2

    def post_init(self):
        pass

    def set_matrix_row(self, row_index, row_input, fast_init=True):
        pass

