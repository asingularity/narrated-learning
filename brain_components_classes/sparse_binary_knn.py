
import time
import numpy as np
import os

class SparseBinaryKNN(object):
    def __init__(self, params):
        '''

        For now this will assume one-hot encoding for each sparse input
        Later, we will need to upgrade to allow multiple selections per sparse input

        params:
            'sparse_io_dim': self.rows_per_tile,
            'num_sparse_inputs': num_predictor_tiles * len(self.prediction_tau_list),
            'num_sparse_outputs': 1

        :param params:
        '''

        # for now, assume only one sparse output, with one-hot
        assert params['num_sparse_outputs'] == 1

        # this is max_index + 1 that we would encounter; ie [0, sparse_io_dim-1] is range of input or output index per sparse io unit
        self.sparse_io_dim = params['sparse_io_dim']

        self.num_sparse_inputs = params['num_sparse_inputs']

        # quick hack for now: just store first N rows as encountered
        self.learn_index = 0  # current index

        self.learn_every_k = params['learn_row_every_k']  # spread out learning to randomize
        self.curr_k_step = self.learn_every_k  # spread out learning; track current index

        self.N = params['num_rows']  # total number of rows learned

        self.train_weights = False
        self.weight_arr = np.ones((self.N, self.num_sparse_inputs), np.float)  * 1.0 / self.num_sparse_inputs
        self.weight_learn_rate = 0.001

        self.input_arr = np.zeros((self.N, self.num_sparse_inputs), np.int)  # int because this is just indices. dim=num_sparse_inputs since assuming one-hot for now on input
        self.output_arr = np.zeros(self.N, np.int)  # dim=1 since assuming one-hot for now on output

        self.output_prop_count = np.zeros((self.N, self.sparse_io_dim), np.float)
        self.output_prop_arr = np.zeros((self.N, self.sparse_io_dim), np.float)
        # above is: given this input pattern, what is the probability distribution of one-hot sparse outputs
        # more detail:
        #   given that this knn input row is the "winner", i.e. closest nearest neighbor in matches to input,
        #   what is probability of each of the indices in the output to be the winning output?
        #       (for artificial input: often, multiple are "equally close": they should all learn in this case)
        #   in an ideal case with sufficient context: these probability distributions should collapse
        #   worst case is that probability distribution is completely flat

        # debug printing
        self.printed_message = [False, False]
        self.messages = ['SparseBinaryKNN::train: learning is started!',
                         'SparseBinaryKNN::train: learning is completed! Weight learning started!']

        self.reset_stats_every_k_sec = 10
        self.last_stat_reset = time.time()
        self.match_ratio_sum = 0.0
        self.match_ratio_num = 0
        self.sum_tie_matches = 0.0

        self.skipped_train_cnt = 0
        self.done_train_cnt = 0

    def _print_message_once(self, index):
        if not self.printed_message[index]:
            print()
            print(self.messages[index])
            print()
            self.printed_message[index] = True

    #@profile
    def predict(self, knn_input_win_rows):
        '''

        For now this will assume one-hot encoding for each sparse input
        Later, we will need to upgrade to allow multiple selections per sparse input
        Later, we will also need to upgrade to allow returning multiple predictions for the sparse output

        # this should return multiple win rows for multi-predict, so this might change soon:
            win_row = self.knn[tile_n].predict(knn_input_win_rows=knn_input_all_tau)

        :param knn_input_win_rows:
        :return: win_row
        '''

        # "distance metric" is number of exact index matches

        # how to use weights here?

        if self.train_weights:
            # this is 1.0 (no match: error for this sparse index) or 0.0 (match for this sparse index):
            match = ((self.input_arr - knn_input_win_rows) == 0).astype(np.float)
            weighted_match = np.multiply(match, self.weight_arr)
            tmp = np.sum(weighted_match, axis=1)
        else:
            # old, non weighted method:
            # from equal matches: for now since one-hot, just pick first one
            tmp = np.sum((self.input_arr[0:max(self.learn_index, 1), :] - knn_input_win_rows) == 0, axis=1)

        # tmp:
        #   len(tmp) is <= self.N, num_rows (of knn)
        #   tmp[i] is number of indices matched, in range [0, num_sparse_inputs]

        num_top_matches = np.count_nonzero(tmp == np.amax(tmp))
        self.sum_tie_matches += num_top_matches

        # old way: one sample from initial learning of row
        # win_row = self.output_arr[np.argmax(tmp)]

        # new way: use prop
        win_knn_row = np.argmax(tmp)
        win_row = np.argmax(self.output_prop_arr[win_knn_row, :])

        self.match_ratio_sum += np.amax(tmp) * 1.0 / len(knn_input_win_rows)
        self.match_ratio_num += 1

        if time.time() - self.last_stat_reset > self.reset_stats_every_k_sec:

            total_cnt = self.skipped_train_cnt + self.done_train_cnt
            if total_cnt > 0:
                train_accept_ratio = self.done_train_cnt / (self.skipped_train_cnt + self.done_train_cnt)
            else:
                train_accept_ratio = None

            print()
            print('SparseBinaryKNN::predict: stats')
            print('    ', 'average input match for best row:', self.match_ratio_sum / self.match_ratio_num)
            print('    ', 'training knn rows prop complete:', float(self.learn_index / self.N))
            print('    ', 'mean num_ties for max input match:', float(self.sum_tie_matches / self.match_ratio_num))
            print('    ', 'train_accept_ratio:', train_accept_ratio)
            print()
            print('    ', 'sample')
            print('    ', 'num_top_matches', num_top_matches)
            print('    ', 'argmax(tmp)', np.argmax(tmp), 'win_row', win_row)
            print()
            print()
            #print('    ', 'output_prop_arr', np.sum(self.output_prop_arr))
            print()


            self.done_train_cnt = 0
            self.skipped_train_cnt = 0

            self.match_ratio_sum = 0.0
            self.match_ratio_num = 0
            self.sum_tie_matches = 0.0

            self.last_stat_reset = time.time()

        return win_row

    #@profile
    def train(self, knn_input_win_rows, knn_output_win_row):
        '''

        train online on one data point

        For now this will assume one-hot encoding for each sparse input
        Later, we will need to upgrade to allow multiple selections per sparse input

        :param knn_input_win_rows:
        :param knn_output_win_row:
        :return:
        '''

        if self.curr_k_step == self.learn_every_k:
            match = np.sum((self.input_arr[0:max(self.learn_index, 1), :] - knn_input_win_rows) == 0, axis=1)

            if self.learn_index < self.N:
                self._print_message_once(index=0)
                # TODO don't learn if this input + output already in table

                if np.amax(match) < len(knn_input_win_rows):

                    self.input_arr[self.learn_index, :] = knn_input_win_rows[:]
                    self.output_arr[self.learn_index] = knn_output_win_row

                    self.output_prop_count[self.learn_index, knn_output_win_row] = 1
                    self.output_prop_arr[self.learn_index, knn_output_win_row] = 1.0

                    self.learn_index += 1
                    self.done_train_cnt += 1
                else:
                    self.skipped_train_cnt += 1

                self.curr_k_step = 1
            else:
                self._print_message_once(index=1)

                win_knn_row = np.argmax(match)  # in range [0, 200]
                self.output_prop_count[win_knn_row, knn_output_win_row] += 1
                self.output_prop_arr[win_knn_row, :] = self.output_prop_count[win_knn_row, :] * 1.0 / np.sum(self.output_prop_count[win_knn_row, :])

                #self.output_prop_arr[win_knn_row, knn_output_win_row] = (1.0 - self.weight_learn_rate) * self.output_prop_arr[win_knn_row, knn_output_win_row] + self.weight_learn_rate * 1.0

                # also need to use output_prop_arr in predict() function

                if self.train_weights:
                    # train weights!
                    # find row in knn with closest input match, from the subset that have correct output win row
                    # TODO what if tmp1 is empty set below? because none stored with this output?
                    tmp1 = np.nonzero(self.output_arr==knn_output_win_row)[0]

                    # todo how often does this happen? make variable to keep track and print in stats
                    if len(tmp1) > 0:
                        tmp2 = self.input_arr[tmp1, :]

                        match = ((tmp2 - knn_input_win_rows) == 0).astype(np.float)
                        weighted_match = np.multiply(match, self.weight_arr[tmp1, :])
                        tmp3 = np.sum(weighted_match, axis=1)

                        tmp4 = np.argmax(tmp3)
                        knn_row = tmp1[tmp4]

                        # train that row for current input match
                        new_w_goal = weighted_match[tmp4, :]

                        self.weight_arr[knn_row, :] = (1.0 - self.weight_learn_rate) * self.weight_arr[knn_row, :] + self.weight_learn_rate * new_w_goal

                        # make sure weights per knn row stay normalized to 1.0!
                        sum_tmp = np.sum(self.weight_arr[knn_row, :])
                        self.weight_arr[knn_row, :] = self.weight_arr[knn_row, :] * 1.0 / sum_tmp
        else:
            self.curr_k_step += 1
