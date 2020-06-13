
import numpy as np


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

        self.learn_every_k = 1  # spread out learning to randomize
        self.curr_k_step = self.learn_every_k  # spread out learning; track current index

        self.N = 2000  # total number of rows learned

        self.input_arr = np.zeros((self.N, self.num_sparse_inputs), np.int)  # int because this is just indices. dim=num_sparse_inputs since assuming one-hot for now on input
        self.output_arr = np.zeros(self.N, np.int)  # dim=1 since assuming one-hot for now on output

        # debug printing
        self.printed_message = [False, False]
        self.messages = ['SparseBinaryKNN::train: learning is started!',
                         'SparseBinaryKNN::train: learning is completed!']

    def _print_message_once(self, index):
        if not self.printed_message[index]:
            print()
            print(self.messages[index])
            print()
            self.printed_message[index] = True

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

        # from equal matches: for now since one-hot, just pick first one
        tmp = np.sum((self.input_arr - knn_input_win_rows) == 0, axis=1)
        win_row = self.output_arr[np.argmax(tmp)]

        return win_row

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
            if self.learn_index < self.N:
                self._print_message_once(index=0)
                self.input_arr[self.learn_index, :] = knn_input_win_rows[:]
                self.output_arr[self.learn_index] = knn_output_win_row

                self.learn_index += 1

                self.curr_k_step = 1
            else:
                self._print_message_once(index=1)
        else:
            self.curr_k_step += 1
