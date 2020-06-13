

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

        return 0

    def train(self, knn_input_win_rows, knn_output_win_row):
        '''

        train online on one data point

        For now this will assume one-hot encoding for each sparse input
        Later, we will need to upgrade to allow multiple selections per sparse input

        :param knn_input_win_rows:
        :param knn_output_win_row:
        :return:
        '''
