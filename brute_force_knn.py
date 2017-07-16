
import numpy as np


class knn(object):
    def __init__(self, input_data):
        # KDTree(self.concat_predictor_input_histories[net_index][0:self.predictor_training_history_step, :])
        #self.X = X

        DIM = input_data.shape[1]

        self.X = np.zeros((1, DIM)).astype(np.float32)
        self.input_data_transpose = np.ascontiguousarray(np.transpose(input_data))
        self.term_2 = np.sum(input_data ** 2, axis=1)

    def query_L0(self, input_vectors_list, k):
        assert len(input_vectors_list) == 1, 'brute_force_knn.knn.query: Only 1 point lookup implemented!'
        assert k == 1, 'brute_force_knn.knn.query'
        input_vector = input_vectors_list[0]

        #print 'QUERY ', self.X.shape, input_vector.shape
        # QUERY  (3700, 15) (15,)
        diff = self.X - input_vector
        dists = np.sum(np.fabs(diff), axis=1)

        ind = np.argmin(dists)
        dist = dists[ind]
        return dist, ind

    def query_L2(self, input_vectors_list, k):
        assert len(input_vectors_list) == 1, 'brute_force_knn.knn.query: Only 1 point lookup implemented!'
        assert k == 1, 'brute_force_knn.knn.query'
        input_vector = input_vectors_list[0]


        X = self.X
        X[0, :] = input_vector[:]
        term_1 = np.dot(X, self.input_data_transpose)
        term_1 = -2 * term_1
        term_2 = self.term_2
        term_3 = np.sum(X ** 2, axis=1)[:, np.newaxis]
        dists = term_1 + term_2 + term_3

        #diff = self.X - input_vector
        #dists = np.sum(np.square(diff), axis=1)

        ind = np.argmin(dists)
        dist = dists[0, ind]
        return dist, ind

    def query(self, input_vectors_list, k):
        return self.query_L2(input_vectors_list, k)
