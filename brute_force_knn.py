
import numpy as np


class knn(object):
    def __init__(self, X):
        # KDTree(self.concat_predictor_input_histories[net_index][0:self.predictor_training_history_step, :])
        self.X = X

    def query(self, input_vectors_list, k):
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
