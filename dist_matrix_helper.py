
import numpy as np
from time import time
from utils.fps_counter import FPSCounter


class DumbDistMatrixHelper(object):
    def __init__(self, num_rows, init_dists_val):
        '''

        :param rows:
        :param init_val:
        :return:
        '''

        self.dist_mat = np.ones((num_rows, num_rows), np.float32) * init_dists_val

    def get_min_dist(self):
        '''

        :return: min_dist, row_0, row_1
        '''
        min_dist_ind = np.unravel_index(np.argmin(self.dist_mat, axis=None), self.dist_mat.shape)
        min_dist = self.dist_mat[min_dist_ind[0], min_dist_ind[1]]
        return min_dist, min_dist_ind[0], min_dist_ind[1]

    def get_max_dist(self):
        '''

        :return: max_dist, row_0, row_1
        '''
        max_dist_ind = np.unravel_index(np.argmax(self.dist_mat, axis=None), self.dist_mat.shape)
        max_dist = self.dist_mat[max_dist_ind[0], max_dist_ind[1]]
        return max_dist, max_dist_ind[0], max_dist_ind[1]

    def set_row_dists(self, row_index, new_dists):
        '''

        :param row_index:
        :param new_dists:
        :return:
        '''
        self.dist_mat[row_index, :] = new_dists
        self.dist_mat[:, row_index] = new_dists


class DistMatrixHelper(object):
    def __init__(self, num_rows, init_dists_val):
        '''

        :param rows:
        :param init_val:
        :return:
        '''
        self.num_rows = num_rows
        self.argmin_by_row = np.zeros(num_rows, np.int)
        self.min_by_row = np.zeros(num_rows, np.float32)

    def get_min_dist(self):
        '''

        :return: min_dist, row_0, row_1
        '''

        row_1 = np.argmin(self.min_by_row)
        min_dist = self.min_by_row[row_1]
        row_0 = self.argmin_by_row[row_1]

        return min_dist, row_0, row_1

    def get_max_dist(self):
        '''

        :return: max_dist, row_0, row_1
        '''

        return None, None, None

    def set_row_dists(self, row_index, new_dists):
        '''

        :param row_index:
        :param new_dists:
        :return:
        '''

        tmp_ind = np.argmin(new_dists)
        tmp_min = new_dists[tmp_ind]
        if tmp_min < self.min_by_row[row_index]:
            self.argmin_by_row[row_index] = tmp_ind
            self.min_by_row[row_index] = tmp_min

        for r in range(self.num_rows):
            if not r == row_index:
                if new_dists[r] < self.min_by_row[r]:
                    self.min_by_row[r] = new_dists[r]
                    self.argmin_by_row[r] = row_index


def test_row_dist(num_rows, test_sec, dumb):
    np.random.seed(0)

    fps = FPSCounter(params={'display_every_k_seconds': 2})

    if dumb:
        h1 = DumbDistMatrixHelper(num_rows=num_rows, init_dists_val=0)
    else:
        h1 = DistMatrixHelper(num_rows=num_rows, init_dists_val=0)

    t0 = time()
    frame = 0

    while time() < t0 + test_sec:

        h1.set_row_dists(row_index=np.random.randint(0, num_rows, size=(1), dtype=np.int)[0],
                         new_dists=np.random.random(size=(num_rows)).astype(np.float32))

        new_min = h1.get_min_dist()
        new_max = h1.get_max_dist()
        #print (new_min, new_max)

        fps.update()

        if frame == 0 or frame == 5 or frame == 10:
            print(frame, new_min, new_max)

        frame += 1

if __name__ == '__main__':

    num_rows_test = 10000
    test_sec_test = 6

    test_row_dist(num_rows=num_rows_test, test_sec=test_sec_test, dumb=True)
