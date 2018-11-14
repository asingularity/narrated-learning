
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

        self.argmax_by_row = np.zeros(num_rows, np.int)
        self.max_by_row = np.zeros(num_rows, np.float32)

        self.dist_mat = np.ones((num_rows, num_rows), np.float32) * init_dists_val

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

        row_1 = np.argmax(self.max_by_row)
        max_dist = self.max_by_row[row_1]
        row_0 = self.argmax_by_row[row_1]

        return max_dist, row_0, row_1

    def post_init(self):
        for r in range(self.num_rows):
            self.argmin_by_row[r] = np.argmin(self.dist_mat[r, :])
            self.min_by_row[r] = self.dist_mat[r, self.argmin_by_row[r]]
            self.argmax_by_row[r] = np.argmax(self.dist_mat[r, :])
            self.max_by_row[r] = self.dist_mat[r, self.argmax_by_row[r]]

    def set_row_dists(self, row_index, new_dists, fast_init=False):
        '''

        :param row_index:
        :param new_dists:
        :return:
        '''

        #print('START')
        num_bad_1 = 0
        num_bad_2 = 0

        self.dist_mat[row_index, :] = new_dists
        self.dist_mat[:, row_index] = new_dists

        if not fast_init:
            tmp_min_ind = np.argmin(new_dists)
            tmp_max_ind = np.argmax(new_dists)
            tmp_min = new_dists[tmp_min_ind]
            tmp_max = new_dists[tmp_max_ind]

            self.argmin_by_row[row_index] = tmp_min_ind
            self.min_by_row[row_index] = tmp_min

            self.argmax_by_row[row_index] = tmp_max_ind
            self.max_by_row[row_index] = tmp_max

            for r in range(self.num_rows):
                if not r == row_index:

                    if self.argmin_by_row[r] == row_index and new_dists[r] > self.min_by_row[r]:
                        tmp_ind = np.argmin(self.dist_mat[r, :])
                        tmp_min = self.dist_mat[r, tmp_ind]

                        num_bad_1 += 1

                        self.argmin_by_row[r] = tmp_ind
                        self.min_by_row[r] = tmp_min
                    else:
                        if new_dists[r] < self.min_by_row[r]:
                            self.argmin_by_row[r] = row_index
                            self.min_by_row[r] = new_dists[r]

                    if self.argmax_by_row[r] == row_index and new_dists[r] < self.max_by_row[r]:
                        tmp_ind = np.argmax(self.dist_mat[r, :])
                        tmp_max = self.dist_mat[r, tmp_ind]

                        num_bad_2 += 1

                        self.argmax_by_row[r] = tmp_ind
                        self.max_by_row[r] = tmp_max
                    else:
                        if new_dists[r] > self.max_by_row[r]:
                            self.argmax_by_row[r] = row_index
                            self.max_by_row[r] = new_dists[r]

            # TODO debug why slow at start:
            # print(num_bad_1, num_bad_2)


def test_row_dist(num_rows, test_sec, test_frames, dumb):
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

        if frame in test_frames:
            print('    ', frame, new_min, new_max)

        frame += 1

if __name__ == '__main__':

    print()
    print('********')
    print()
    print('...slow...')
    print()
    test_row_dist(num_rows=200, test_sec=4, test_frames = [1000, 5000, 10000], dumb=1)
    print()
    print('...fast...')
    print()
    test_row_dist(num_rows=200, test_sec=4, test_frames = [1000, 5000, 10000], dumb=0)


    print()
    print('********')
    print()
    print('...slow...')
    print()
    test_row_dist(num_rows=20000, test_sec=8, test_frames = [1, 5, 10, 30, 50], dumb=1)
    print()
    print('...fast...')
    print()
    test_row_dist(num_rows=20000, test_sec=20, test_frames = [1, 5, 10, 30, 50], dumb=0)
