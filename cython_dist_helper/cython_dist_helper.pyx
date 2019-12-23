
from libcpp cimport bool

import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

from cython.parallel import prange
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def set_row_dists(long row_index,
                  np.ndarray[DTYPE_t, ndim=1] new_dists,
                  np.ndarray[DTYPE_t, ndim=2] dist_mat,
                  np.ndarray[long, ndim=1] argmin_by_row,
                  np.ndarray[DTYPE_t, ndim=1] min_by_row,
                  np.ndarray[long, ndim=1] argmax_by_row,
                  np.ndarray[DTYPE_t, ndim=1] max_by_row,
                  int num_rows,
                  int fast_init=0):

    dist_mat[row_index, :] = new_dists
    dist_mat[:, row_index] = new_dists

    if not fast_init:
        # min and max values and indices of new dists to updated row (row_index)
        tmp_min_ind = np.argmin(new_dists)
        tmp_max_ind = np.argmax(new_dists)
        tmp_min = new_dists[tmp_min_ind]
        tmp_max = new_dists[tmp_max_ind]

        # for the updated row, new min is the min
        argmin_by_row[row_index] = tmp_min_ind
        min_by_row[row_index] = tmp_min

        # for the updated row, new max is the max
        argmax_by_row[row_index] = tmp_max_ind
        max_by_row[row_index] = tmp_max

        for r in range(num_rows):
            # for all rows except the newly updated row
            if not r == row_index:

                # if updated row was previously the min dist index for current row in loop, but it is no longer
                if argmin_by_row[r] == row_index and new_dists[r] > min_by_row[r]:

                    # find new min, taking update into account
                    tmp_ind = np.argmin(dist_mat[r, :])
                    tmp_min = dist_mat[r, tmp_ind]

                    # update current row's min dist index / value to the new one after finding this new min
                    argmin_by_row[r] = tmp_ind
                    min_by_row[r] = tmp_min

                    # debugging
                else:
                    # if updated row becomes the new min
                    if new_dists[r] < min_by_row[r]:
                        argmin_by_row[r] = row_index
                        min_by_row[r] = new_dists[r]

                # if updated row was previously the max dist index for current row in loop, but it is no longer
                if argmax_by_row[r] == row_index and new_dists[r] < max_by_row[r]:

                    # find new max, taking update into account
                    tmp_ind = np.argmax(dist_mat[r, :])
                    tmp_max = dist_mat[r, tmp_ind]

                    # update current row's max dist index / value to the new one after finding this new max
                    argmax_by_row[r] = tmp_ind
                    max_by_row[r] = tmp_max

                    # debugging
                else:
                    # if updated row becomes the new max
                    if new_dists[r] > max_by_row[r]:
                        argmax_by_row[r] = row_index
                        max_by_row[r] = new_dists[r]

    return 1