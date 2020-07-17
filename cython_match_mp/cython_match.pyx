
from libcpp cimport bool

import numpy as np
cimport numpy as np
cimport openmp

#DTYPE = np.int
#ctypedef np.int DTYPE_t

from cython.parallel import prange
cimport cython


# https://stackoverflow.com/questions/21851985/difference-between-np-int-np-int-int-and-np-int-t-in-cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def test(list list_of_2darrays):
    cdef np.ndarray[ndim=2, dtype=np.float64_t] buff2d # check your dtype
    cdef int i, j

    for buff2d in list_of_2darrays:
        for i in range(buff2d.shape[0]):
            for j in range(buff2d.shape[1]):
                buff2d[i, j] += i + j


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def sum_match(np.ndarray[np.int32_t, ndim=2] arr1,
              np.ndarray[np.int32_t, ndim=2] arr2,
              np.ndarray[np.int32_t, ndim=1] arr_out,
              np.int32_t num_knn,
              np.int32_t N,
              np.ndarray[np.int32_t, ndim=1] learn_index):

    #tmp = np.sum((arr1 - arr2) == 0, axis=1)
    #return tmp

    cols = arr1.shape[1]

    cdef np.int32_t n_start, n_end, r, knn_ind, c

    for knn_ind in prange(num_knn,  nogil=True, schedule='dynamic', num_threads=20):
        n_start = knn_ind * N
        n_end = knn_ind * N + learn_index[knn_ind]
        # n_end = (knn_ind + 1) * N

        # can't do this without gil:
        # also, this is outdated: we no longer send in an expanded (tiled) arr2
        # arr_out[n_start:n_end] = np.sum((arr1[n_start:n_end, :] - arr2[n_start:n_end, :]) == 0, axis=1)

        for r in range(n_start, n_end):
             arr_out[r] = 0
             for c in range(cols):
                 arr_out[r] += (arr1[r, c] - arr2[knn_ind, c]) == 0

    return 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def learn_update_DUMB(np.int32_t num_knn,
                      np.int32_t N,
                      np.ndarray[np.int32_t, ndim=1] tmp,
                      np.ndarray[np.int32_t, ndim=1] learn_index,
                      np.ndarray[np.int32_t, ndim=2] input_arr,
                      np.int32_t len_input,
                      np.ndarray[np.int32_t, ndim=2] knn_input_win_rows_2d_arr,
                      np.ndarray[np.float32_t, ndim=2] output_prop_count,
                      np.ndarray[np.float32_t, ndim=2] output_prop_arr,
                      np.ndarray[np.int32_t, ndim=1] knn_output_win_row_arr,
                      np.ndarray[np.int32_t, ndim=1] win_knn_rows_arr,
                      np.ndarray[np.float32_t, ndim=1] unused_ignore):

    for knn in range(num_knn):
        match = tmp[knn * N:(knn + 1) * N]

        if learn_index[knn] < N:
            if np.amax(match) < len_input:  # not perfect match
                input_arr[knn * N + learn_index[knn], :] = knn_input_win_rows_2d_arr[knn, :]
                output_prop_count[knn * N + learn_index[knn], knn_output_win_row_arr[knn]] = 1
                output_prop_arr[knn * N + learn_index[knn], knn_output_win_row_arr[knn]] = 1.0

                learn_index[knn] += 1
        else:
            output_prop_count[win_knn_rows_arr[knn], knn_output_win_row_arr[knn]] += 1
            output_prop_arr[win_knn_rows_arr[knn], :] = output_prop_count[win_knn_rows_arr[knn], :] * 1.0 / np.sum(output_prop_count[win_knn_rows_arr[knn], :])


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def learn_update(np.int32_t num_knn,
                 np.int32_t N,
                 np.ndarray[np.int32_t, ndim=1] tmp,
                 np.ndarray[np.int32_t, ndim=1] learn_index,
                 np.ndarray[np.int32_t, ndim=2] input_arr,
                 np.int32_t len_input,
                 np.ndarray[np.int32_t, ndim=2] knn_input_win_rows_2d_arr,
                 np.ndarray[np.int32_t, ndim=1] knn_output_win_row_arr,
                 np.ndarray[np.int32_t, ndim=1] win_knn_rows_arr,
                 np.ndarray[np.int32_t, ndim=1] output_prop_count_incr_rows,
                 np.ndarray[np.int32_t, ndim=1] output_prop_count_incr_cols,
                 np.ndarray[np.int32_t, ndim=1] output_prop_count_incr_vals):

    cdef np.int32_t r1
    r1 = knn_input_win_rows_2d_arr.shape[1]
    cdef np.int32_t max_match, knn, k, k2

    for knn in prange(num_knn,  nogil=True, schedule='dynamic', num_threads=20):
        max_match = 0

        # for this knn, find best matching input row

        for k in range(knn * N, knn * N + learn_index[knn]):
            if tmp[k] > max_match:
                max_match = tmp[k]

        if learn_index[knn] < N:

            # if not perfect match, store this row and update output prop count and arr for this row
            if max_match < len_input:
                for k in range(r1):
                    input_arr[knn * N + learn_index[knn], k] = knn_input_win_rows_2d_arr[knn, k]

                # output_prop_count[knn * N + learn_index[knn], knn_output_win_row_arr[knn]] = 1

                output_prop_count_incr_rows[knn] = knn * N + learn_index[knn]   # NOTE: absolute indexing for ease of assignment later
                output_prop_count_incr_cols[knn] = knn_output_win_row_arr[knn]
                output_prop_count_incr_vals[knn] = 1

                learn_index[knn] += 1
            else:

                # this is new:

                output_prop_count_incr_rows[knn] = knn * N + win_knn_rows_arr[knn]
                output_prop_count_incr_cols[knn] = knn_output_win_row_arr[knn]
                output_prop_count_incr_vals[knn] = 1
        else:
            # update output prop count
            # output_prop_count[knn * N + win_knn_rows_arr[knn], knn_output_win_row_arr[knn]] += 1

            output_prop_count_incr_rows[knn] = knn * N + win_knn_rows_arr[knn]
            output_prop_count_incr_cols[knn] = knn_output_win_row_arr[knn]
            output_prop_count_incr_vals[knn] = 1





































