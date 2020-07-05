
from libcpp cimport bool

import numpy as np
cimport numpy as np
cimport openmp

#DTYPE = np.int
#ctypedef np.int DTYPE_t

from cython.parallel import prange
cimport cython


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

# https://stackoverflow.com/questions/21851985/difference-between-np-int-np-int-int-and-np-int-t-in-cython

def test(list list_of_2darrays):
    cdef np.ndarray[ndim=2, dtype=np.float64_t] buff2d # check your dtype
    cdef int i, j

    for buff2d in list_of_2darrays:
        for i in range(buff2d.shape[0]):
            for j in range(buff2d.shape[1]):
                buff2d[i, j] += i + j

# def sum_match_exp_no_work(list list_of_input_arr1,
#               list list_of_input_arr2,
#               np.ndarray[np.int32_t, ndim=1] arr_out,
#               np.int32_t num_knn,
#               np.int32_t N,
#               np.int32_t cols):

#     cdef np.ndarray[np.int32_t, ndim=2] buff2d_1
#     cdef np.ndarray[np.int32_t, ndim=2] buff2d_2
#     cdef np.int32_t tmp

#     for knn_ind in prange(num_knn,  nogil=True, schedule='static', num_threads=20):

#         buff2d_1 = list_of_input_arr1[knn_ind]
#         buff2d_2 = list_of_input_arr2[knn_ind]

 #        for r in range(0, N):
 #             arr_out[knn_ind * N + r] = 0
 #             for c in range(cols):
 #                 arr_out[knn_ind * N + r] += (buff2d_1[r, c] - buff2d_2[r, c]) == 0

  #   return 0

def sum_match(np.ndarray[np.int32_t, ndim=2] arr1,
              np.ndarray[np.int32_t, ndim=2] arr2,
              np.ndarray[np.int32_t, ndim=1] arr_out,
              np.int32_t num_knn,
              np.int32_t N):

    #tmp = np.sum((arr1 - arr2) == 0, axis=1)
    #return tmp

    cols = arr1.shape[1]

    cdef np.int32_t n_start, n_end, r, knn_ind, c

    for knn_ind in prange(num_knn,  nogil=True, schedule='dynamic', num_threads=20):
        n_start = knn_ind * N;
        n_end = (knn_ind + 1) * N;

        # can't do this without gil:
        # arr_out[n_start:n_end] = np.sum((arr1[n_start:n_end, :] - arr2[n_start:n_end, :]) == 0, axis=1)

        for r in range(n_start, n_end):
             arr_out[r] = 0
             for c in range(cols):
                 # TODO optimize here: arr2 doesn't longer need to be expanded!
                 arr_out[r] += (arr1[r, c] - arr2[knn_ind, c]) == 0

    return 0