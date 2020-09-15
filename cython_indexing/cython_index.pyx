
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

def do_indexing(np.ndarray[np.int32_t, ndim=2] index_arr,
                np.ndarray[np.float32_t, ndim=2] dst_arr,
                np.ndarray[np.float32_t, ndim=1] src_arr):

    cdef np.int32_t dim_r = index_arr.shape[0]
    cdef np.int32_t dim_c = index_arr.shape[1]

    cdef np.int32_t r, c

    #for r in range(dim_r):
    for r in prange(dim_r,  nogil=True, schedule='dynamic', num_threads=20):
        for c in range(dim_c):
            dst_arr[r, c] = src_arr[index_arr[r, c]]

    return 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def get_match(np.ndarray[np.float32_t, ndim=2] weights_arr,
              np.ndarray[np.float32_t, ndim=2] inputs_arr,
              np.ndarray[np.float32_t, ndim=1] match_arr):

    cdef np.int32_t dim_r = weights_arr.shape[0]
    cdef np.int32_t dim_c = weights_arr.shape[1]

    cdef np.int32_t r, c
    cdef np.float32_t tmp_sum, weights_sum

    for r in range(dim_r):
    #for r in prange(dim_r,  nogil=True, schedule='dynamic', num_threads=20):
        tmp_sum = 0
        weights_sum = 0

        for c in range(dim_c):
            tmp_sum += weights_arr[r, c] * inputs_arr[r, c]
            weights_sum += weights_arr[r, c]

        match_arr[r] = tmp_sum / weights_sum

    return 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def get_match_parallel(np.ndarray[np.float32_t, ndim=2] weights_arr,
                       np.ndarray[np.float32_t, ndim=2] inputs_arr,
                       np.ndarray[np.float32_t, ndim=1] match_arr,
                       np.ndarray[np.float32_t, ndim=2] tmp_match_arr_1,
                       np.ndarray[np.float32_t, ndim=2] tmp_match_arr_2,
                       np.int32_t rows_per_thread,
                       np.int32_t num_threads):

    cdef np.int32_t dim_r = weights_arr.shape[0]
    cdef np.int32_t dim_c = weights_arr.shape[1]

    cdef np.int32_t r, c, thr

    #for thr in prange(num_threads,  nogil=True, schedule='dynamic', num_threads=num_threads):
    for thr in range(num_threads):

        for r in range(thr * rows_per_thread, (thr + 1) * rows_per_thread):
            for c in range(dim_c):
                tmp_match_arr_1[r, 0] += weights_arr[r, c] * inputs_arr[r, c]
                tmp_match_arr_2[r, 0] += weights_arr[r, c]

    for r in range(dim_r):
        match_arr[r] = tmp_match_arr_1[r, 0] / tmp_match_arr_2[r, 0]

    return 0


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def multiply_parallel(np.ndarray[np.float32_t, ndim=2] weights_arr,
                      np.ndarray[np.float32_t, ndim=2] inputs_arr,
                      np.ndarray[np.float32_t, ndim=2] results_arr,
                      np.int32_t rows_per_thread,
                      np.int32_t num_threads):

    cdef np.int32_t dim_r = weights_arr.shape[0]
    cdef np.int32_t dim_c = weights_arr.shape[1]

    cdef np.int32_t r, c, thr

    for thr in prange(num_threads,  nogil=True, schedule='dynamic', num_threads=num_threads):
        for r in range(thr * rows_per_thread, (thr + 1) * rows_per_thread):
            for c in range(dim_c):
                results_arr[r, c] = weights_arr[r, c]  * inputs_arr[r, c]

    return 0

































