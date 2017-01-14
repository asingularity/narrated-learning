

from libc.math cimport fabs as c_fabs

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

from cython.parallel import prange
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def knn_query(
          np.ndarray[DTYPE_t, ndim = 2] X,
          np.ndarray[DTYPE_t, ndim=1] input_vector,
          np.ndarray[DTYPE_t, ndim=1] tmp,
          long num_points,
          long dim):

    #assert X.dtype == DTYPE
    #assert input_vector.dtype == DTYPE
    #assert tmp.dtype == DTYPE

    cdef long pt_index, dim_index, min_index
    cdef DTYPE_t min_dist, diff, dist

    min_dist = 9e9
    min_index = -1


    # for pt_index in range(num_points):
    for pt_index in prange(num_points, nogil=True, schedule='static'):
        dist = 0.0
        for dim_index in range(dim):
            diff = X[pt_index, dim_index] - input_vector[dim_index]

            dist = dist + c_fabs(diff)

        tmp[pt_index] = dist


    for pt_index in range(num_points):
        dist = tmp[pt_index]
        if dist < min_dist:
            min_dist = dist
            min_index = pt_index

    return min_dist, min_index
