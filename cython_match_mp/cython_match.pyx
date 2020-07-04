
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

def sum_match(np.ndarray[np.int32_t, ndim=2] arr1,
              np.ndarray[np.int32_t, ndim=2] arr2,
              np.ndarray[np.int32_t, ndim=1] arr_out,
              np.int32_t num_knn,
              np.int32_t N):

    #tmp = np.sum((arr1 - arr2) == 0, axis=1)
    #return tmp

    cols = arr1.shape[1]

    cdef np.int32_t n_start, n_end, r, knn_ind, c

    for knn_ind in prange(num_knn,  nogil=True, schedule='static', num_threads=20):
        n_start = knn_ind * N;
        n_end = (knn_ind + 1) * N;

        # can't do this without gil:
        # arr_out[n_start:n_end] = np.sum((arr1[n_start:n_end, :] - arr2[n_start:n_end, :]) == 0, axis=1)

        for r in range(n_start, n_end):
             arr_out[r] = 0
             for c in range(cols):
                 arr_out[r] += (arr1[r, c] - arr2[r, c]) == 0

    return 0