
from libcpp cimport bool

import numpy as np
cimport numpy as np

#DTYPE = np.int
#ctypedef np.int DTYPE_t

from cython.parallel import prange
cimport cython



@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

# https://stackoverflow.com/questions/21851985/difference-between-np-int-np-int-int-and-np-int-t-in-cython

def sum_match(np.ndarray[np.int32_t, ndim=2] arr1,
              np.ndarray[np.int32_t, ndim=2] arr2):

    tmp = np.sum((arr1 - arr2) == 0, axis=1)
    return tmp
    #tmp =
    #for k in prange(arr1.shape[0]):
