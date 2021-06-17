
from libcpp cimport bool

import numpy as np
cimport numpy as np
cimport openmp

#DTYPE = np.int
#ctypedef np.int DTYPE_t

from cython.parallel import prange
cimport cython


from libc.math cimport fabs








@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def cy_tile_the_input(np.ndarray[np.float32_t, ndim=1] input_events_p,
                      np.ndarray[np.float32_t, ndim=1] input_events_n,
                      np.ndarray[np.int64_t, ndim=1] event_coords_r,
                      np.ndarray[np.int64_t, ndim=1] event_coords_c,
                      np.int64_t num_tiles_NxN,
                      np.int64_t tile_input_state_dim,
                      np.int64_t tile_dim_NxN,
                      np.ndarray[np.float32_t, ndim=2] input_states_tiles):
    pass




@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def cy_compute_error_measures(np.ndarray[np.float32_t, ndim=2] rfs,
                              np.ndarray[np.float32_t, ndim=2] input_states_tiles,
                              np.float32_t fix_offset,
                              np.ndarray[np.float32_t, ndim=2] RF_norm_dot):

    # for now, just RF-norm-dot
    pass





@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def cy_kmeans_do_learning(np.ndarray[np.int64_t, ndim=1] best_rf_per_tile,
                          np.ndarray[np.float32_t, ndim=2] weights,
                          np.ndarray[np.float32_t, ndim=2] input_states_tiles):
    pass






