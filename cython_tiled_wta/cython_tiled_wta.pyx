
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

    # input_states_tiles = np.zeros((int(self.num_tiles_NxN * self.num_tiles_NxN), self.tile_input_state_dim), np.float32)
    
    cdef np.int32_t n_events, event_r, event_c, tile_index_r, tile_index_c, tile_index, r_in_tile, c_in_tile, index_in_tile
    cdef np.float32_t event_val_p, event_val_n

    n_events = input_events_p.shape[0]

    for k in range(n_events):
        event_val_p = input_events_p[k]
        event_val_n = input_events_n[k]
        event_r = event_coords_r[k]
        event_c = event_coords_c[k]
    
        # find tile index for this event_r, event_c
        tile_index_r = event_r / tile_dim_NxN
        tile_index_c = event_c / tile_dim_NxN
        tile_index = tile_index_c * num_tiles_NxN + tile_index_r

        # set input states
        # self.tile_input_state_dim = 2 * self.tile_dim_NxN * self.tile_dim_NxN
        
        r_in_tile = event_r % tile_dim_NxN
        c_in_tile = event_c % tile_dim_NxN
        index_in_tile = r_in_tile * tile_dim_NxN + c_in_tile

        input_states_tiles[tile_index, index_in_tile] = event_val_p
        input_states_tiles[tile_index, index_in_tile + tile_dim_NxN * tile_dim_NxN] = event_val_n




@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cy_compute_error_measures(np.ndarray[np.float32_t, ndim=2] rfs,
                              np.ndarray[np.float32_t, ndim=2] input_states_tiles,
                              np.float32_t fix_offset,
                              np.ndarray[np.float32_t, ndim=2] RF_norm_dot):

    # for now, just RF-norm-dot
    cdef np.int32_t num_rfs, num_tiles, input_dim
    cdef np.int32_t rf_index, tile_index, k

    cdef np.float32_t tmp_sum, rf_sum

    num_tiles = RF_norm_dot.shape[0]
    num_rfs = RF_norm_dot.shape[1]
    input_dim = input_states_tiles.shape[1]

    #for rf_index in prange(num_rfs, nogil=True, schedule='dynamic', num_threads=4):
    for rf_index in range(num_rfs):
        
        rf_sum = 0.0
        for k in range(input_dim):
            rf_sum = rf_sum + rfs[rf_index, k]

        for tile_index in range(num_tiles):
            tmp_sum = 0.0

            for k in range(input_dim):
                tmp_sum = tmp_sum + rfs[rf_index, k] * input_states_tiles[tile_index, k]
            
            tmp_sum = tmp_sum / (rf_sum + fix_offset)

            RF_norm_dot[tile_index, rf_index] = -tmp_sum

    # RF_norm_dot = -np.divide(np.sum(np.multiply(rfs, input_arr), axis=1), np.sum(rfs, axis=1)+fix_offset)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cy_kmeans_do_learning(np.ndarray[np.int64_t, ndim=1] best_rf_per_tile,
                          np.ndarray[np.float32_t, ndim=2] weights,
                          np.ndarray[np.float32_t, ndim=2] input_states_tiles,
                          np.float32_t lr):
    

    # best_rf_per_tile: (256,) 
    # weights: (400, 128)
    # input_states_tiles: (256, 128)

    cdef np.int32_t num_rfs, num_tiles, input_dim

    cdef np.int32_t tile_index, best_rf, k

    num_tiles = best_rf_per_tile.shape[0]
    num_rfs = weights.shape[0]
    input_dim = weights.shape[1]

    for tile_index in range(num_tiles):
    #for tile_index in prange(num_tiles, nogil=True, schedule='dynamic', num_threads=4):
        best_rf = best_rf_per_tile[tile_index]

        for k in range(input_dim):
            weights[best_rf, k] = lr * input_states_tiles[tile_index, k] + (1.0 - lr) * weights[best_rf, k]


    
    # self.weights[best_rf, :] = lr * input_state + (1.0 - lr) * self.weights[best_rf, :]
