
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
def cy_weigh_in_with_out(np.ndarray[np.float32_t, ndim=3] input_events,
                         np.int64_t  input_num_tiles_NxN,
                         np.int64_t  input_num_rfs_per_tile,
                         np.ndarray[np.float32_t, ndim=3] output_events,
                         np.ndarray[np.float32_t, ndim=2] output_weights,
                         np.int64_t  output_num_tiles_NxN,
                         np.int64_t  output_num_rfs_per_tile,
                         np.int64_t  output_tile_dim_NxN):

    # input_events shape: (32, 32, 800)
    # output_events shape: (16, 16, 800)
    # output_weights shape: (800, 3200)  # num rfs output layer X (input dim pre output tile == 2*2*800)

    # we are going to change input_events in-place by weight

    assert input_events.shape[0] == input_num_tiles_NxN
    assert input_events.shape[1] == input_num_tiles_NxN
    assert input_events.shape[2] == input_num_rfs_per_tile

    assert output_events.shape[0] == output_num_tiles_NxN
    assert output_events.shape[1] == output_num_tiles_NxN
    assert output_events.shape[2] == output_num_rfs_per_tile

    assert output_weights.shape[0] == output_num_rfs_per_tile
    assert output_weights.shape[1] == output_tile_dim_NxN * output_tile_dim_NxN * input_num_rfs_per_tile

    cdef np.int32_t output_flat_in_dim, out_tile_r, out_tile_c, in_tile_r_start, in_tile_c_start, in_tile_r, in_tile_c, in_rf, out_rf, in_index
    cdef np.float32_t w_tmp

    output_flat_in_dim = output_tile_dim_NxN * output_tile_dim_NxN * input_num_rfs_per_tile

    for out_tile_r in range(output_num_tiles_NxN):
    #for out_tile_r in prange(output_num_tiles_NxN, nogil=True, schedule='dynamic', num_threads=22):
        for out_tile_c in range(output_num_tiles_NxN):
            in_tile_r_start = out_tile_r * output_tile_dim_NxN
            in_tile_c_start = out_tile_c * output_tile_dim_NxN

            for out_rf in range(output_num_rfs_per_tile):
                if output_events[out_tile_r, out_tile_c, out_rf] == 1:
                    in_index = 0

                    # TODO found the winning output RF for this output tile

                    # TODO for every input tile:
                    for in_tile_r in range(in_tile_r_start, in_tile_r_start + output_tile_dim_NxN):
                        for in_tile_c in range(in_tile_c_start, in_tile_c_start + output_tile_dim_NxN):
                            # TODO find winner
                            for in_rf in range(input_num_rfs_per_tile):

                                if input_events[in_tile_r, in_tile_c, in_rf] == 1:
                                    # TODO found winner
                                    input_events[in_tile_r, in_tile_c, in_rf] = input_events[in_tile_r, in_tile_c, in_rf] * output_weights[out_rf, in_index]

                                in_index = in_index + 1

# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!

def cy_weigh_in_with_out_L0(np.ndarray[np.float32_t, ndim=3] input_events,
                            np.int64_t  input_num_tiles_NxN,
                            np.int64_t  input_num_rfs_per_tile,
                            np.ndarray[np.float32_t, ndim=3] output_events,
                            np.ndarray[np.float32_t, ndim=2] output_weights,
                            np.int64_t  output_num_tiles_NxN,
                            np.int64_t  output_num_rfs_per_tile,
                            np.int64_t  output_tile_dim_NxN):

    # input_events shape: (32, 32, 800)
    # output_events shape: (16, 16, 800)
    # output_weights shape: (800, 3200)  # num rfs output layer X (input dim pre output tile == 2*2*800)

    # we are going to change input_events in-place by weight

    assert input_events.shape[0] == input_num_tiles_NxN
    assert input_events.shape[1] == input_num_tiles_NxN
    assert input_events.shape[2] == input_num_rfs_per_tile

    assert output_events.shape[0] == output_num_tiles_NxN
    assert output_events.shape[1] == output_num_tiles_NxN
    assert output_events.shape[2] == output_num_rfs_per_tile

    assert output_weights.shape[0] == output_num_rfs_per_tile
    assert output_weights.shape[1] == 2 * output_tile_dim_NxN * output_tile_dim_NxN * input_num_rfs_per_tile

    cdef np.int32_t output_flat_in_dim, out_tile_r, out_tile_c, in_tile_r_start, in_tile_c_start, in_tile_r, in_tile_c, in_rf, out_rf, in_index
    cdef np.float32_t w_tmp

    output_flat_in_dim = output_tile_dim_NxN * output_tile_dim_NxN * input_num_rfs_per_tile

    for out_tile_r in range(output_num_tiles_NxN):
    #for out_tile_r in prange(output_num_tiles_NxN, nogil=True, schedule='dynamic', num_threads=22):
        for out_tile_c in range(output_num_tiles_NxN):
            in_tile_r_start = out_tile_r * output_tile_dim_NxN
            in_tile_c_start = out_tile_c * output_tile_dim_NxN

            for out_rf in range(output_num_rfs_per_tile):
                if output_events[out_tile_r, out_tile_c, out_rf] > 0:
                    in_index = 0

                    # TODO found the winning output RF for this output tile

                    # TODO for every input tile:
                    for in_tile_r in range(in_tile_r_start, in_tile_r_start + output_tile_dim_NxN):
                        for in_tile_c in range(in_tile_c_start, in_tile_c_start + output_tile_dim_NxN):
                            # TODO find winner
                            for in_rf in range(input_num_rfs_per_tile):

                                #if input_events[in_tile_r, in_tile_c, in_rf] == 1:
                                #    # TODO found winner
                                input_events[in_tile_r, in_tile_c, in_rf] = input_events[in_tile_r, in_tile_c, in_rf] * output_weights[out_rf, in_index]

                                in_index = in_index + 1
