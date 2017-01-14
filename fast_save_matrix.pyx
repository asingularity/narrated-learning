import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function


def savetxt(fname, np.ndarray[DTYPE_t, ndim = 2] data):
    f = open(fname, 'w')
    cdef long rshape = data.shape[0]
    cdef long cshape = data.shape[1]

    for row in range(rshape):
        for col in range(cshape):
            f.write("%f "% data[row, col])
        f.write('\n')
    f.close()
