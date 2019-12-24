
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg
import skcuda.misc as misc
from utils.fps_counter import FPSCounter

if __name__ == '__main__':

    input_dim = 48
    num_entries = 8000

    table_i_only = np.random.random((num_entries, input_dim)).astype(np.float32)

    table_i_gpu = gpuarray.to_gpu(np.ascontiguousarray(np.transpose(table_i_only)))
    i_d_t_gpu = table_i_gpu

    fps = FPSCounter(params={'display_every_k_seconds': 5})
    linalg.init()

    while True:
        X = np.random.random((1024, input_dim)).astype(np.float32)

        X_gpu = gpuarray.to_gpu(X)

        term_1 = (-2 * linalg.dot(X_gpu, i_d_t_gpu)).get()
        fps.update()
        #print()
        #print(X[0:4, 0:4])
        #print(Y[0:4, 0:4])
