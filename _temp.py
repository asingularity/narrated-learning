
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
    X = np.random.random((1024, 48)).astype(np.float32)

    fps = FPSCounter(params={'display_every_k_seconds': 5})
    linalg.init()

    while True:
        X_gpu = gpuarray.to_gpu(X)
        Y = (-2 * X_gpu).get()
        fps.update()
        #print()
        #print(X[0:4, 0:4])
        #print(Y[0:4, 0:4])
