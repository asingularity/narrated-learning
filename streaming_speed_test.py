
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg
import skcuda.misc as misc
import random
from utils.fps_counter import FPSCounter

`

if __name__ == '__main__':
    '''
    Goal of this experiment:
    How many samples per second can we stream?

    I.e.
    1 sample - 10 bit

    Equivalent: how much time does it take to send one to GPU, and receive one back.

    I/O Sample rate.
    How soon can I send next sample.
    It does not matter how fast the computation itself is.

    '''
    linalg.init()
    fps = FPSCounter(params={'display_every_k_seconds': 5})

    while True:
        a = np.array([random.random()], np.float32)
        a_gpu = gpuarray.to_gpu(a)
        fps.update()
