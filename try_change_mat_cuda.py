
import numpy as np
import time
from brute_force_knn import knn
from knn_cython import knn_query
from knn_parallel import knn_query as knn_parallel_query
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg
import skcuda.misc as misc
from utils.fps_counter import FPSCounter


def try_set_row():
    print
    print '*** setting row ***'
    print

    fps = FPSCounter(params={'display_every_k_seconds': 5})

    m = np.random.random((10000, 1000)).astype(np.float32)
    m_gpu = gpuarray.to_gpu(m)

    arr = np.random.random(1000).astype(np.float32)
    arr_gpu = gpuarray.to_gpu(arr)
    #print 'arr', arr[0:5], arr[-1]
    print 'm_gpu 2', m_gpu.get()[2, 0:5], m_gpu.get()[2, -1]
    print 'm_gpu 3', m_gpu.get()[3, 0:5], m_gpu.get()[3, -1]
    print 'm_gpu 4', m_gpu.get()[4, 0:5], m_gpu.get()[4, -1]

    print 'setting...'
    t0 = time.time()
    while time.time() - t0 < 11:
        row = 3
        arr = np.random.random(1000).astype(np.float32)
        arr_gpu = gpuarray.to_gpu(arr)
        misc.set_by_index(dest_gpu=m_gpu, ind=row * 1000 + np.arange(1000), src_gpu=arr_gpu, ind_which='dest')
        fps.update()

    print 'after...'

    print 'arr', arr[0:5], arr[-1]
    print 'm_gpu 2', m_gpu.get()[2, 0:5], m_gpu.get()[2, -1]
    print 'm_gpu 3', m_gpu.get()[3, 0:5], m_gpu.get()[3, -1]
    print 'm_gpu 4', m_gpu.get()[4, 0:5], m_gpu.get()[4, -1]


def try_set_column():
    print
    print '*** setting column ***'
    print

    fps = FPSCounter(params={'display_every_k_seconds': 5})

    m = np.random.random((1000, 10000)).astype(np.float32)
    m_gpu = gpuarray.to_gpu(m)

    arr = np.random.random(1000).astype(np.float32)
    arr_gpu = gpuarray.to_gpu(arr)
    #print 'arr', arr[0:5], arr[-1]
    print 'm_gpu 2', m_gpu.get()[0:5, 2], m_gpu.get()[-1, 2]
    print 'm_gpu 3', m_gpu.get()[0:5, 3], m_gpu.get()[-1, 3]
    print 'm_gpu 4', m_gpu.get()[0:5, 4], m_gpu.get()[-1, 4]

    print 'setting...'
    t0 = time.time()
    while time.time() - t0 < 11:
        col = 3
        arr = np.random.random(1000).astype(np.float32)
        arr_gpu = gpuarray.to_gpu(arr)
        misc.set_by_index(dest_gpu=m_gpu, ind=col + 10000 * np.arange(1000), src_gpu=arr_gpu, ind_which='dest')
        fps.update()

    print 'after...'

    print 'arr', arr[0:5], arr[-1]
    print 'm_gpu 2', m_gpu.get()[0:5, 2], m_gpu.get()[-1, 2]
    print 'm_gpu 3', m_gpu.get()[0:5, 3], m_gpu.get()[-1, 3]
    print 'm_gpu 4', m_gpu.get()[0:5, 4], m_gpu.get()[-1, 4]



if __name__ == '__main__':
    try_set_column()
    try_set_row()
