
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import skcuda
import skcuda.misc as misc
import pycuda.gpuarray as gpuarray


def get_table():
    print 'start table init...'
    dim = 16 * 3
    entries = 8000  # 16K, 40K

    #table = np.zeros((entries, dim * 3)).astype(np.float32)

    table = None
    output_input_distance = np.random.random((entries, entries)).astype(np.float32)
    print 'finished table init.'
    return output_input_distance, dim, entries


def run_cpu_test(test_seconds):
    output_input_dist, dim, entries = get_table()

    t_0 = time.time()
    fps_last_time = time.time()
    fps_frames = 0
    frame = 0

    while time.time() < t_0 + test_seconds:
        v1 = np.random.random((entries, 1)).astype(np.float32)

        tmp = output_input_dist + v1
        uncert = np.amin(tmp, axis=0)

        if frame == 2:
            print '    data check:', uncert[0:4], 'shape:', uncert.shape

        fps_frames += 1
        if time.time() - fps_last_time > 5.0:
            FPS = fps_frames * 1.0 / (time.time() - fps_last_time)
            print 'FPS: ', FPS
            fps_frames = 0
            fps_last_time = time.time()

        frame += 1


def run_cuda_test_single(test_seconds):
    output_input_dist, dim, entries = get_table()

    t_0 = time.time()
    fps_last_time = time.time()
    fps_frames = 0
    frame = 0

    skcuda.misc.init()
    o_i_dist_gpu = gpuarray.to_gpu(output_input_dist)

    while time.time() < t_0 + test_seconds:
        v1 = np.random.random((entries, 1)).astype(np.float32)

        #tmp = output_input_dist + v1
        #uncert = np.amin(tmp, axis=0)

        v1_gpu = gpuarray.to_gpu(v1)
        sum_gpu = misc.add(o_i_dist_gpu, v1_gpu)
        uncert_gpu = misc.min(sum_gpu, axis=0)  # , keepdims=False)
        uncert = uncert_gpu.get()

        if frame == 2:
            print '    data check:', uncert[0:4], 'shape:', uncert.shape

        fps_frames += 1
        if time.time() - fps_last_time > 5.0:
            FPS = fps_frames * 1.0 / (time.time() - fps_last_time)
            print 'FPS: ', FPS
            fps_frames = 0
            fps_last_time = time.time()

        frame += 1


def run_cuda_test(test_seconds):
    output_input_dist, dim, entries = get_table()

    t_0 = time.time()
    fps_last_time = time.time()
    fps_frames = 0
    frame = 0

    skcuda.misc.init()
    o_i_dist_gpu = gpuarray.to_gpu(output_input_dist)

    while time.time() < t_0 + test_seconds:
        v1 = np.random.random((entries, 1)).astype(np.float32)

        #tmp = output_input_dist + v1
        #uncert = np.amin(tmp, axis=0)

        v1_gpu = gpuarray.to_gpu(v1)
        sum_gpu = misc.add(o_i_dist_gpu, v1_gpu)
        uncert_gpu = misc.min(sum_gpu, axis=0)  # , keepdims=False)
        uncert = uncert_gpu.get()

        if frame == 2:
            print '    data check:', uncert[0:4], 'shape:', uncert.shape

        fps_frames += 1
        if time.time() - fps_last_time > 5.0:
            FPS = fps_frames * 1.0 / (time.time() - fps_last_time)
            print 'FPS: ', FPS
            fps_frames = 0
            fps_last_time = time.time()

        frame += 1



if __name__ == '__main__':
    '''
        test single table that fits onto GPU
        test larger table that fits onto GPU in sections only

        entries, cuda fps, cpu fps, speedup:

        1K:
        2K: 450, 76,    5.9
        4K: 125, 17,    7.3
        8K: 28, 4,      7.0
        12K: 10, 2,     5.0
        16k: 2.8, 1.2   2.0
        24K: cuda can't allocate, cpu 0.54
    '''


    test_seconds = 11

    print
    print '--- cuda ---'
    np.random.seed(2)
    run_cuda_test(test_seconds=test_seconds)

    print
    print '--- cpu ---'
    np.random.seed(2)
    run_cpu_test(test_seconds=test_seconds)
