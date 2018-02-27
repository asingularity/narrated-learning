
import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import skcuda
import skcuda.misc as misc
import pycuda.gpuarray as gpuarray
from fast_save_matrix import savetxt

def loadtxt():
    pass

DTYPE = np.float32


def get_table(load_from_file=False):
    print 'start table init...'

    if load_from_file:
        pass
    else:
        dim = 16 * 3
        entries = 40000  # 16K, 40K
        #table = np.zeros((entries, dim * 3)).astype(DTYPE)
        table = None
        output_input_distance = np.random.random((entries, entries)).astype(DTYPE)

        #filename = '/home/intec/NL-tmp/output_input_distance_' + str(entries) + '_' + str(dim) + '.txt'
        #print 'saving to file: ', filename
        #savetxt(filename, output_input_distance)

    print 'finished table init.'
    return output_input_distance, dim, entries


def run_cpu_test(output_input_dist, dim, entries, test_seconds):
    t_0 = time.time()
    fps_last_time = time.time()
    fps_frames = 0
    frame = 0

    while time.time() < t_0 + test_seconds:
        v1 = np.random.random((entries, 1)).astype(DTYPE)

        tmp = output_input_dist + v1
        uncert = np.amin(tmp, axis=0)

        if frame == 0:
            print '    data check:', uncert[[0, 1, 100, -1]], 'shape:', uncert.shape

        fps_frames += 1
        if time.time() - fps_last_time > 5.0:
            FPS = fps_frames * 1.0 / (time.time() - fps_last_time)
            print 'FPS: ', FPS
            fps_frames = 0
            fps_last_time = time.time()

        frame += 1


def run_cuda_test_multiple(output_input_dist, dim, entries, test_seconds):

    skcuda.misc.init()

    split_n = 20

    t_s1 = 0.0
    t_s2 = 0.0
    t_s3 = 0.0
    t_s4 = 0.0
    t_s5 = 0.0
    t_s6 = 0.0
    t_s7 = 0.0

    t_0 = time.time()
    fps_last_time = time.time()
    fps_frames = 0
    frame = 0

    while time.time() < t_0 + test_seconds:

        t0 = time.time()
        v1 = np.random.random((entries, 1)).astype(DTYPE)
        uncert = np.ones(entries) * np.inf
        t_s6 += time.time() - t0
        #tmp = output_input_dist + v1
        #uncert = np.amin(tmp, axis=0)

        entries_per_split = entries * 1.0 / split_n
        assert int(entries_per_split) == entries_per_split
        entries_per_split = int(entries_per_split)

        indices_start_end_list = []
        for k in range(split_n):
            indices_start_end_list.append((k * entries_per_split, (k+1) * entries_per_split))

        for indices_start_end in indices_start_end_list:
            i0 = indices_start_end[0]
            i1 = indices_start_end[1]

            t0 = time.time()
            o_i_dist_gpu = gpuarray.to_gpu(output_input_dist[i0:i1, :])
            t_s7 += time.time() - t0
            t0 = time.time()
            v1_gpu = gpuarray.to_gpu(v1[i0:i1, :])
            t_s1 += time.time() - t0
            t0 = time.time()
            sum_gpu = misc.add(o_i_dist_gpu, v1_gpu)
            t_s2 += time.time() - t0
            t0 = time.time()
            uncert_gpu = misc.min(sum_gpu, axis=0)  # , keepdims=False)
            t_s3 += time.time() - t0
            t0 = time.time()
            new_uncert = uncert_gpu.get()
            t_s4 += time.time() - t0
            t0 = time.time()
            uncert = np.minimum(uncert, new_uncert[:])
            t_s5 += time.time() - t0

        if frame == 0:
            print '    data check:', uncert[[0, 1, 100, -1]], 'shape:', uncert.shape

        fps_frames += 1
        if time.time() - fps_last_time > 5.0:
            FPS = fps_frames * 1.0 / (time.time() - fps_last_time)
            T = time.time() - t_0
            print 'FPS: ', FPS, 't_s1', t_s1 / T, 't_s2', t_s2 / T, 't_s3', t_s3 / T, \
                't_s4', t_s4 / T, 't_s5', t_s5 / T,  't_s6', t_s6 / T,   't_s7', t_s7 / T
            fps_frames = 0
            fps_last_time = time.time()

        frame += 1


def run_cuda_test(output_input_dist, dim, entries, test_seconds):

    skcuda.misc.init()
    o_i_dist_gpu = gpuarray.to_gpu(output_input_dist)

    t_to_gpu = 0.0
    t_add = 0.0
    t_min = 0.0
    t_get = 0.0

    t_0 = time.time()
    fps_last_time = time.time()
    fps_frames = 0
    frame = 0

    while time.time() < t_0 + test_seconds:
        #        v1 = np.random.random((entries, 1)).astype(DTYPE)
        #        v1_gpu = gpuarray.to_gpu(v1)
        #        sum_gpu = misc.add(o_i_dist_gpu, v1_gpu)

        v1 = np.random.random(entries).astype(DTYPE)

        t0 = time.time()
        v1_gpu = gpuarray.to_gpu(v1)
        t_to_gpu += time.time() - t0

        t0 = time.time()
        sum_gpu = misc.add_matvec(o_i_dist_gpu, v1_gpu, axis=0)
        t_add += time.time() - t0

        t0 = time.time()
        uncert_gpu = misc.min(sum_gpu, axis=0)  # , keepdims=False)
        t_min += time.time() - t0

        t0 = time.time()
        uncert = uncert_gpu.get()
        t_get += time.time() - t0

        if frame == 0:
            print '    data check:', uncert[[0, 1, 100, -1]], 'shape:', uncert.shape

        fps_frames += 1
        if time.time() - fps_last_time > 5.0:
            FPS = fps_frames * 1.0 / (time.time() - fps_last_time)
            T = time.time() - t_0
            print 'FPS: ', FPS, 'to_gpu', t_to_gpu / T, 'add', t_add / T, 'min', t_min / T, 'get', t_get / T
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

    np.random.seed(2)
    output_input_dist, dim, entries = get_table()

    test_seconds = 31

    print
    print '--- cuda multiple split ---'
    np.random.seed(2)
    run_cuda_test_multiple(output_input_dist, dim, entries, test_seconds=test_seconds)

    print
    print '--- cpu ---'
    np.random.seed(2)
    run_cpu_test(output_input_dist, dim, entries, test_seconds=test_seconds)

    print
    print '--- cuda single ---'
    np.random.seed(2)
    run_cuda_test(output_input_dist, dim, entries, test_seconds=test_seconds)
