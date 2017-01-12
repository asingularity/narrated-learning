
import numpy as np
import time
from brute_force_knn import knn
from sklearn.neighbors import KDTree
from knn_cython import knn_query

np.random.seed(0)

def run_test_incremental():
    frames = 1000000
    dim = 20
    data = np.random.random((frames, dim))
    last_time = time.time()
    last_frame = 0

    for frame in range(frames):
        if frame > 2:
            knn_1 = knn(data[0:frame - 1, :])
            dist, ind = knn_1.query([data[frame, :]], 1)
        if time.time() - last_time > 5:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()


def run_test_full(tree, test_seconds):
    data_frames = 1000000 * 2
    dim = 60
    #data = 1.0 + 0.0 * np.random.random((data_frames, dim))
    data = np.random.random((data_frames, dim))

    if tree:
        print 'building KDTree...'
        build_start_time = time.time()
        knn_1 = KDTree(data, leaf_size=40)
        build_end_time = time.time()
        print 'done. time: ' + str(build_end_time - build_start_time)
    else:
        print 'building KNN...'
        knn_1 = knn(data)
        print 'done.'

    start_time = time.time()
    last_time = time.time()
    last_frame = 0

    test_frames = 10000

    for frame in range(test_frames):

        dist, ind = knn_1.query([np.random.random(dim)], 1)
        print dist, ind
        if time.time() - last_time > 5:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()
        if time.time() - start_time > test_seconds:
            return


def run_cython_knn_test(test_seconds):
    data_frames = 1000000 * 2
    dim = 60
    data = np.random.random((data_frames, dim))

    start_time = time.time()
    last_time = time.time()
    last_frame = 0

    test_frames = 10000

    for frame in range(test_frames):

        query_data = np.random.random(dim)

        dist, ind = knn_query(data, query_data, data_frames, dim)
        print dist, ind
        if time.time() - last_time > 5:
            FPS = (frame - last_frame) * 1.0 / (time.time() - last_time)
            print 'frame: ', frame, 'FPS: ', FPS
            last_frame = frame
            last_time = time.time()
        if time.time() - start_time > test_seconds:
            return

if __name__ == '__main__':
    test_seconds = 11
    #run_test_full(tree=True, test_seconds=test_seconds)
    run_test_full(tree=False, test_seconds=test_seconds)
    #run_cython_knn_test(test_seconds=test_seconds)

    # TODO test scaling of training time for KDtree vs. data size
    # TODO verify KDTree lookup: same neighbors as brute force for same random seed?
    # TODO ALSO try with real data here! in case it influences speed of build: YES IT DOES