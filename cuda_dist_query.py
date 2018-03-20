
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import skcuda.linalg as linalg
import skcuda.misc as misc


class CudaQuery(object):
    def __init__(self, input_data):
        self.original_input_data_shape = input_data.shape
        query_data = np.zeros(input_data.shape[1], np.float32)

        self.input_data = input_data #.flatten()  # row-major order. do not have to flatten.
        self.tmp_data = np.zeros(self.input_data.shape, np.float32)

        #self.func = mod.get_function("knn_query")

        #self.input_data_gpu = cuda.mem_alloc(self.input_data.size * self.input_data.dtype.itemsize)
        #self.query_data_gpu = cuda.mem_alloc(query_data.size * query_data.dtype.itemsize)
        #self.tmp_data_gpu = cuda.mem_alloc(self.tmp_data.size * self.tmp_data.dtype.itemsize)

        #cuda.memcpy_htod(self.input_data_gpu, self.input_data)

        self.X = np.zeros((1, input_data.shape[1])).astype(np.float32)
        self.input_data_transpose = np.ascontiguousarray(np.transpose(self.input_data))

        self.term_2 = np.sum(self.input_data ** 2, axis=1)

        #print 'contiguous?'
        #print self.X.flags['C_CONTIGUOUS']
        #print self.input_data_transpose.flags['C_CONTIGUOUS']

        linalg.init()
        self.i_d_t_gpu = gpuarray.to_gpu(self.input_data_transpose)

    def retransfer_matrix_for_test(self):
        #self.term_2 = np.sum(self.input_data ** 2, axis=1)
        del self.i_d_t_gpu
        self.i_d_t_gpu = gpuarray.to_gpu(self.input_data_transpose)

    def set_matrix_row(self, row_index, row_data):
        '''
            since we store transposed matrix: row becomes column
        '''

        col = row_index
        rows = self.input_data_transpose.shape[0]
        cols = self.input_data_transpose.shape[1]

        assert row_data.shape[0] == rows

        arr_gpu = gpuarray.to_gpu(row_data)
        misc.set_by_index(dest_gpu=self.i_d_t_gpu, ind=col + cols * np.arange(rows), src_gpu=arr_gpu, ind_which='dest')

        # also change self.term2
        # self.term_2.shape  # 10000 (original rows, transposed cols)
        # self.term_2 = np.sum(self.input_data ** 2, axis=1)
        self.term_2[row_index] = np.sum(row_data ** 2)

    def query(self, query_data):
        X = self.X
        X[0, :] = query_data[:]

        start_all = time.time()
        i_d_t_gpu = self.i_d_t_gpu

        #i_d_t_gpu = gpuarray.to_gpu(np.random.randn(4, 4).astype(np.float32))
        #X_gpu = gpuarray.to_gpu(np.random.randn(4, 4).astype(np.float32))

        X_gpu = gpuarray.to_gpu(X)
        start_1 = time.time()
        # DOESN'T WORK:
        # term_1 = gpuarray.dot(X_gpu, i_d_t_gpu).get()
        # WORKS:
        term_1 = linalg.dot(X_gpu, i_d_t_gpu).get()
        end_1 = time.time() - start_1

        start_2 = time.time()
        term_1 = -2 * term_1  # 15%
        end_2 = time.time() - start_2

        term_2 = self.term_2

        term_3 = np.sum(X ** 2, axis=1)[:, np.newaxis]

        dists = term_1 + term_2 + term_3  # 25%

        #ind = np.argmin(dists)  # <= 10%

        #dist = dists[0, ind]
        #end_all = time.time() - start_all

        # print 'ratio of time spent in dot product: ', end_1 / end_all
        # print 'ratio of time spent in 2: ', end_2 / end_all
        return dists


if __name__ == '__main__':
    pass