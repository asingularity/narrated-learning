
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


def test_simple_streaming():

    linalg.init()
    fps = FPSCounter(params={'display_every_k_seconds': 5})

    batch_size = 10000000  # how many 32-bit floats to stream and multiply

    # compute samples per second from FPS and batch_size
    print('samples per second: ' + str(batch_size) + ' x FPS')

    while True:
        a = np.array([random.random()] * batch_size, np.float32)
        a_gpu = gpuarray.to_gpu(a)
        b = np.array([random.random()]*batch_size, np.float32)
        b_gpu = gpuarray.to_gpu(b)
        term_1 = linalg.multiply(a_gpu, b_gpu, overwrite=False).get()
        #print(a, b, term_1)
        measured_fps = fps.update()
        if measured_fps is not None:
            print('32-bit kilo-samples per second: ' + str((measured_fps * batch_size) / 1000.0))

class CudaMultiplierTester(object):
    def __init__(self, init_mult_vector, batch_size, mult_dim, n_threads):
        '''

        init kernel and memory

        '''

        self.threads = n_threads

        print()
        print('*****')
        print(mult_dim, mult_dim * 1.0 / self.threads)
        print('*****')
        print()

        mod = SourceModule("""
            __global__ void stream_test(float *mult_vector, float *in_data_batch, float *out_sum)
            {
                int n_threads = """ + str(self.threads) + """;
                int mult_dim = """ + str(mult_dim) + """;
                int batch_size = """ + str(batch_size) + """;
                int idx = threadIdx.x;

                int n_mult = mult_dim / n_threads;
                int i0 = idx * n_mult;
                int sample_index;
                int mult_index;
                float total_sum;
                int tmp;

                float out_sum_idx = 0.0;
                for(sample_index = 0; sample_index < batch_size; sample_index++)
                {
                    for(mult_index = i0; mult_index < i0 + n_mult; mult_index++)
                    {
                        // SHARED MEMORY DOESN"T WORK THE WAY WE WANT
                        // SUPER SLOW out_sum[idx] += ...
                        out_sum_idx += mult_vector[mult_index] * in_data_batch[sample_index];
                    }

                    // sync threads before sum is used
                    __syncthreads();
                    /*
                    if (idx == 0)
                    {
                        total_sum = 0.0;
                        for(tmp = 0; tmp < n_threads; tmp++)
                        {
                            total_sum += out_sum[tmp];
                            if(sample_index < batch_size - 1)
                                out_sum[tmp] = 0.0;
                        }
                    }

                    __syncthreads();

                   for(mult_index = i0; mult_index < i0 + n_mult; mult_index++)
                   {
                        mult_vector[mult_index] += total_sum * 0.0000001;
                   }

                    __syncthreads();
                    */
                }

                // TODO WHY SO SLOW
                // out_sum[0] = out_sum_idx;
                // mult_vector[idx] = out_sum_idx;
            }

            """)

        self.mod = mod
        self.func = self.mod.get_function("stream_test")

        # TODO proper
        tmp = np.zeros(batch_size, np.float32)
        self.in_data_batch_gpu = cuda.mem_alloc(tmp.size * tmp.dtype.itemsize)

        self.mult_vector_gpu = cuda.mem_alloc(init_mult_vector.size * init_mult_vector.dtype.itemsize)

        tmp = np.zeros(self.threads, np.float32)
        self.out_sum_gpu = cuda.mem_alloc(tmp.size * tmp.dtype.itemsize)

    def run_one_batch(self, in_data_batch):
        '''

        :return:
        '''

        cuda.memcpy_htod(self.in_data_batch_gpu, in_data_batch)

        self.func(self.mult_vector_gpu, self.in_data_batch_gpu, self.out_sum_gpu, block=(self.threads, 1, 1))
        current_sum = np.array([0.0] * self.threads, np.float32)

        cuda.memcpy_dtoh(current_sum, self.out_sum_gpu)

        return current_sum


def test_full_reqs_GPU():
    '''

    REQUIREMENTS more specifically:
    (A) 96K samples per second input
    (B) per sample, need to compute 13k multiplies - *** THIS IS PARALLELIZABLE ON GPU *** - and then a sum over that
    (C) processed sample depends on previous sample sequentially i.e. iterative algorithm per-sample
            specifically, max delay 3 need from past sample processing. more delay, algorithm decays.
            i.e. feedback from output (sum) back to input, inside the multiple

    What is the test?
        - pass in a batch of data, and initial/current state of algorithm
        - run iterative algorithm on gpu over whole batch
        - pass output back, and current state of algorithm after batch

    How to implement?
        - need a custom kernel to i.e. multiply vectors in a loop

    :return:
    '''

    fps = FPSCounter(params={'display_every_k_seconds': 5})

    mult_dim = 15360 * 1
    batch_size = 1000 * 1
    n_threads = int(1024 / 1)

    mult_vector = np.random.random(mult_dim).astype(np.float32)

    cuda_tester = CudaMultiplierTester(init_mult_vector=mult_vector,
                                       batch_size=batch_size,
                                       mult_dim=mult_dim,
                                       n_threads=n_threads)

    in_data_batch = np.array([random.random()] * batch_size, np.float32)

    while True:
        # CALL CUSTOM KERNEL

        out = cuda_tester.run_one_batch(in_data_batch=in_data_batch)

        measured_fps = fps.update()
        if measured_fps is not None:
            print('example out:', out[0:5])
            print('32-bit kilo-samples per second: ' + str((measured_fps * batch_size) / 1000.0))




def test_full_reqs_GPU_simple():
    '''

    # TODO reproduce if possible with skcuda linalg what we have below
    # TODO how to do iterative part and sum? for loop here instead of batch?
    # TODO can a gpuarray already on gpu be indexed for a partial multiply-sum?
    # TODO is there already an algorithm like this on skcuda linalg?
    # TODO loop in python, per sample, no batch stuff

    :return:
    '''

    linalg.init()

    mult_dim = 15360 * 100  # same rate even if this is *100

    fps = FPSCounter(params={'display_every_k_seconds': 5})

    mult_vector = np.random.random(mult_dim).astype(np.float32)
    mult_vector_gpu = gpuarray.to_gpu(mult_vector)

    tmp_sum = 0.0
    sample = np.float32(random.random())
    one_over_sample = np.float32(1.0/sample)

    while True:
        # generate new sample

        # use linalg.scale to multiply new sample by vector
        linalg.scale(alpha=sample, x_gpu=mult_vector_gpu)

        # get sum
        # tmp_sum = misc.sum(mult_vector_gpu)

        # undo scaling
        linalg.scale(alpha=one_over_sample, x_gpu=mult_vector_gpu)

        # modify mult_vector with sum
        #if random.random() < 0.5:
        #    scale = 0.000001 * tmp_sum
        #else:
        #    scale = 0.000001 * tmp_sum

        # mult_vector_gpu = mult_vector_gpu + scale

        measured_fps = fps.update()
        if measured_fps is not None:
            print('32-bit kilo-samples per second: ' + str((measured_fps) / 1000.0))
            print('sum', tmp_sum)


def test_full_reqs_CPU():
    mult_dim = 15360 * 100

    fps = FPSCounter(params={'display_every_k_seconds': 5})
    mult_vector = np.random.random(mult_dim).astype(np.float32)
    sample = np.float32(random.random())
    one_over_sample = np.float32(1.0/sample)

    while True:
        mult_vector = sample * mult_vector
        mult_vector = one_over_sample * mult_vector

        #c = np.sum(b)
        #mult_vector += c * 0.0000001

        fps.update()


if __name__ == '__main__':
    '''
    Goal of this experiment:
    How many samples per second can we stream?

    I.e.
    1 sample - 10 bit

    Equivalent: how much time does it take to send one to GPU, and receive one back.   !!!!!! 10/17: obviously, it also helps to send 1000000 to the gpu and get them back in the same time !!!!!!
    '''

    #test_simple_streaming()

    #test_full_reqs_CPU()
    test_full_reqs_GPU_simple()

    #test_full_reqs_GPU()

