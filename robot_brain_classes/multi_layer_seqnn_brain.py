'''

this file:
    multi_layer_seqnn_brain.py
    started 3/6/21

goes with:
    run_demo_perception.py
    offline_analyses/...

this is the multi layer, optimized, sped up newer version of:
    [ ]     run_demo_predict.py
    [X]     robot_brain_classes/two_stage_seqnn_brain.py
    [ ]     offline_analyses/try_sparsify_3.py

'''

import time
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt
from brain_components_classes.states_history import StatesLimitedHistory
from cython_eff import compute_eff

from utils.one_time_messages import OneTimeMessages

from cuda_dist_query import CudaTable

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log






class MultiLayerSeqNNBrain(object):
    def __init__(self, params):
        '''

        :param params:
        '''

        self.last_input_im = None
        self.bins_per_pixel = 6
        self.input_im_dim = params['input_im_dim']

        layer_parmas = {}
        layer_parmas['input_state_dim'] = self.input_im_dim * self.input_im_dim * self.bins_per_pixel
        layer_parmas['layer_name'] = '0'
        layer_parmas['cuda_table_rows'] = 3200

        self.layers = [SingleLayer(params=layer_parmas)]

        self.display_im_dim = 3000

    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        # pixels to bins
        input_pixels_1 = input_im
        input_pixels_flat = input_pixels_1.flatten()
        input_arr_1 = input_pixels_flat[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

        self.last_input_im = input_pixels_1.copy()

        layer_output = self.layers[0].step(layer_input=input_exp_1)

        # multi-layer later:
        # for layer_n in range(self.num_layers):
        #    layer_output = self.layers[layer_n].step(layer_input=layer_input)
        #    # store output for current frame into a states history, to use as input for next layer

    def _get_cuda_table_im(self):

        # THIS ONLY MAKES SENSE FOR LAYER 0; this converts directly to pixels!
        table_weights = self.layers[0].cuda_table.table_i.copy()

        tmp_im_all = None
        tmp_im = None
        arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=table_weights,
                                                               num_bins_per_pixel=self.bins_per_pixel)

        arbitrary_constant = 60  # number per column or something like that

        for r_tmp in range(weights.shape[0]):  # per rf

            im0 = arr[r_tmp, :].reshape((self.input_im_dim, self.input_im_dim))  # rf_dim
            im1 = weights[r_tmp, :].reshape((self.input_im_dim, self.input_im_dim))
            tmp2 = np.hstack((im0, im1))

            if tmp_im is None:
                tmp_im = tmp2.copy()
            else:
                tmp3 = 0.2 * np.ones((2, tmp_im.shape[1]))

                tmp_im = np.vstack((tmp_im, tmp3, tmp2))

            if r_tmp > 0 and (r_tmp + 1) % arbitrary_constant == 0:
                if tmp_im_all is None:
                    tmp_im_all = tmp_im.copy()
                else:
                    spacer = 0.5 * np.ones((tmp_im_all.shape[0], 3))
                    tmp_im_all = np.hstack((tmp_im_all, spacer, tmp_im))

                tmp_im = None

        tmp_all_im = tmp_im_all

        max_dim = max(tmp_all_im.shape[0], tmp_all_im.shape[1])
        imscale = self.display_im_dim / max_dim  # 0.2: full table, 2.0
        tmp_im = cv2.resize(tmp_all_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        cuda_table_im = tmp_im

        return cuda_table_im

    def get_table_ims(self):
        ims_list = []
        ims_names_list = []

        ims_list.append(self._get_cuda_table_im())
        ims_names_list.append('layer_0_cuda_table')

        return ims_list, ims_names_list

    def _bin_pixels_expand_columns(self, arr, num_bins_per_pixel):
        '''

        keep number of rows, expand in number of columns
        construct binary image: (0.0 or 1.0) per bin

        https://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
            use: digitize, or histogram

        https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.digitize.html
            digitize: returns index of bin to which each pixel belongs
            expanded array will have a set of bins per pixel, in the expanded columns

        :param arr:
        :return:
        '''
        # print('***')
        # print('arr', arr.shape)

        bins = np.linspace(0, 1 + 1e-9, num_bins_per_pixel + 1)  # TODO to fix binning add 1e-9 to 1 in second argument!
        # print('bins', bins.shape, bins)
        bin_indices = np.digitize(arr, bins) - 1  # same shape as arr; which bin, per pixel
        # print('bin_indices', bin_indices.shape, bin_indices)
        self.use_negative_input = False  # instead of zero, set non-inputs to -1 for all layers
        if self.use_negative_input:
            arr_exp = -np.ones((arr.shape[0], arr.shape[1] * num_bins_per_pixel))
        else:
            arr_exp = np.zeros((arr.shape[0], arr.shape[1] * num_bins_per_pixel))

        # print('arr_exp', arr_exp.shape)

        # term1 = np.tile(self.num_bins_per_pixel * np.arange(arr.shape[1]), 2)  # can't remember why this is np.tile(..., 2)
        term1 = num_bins_per_pixel * np.arange(arr.shape[1])  # only works if arr rows is 1?
        term2 = bin_indices.flatten()
        # print('term1', term1.shape, term1)
        # print('term2', term2.shape, term2)
        c = term1 + term2
        r = np.repeat(np.arange(arr.shape[0]), arr.shape[1])

        arr_exp[r, c] = 1.0

        return arr_exp


    def _collapse_binned_columns_to_pixels(self, arr_exp, num_bins_per_pixel):
        '''

        non-trivial: arr_exp may no longer be binary; need to figure out how to collapse multiple values of different weight to one pixel:
            weighted average? take max value?

        :param arr_exp:
        :return:
        '''

        num_pixels = int((arr_exp.shape[1] / num_bins_per_pixel))
        num_rows = arr_exp.shape[0]

        # reshape so we can take max along one dim
        #   ie each row in this matrix is a pixel, temporarily
        # then reshape back

        tmp = arr_exp.reshape((num_rows * num_pixels, num_bins_per_pixel))

        bins = np.linspace(0, 1 + 1e-9, num_bins_per_pixel + 1)

        # max val for display:
        max_index = np.argmax(tmp, axis=1)
        max_vals = bins[max_index]

        weights = tmp[np.arange(tmp.shape[0]), max_index]

        # print(max_vals.shape, weights.shape)

        # weighted mean val for display:
        # tmp1 = np.multiply(tmp, bins[np.newaxis, 0:num_bins_per_pixel])
        # tmp2 = np.sum(tmp1, axis=1)  # 65536
        # tmp3 = np.divide(tmp2, np.sum(tmp, axis=1))
        # max_vals = tmp3

        weights = weights.reshape(num_rows, num_pixels)
        arr = max_vals.reshape(num_rows, num_pixels)

        return arr, weights




class SingleLayer(object):
    def __init__(self, params):
        '''

        :param params:
        '''

        self.input_state_dim = params['input_state_dim']
        self.layer_name = params['layer_name']

        self.cuda_table_rows = params['cuda_table_rows']
        self.rf_training_times = [20000, 40000, 80000]  # which frames to train RFs off of cuda table

        self.cuda_table = CudaTable(num_entries=self.cuda_table_rows,
                                    input_dim=self.input_state_dim)
        self.init_I_row_num = None

        self.t = 0

        self.num_row_changes_for_disp = 0
        self.frames_since_row_change_disp = 0

    def step(self, layer_input):

        # update seq-nn table

        input_state = layer_input.flatten().astype(np.float32)
        dists = self.cuda_table.query(query_input=input_state)
        row_replaced = self._learn_table(cuda_table_I=self.cuda_table, dists=dists, input_state=input_state)  # hl only needed for init_I_row_nums

        if row_replaced:
            self.num_row_changes_for_disp += 1
        self.frames_since_row_change_disp += 1

        # periodically, re-train RFs using max-bin-append; library from offline_analyses

        layer_output = None

        return layer_output

    def _learn_table(self, cuda_table_I, dists, input_state):
        '''

        :param dists:
        :param input_state:
        :return:
        '''

        # to do later on replacement or learning: should zero out W from that row for all predictions
        #   why don't we do this now?
        #   because, for now, we learn table, then we leave it alone when learning predictions later

        row_replaced = False

        # assert input_state.shape[0] == self.input_dim

        sorted_dist_indices = np.argsort(dists)
        new_min_ind = sorted_dist_indices[0]
        new_min_dist = dists[new_min_ind]

        if self.init_I_row_num is None:
            self.init_I_row_num = 0

        if self.init_I_row_num < cuda_table_I.get_num_rows():
            # necessary so dist matrix helper is not so slow at start
            dists[self.init_I_row_num] = np.inf
            try:
                cuda_table_I.set_matrix_row(row_index=self.init_I_row_num,
                                            row_input=input_state,
                                            row_to_table_dists=dists,
                                            fast_init=True)
            except AssertionError:
                print('\nError! Invalid GPU data type. input_state.dtype: ' + str(input_state.dtype) + '\n')
                raise

            row_replaced = True
            self.init_I_row_num += 1
        else:
            if not cuda_table_I.post_init_done:
                cuda_table_I.post_init()

            table_min_dist, table_min_dist_r, table_min_dist_c = cuda_table_I.get_min_dist()

            if new_min_dist > table_min_dist:
                # minimum distance of new row to current rows is greater than current minimum row-row distance
                # so: replace one row of current minimum, with new row

                # get one of the row indices of current minimum dist pair
                r_r_ind = table_min_dist_r  # could be table_min_dist_c

                dists[r_r_ind] = np.inf

                # replace the current min dist row, with the new row
                cuda_table_I.set_matrix_row(row_index=r_r_ind,
                                            row_input=input_state,
                                            row_to_table_dists=dists)

                row_replaced = True

        return row_replaced







class MultiLayerSeqNNBrainWithPredict(object):
    def __init__(self, params):
        '''

        FOR LATER

        :param params:
        '''

    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        for layer_n in range(self.num_layers):

            # define layer_input

            self._process_layer(layer_n=layer_n, layer_input=layer_input)

    def _process_layer(self, layer_input):
        '''

        :param layer_input:
        :return:
        '''

        # layer is composed of two stages:
        #   "sparsify"
        #   "predict"

        self._do_stage_sparsify(layer_input)
        #self._do_stage_predict(layer_input)

    def _do_stage_sparsify(self, layer_input):
        '''

        (1) compute sparsify stage events -> training for predictive stage
        (2) learn sparsify stage given layer input

        :param layer_input:
        :return:
        '''

    def _do_stage_predict(self, layer_input):
        '''

        (1) compute predictive stage events -> input to next layer
        (2) learn predictive stage given sparsify stage events

        :param layer_input:
        :return:
        '''

    def get_table_ims(self):
        '''

        :return:
        '''

        return ims_list, ims_names_list


































