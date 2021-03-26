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
from offline_analyses.try_sparsify_3 import get_sparse_features, make_im

from utils.one_time_messages import OneTimeMessages

from cuda_dist_query import CudaTable

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log

np.set_printoptions(suppress=True, precision=2)


class AnotherBrain(object):
    def __init__(self, params):
        pass


class TemporalWTALearningBrain(object):
    def __init__(self, params):
        '''

        This is testing the temporal wta idea

        (1) get reference frames via seq-nn
        (2) for each reference frame, find best partial pattern cluster:
                for each input frame, if this input has most in common with reference frame over last 100 frames, store in common as the pattern

        :param params:
        '''

        self.rf_im = None

        self.num_rfs = 80

        self.bins_per_pixel = 12
        self.input_im_dim = params['input_im_dim']

        self.input_state_dim = self.input_im_dim * self.input_im_dim * self.bins_per_pixel

        self.rfs = np.random.random((self.num_rfs, self.input_state_dim)).astype(np.float32)

        self.t = 0
        self.wta_steps = 100  # 1000
        self.lr = 0.001 * 10  # 0.01

        self.last_train_time = -1000

        self.input_states_history = StatesLimitedHistory(params={'max_delay': self.wta_steps,
                                                                 'states_dim_list': [self.input_state_dim],
                                                                 'store_extra_data': False})

        self.weight_sum_history = StatesLimitedHistory(params={'max_delay': self.wta_steps,
                                                               'states_dim_list': [self.num_rfs],
                                                               'store_extra_data': False})


    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        # pixels to bins
        input_pixels_1 = input_im
        input_pixels_flat = input_pixels_1.flatten()
        input_arr_1 = input_pixels_flat[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel).flatten()

        input_exp_flat = input_exp_1.flatten()

        tmp  = np.multiply(input_exp_1, self.rfs)
        weight_sum = np.sum(tmp, axis=1) * 1.0 / (self.input_im_dim * self.input_im_dim)

        input_exp_flat -= np.sum(np.multiply(weight_sum[:, np.newaxis], tmp), axis=0)
        input_exp_flat[input_exp_1 < 0] = 0

        self.weight_sum_history.store_new_states(newest_states_list=[weight_sum])
        self.input_states_history.store_new_states(newest_states_list=[input_exp_flat])

        # learn first rf
        if self.t > self.last_train_time + self.wta_steps / 10:  # a hack for speed
            input_states_sequence = self.input_states_history.get_state_sequence(state_index=0,
                                                                                 delay_long=self.wta_steps - 1,
                                                                                 delay_short=0,
                                                                                 oldest_first=False)

            weight_sum_sequence = self.weight_sum_history.get_state_sequence(state_index=0,
                                                                             delay_long=self.wta_steps - 1,
                                                                             delay_short=0,
                                                                             oldest_first=False)

            # print(input_states_sequence.shape, weight_sum_sequence.shape)  # (100, 3072) (100, 40)

            for rf_index in range(self.num_rfs):
                first_rf_weight_sum_sequence = weight_sum_sequence[:, rf_index]
                best_input_index = np.argmax(first_rf_weight_sum_sequence)
                best_input = input_states_sequence[best_input_index, :]

                # disqualiy for others
                weight_sum_sequence[best_input_index, :] = 0

                self.rfs[rf_index, :] = (1.0 - self.lr) * self.rfs[rf_index, :] + self.lr * best_input

            self.last_train_time = self.t

        self.t += 1

    def get_table_ims(self):
        ims_list = []
        ims_names_list = []

        if self.rfs is not None:
            self.rf_im = make_im(input_arrays=self.rfs,
                                 num_bins_per_pixel=self.bins_per_pixel,
                                 input_im_dim=self.input_im_dim,
                                 im_final_dim=2000)

        if self.rf_im is not None:
            ims_list.append(self.rf_im)
            ims_names_list.append('rfs')

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





class GreedyPairwiseBins(object):
    def __init__(self, params):
        '''

        A simplified version of max-bin-append (try_sparsify_3.py) where we don't try to optimize size / remainder; just find max bins with max bins, not whole RF

        :param params:
        '''

        # TODO testing this was 24!
        self.bins_per_pixel = 6
        self.create_rf_times = [20000, 40000, 80000]
        self.input_im_dim = params['input_im_dim']

        self.input_state_dim = self.input_im_dim * self.input_im_dim * self.bins_per_pixel

        # for each input bin, track:
        #   number of times of occurrence
        #   number of times of co-occurrence with other bins
        self.occur = np.zeros(self.input_state_dim, np.int)

        self.co_occur = np.zeros((self.input_state_dim, self.input_state_dim), np.int)

        self.rf_im = None

        self.t = 0

    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        # pixels to bins
        input_pixels_1 = input_im
        input_pixels_flat = input_pixels_1.flatten()
        input_arr_1 = input_pixels_flat[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel).flatten()

        event_bins = np.nonzero(input_exp_1)[0]
        self.occur[event_bins] = self.occur[event_bins] + 1

        # for now, we will only look for co-occurrence within same frame; this should be expanded to look over some time window
        r, c = np.meshgrid(event_bins, event_bins)
        self.co_occur[r, c] = self.co_occur[r, c] + 1
        # zero out diagonal / self co-occurrence?
        # self.co_occur[event_bins, event_bins] = 0

        if self.t in self.create_rf_times:
            self._create_rfs()

        self.t += 1

    def _create_rfs(self):

        # these are not rfs per se, just the starting point for the event sieve.
        # later, we will divide these into pre / post (what is predictor, what is anticipated on average)
        # later we can do some remainder stuff etc. so there is pressure for rarer features

        num_rfs = 80
        rf_size = 60  # set size of rf in terms of weight==1 bins

        self.rfs = np.zeros((num_rfs, self.input_state_dim))

        occur_tmp = self.occur.copy()

        norm_co_occur = self.co_occur.copy()
        # the above should be more like correlation: normalized against occurrence false positives/negatives!
        #   do that here!!!
        occur_sum_mat = occur_tmp[:, None] + occur_tmp[None,:]
        norm_co_occur = np.divide(norm_co_occur, occur_sum_mat + 1e-12)

        # TESTING this should be below instead!!! where its commented, and comment here!
        # co_occur_tmp = norm_co_occur.copy()

        for rf in range(num_rfs):
            co_occur_tmp = norm_co_occur.copy()

            new_rf = np.zeros(self.input_state_dim)
            # choose a random nonzero bin
            nnz_bins = np.nonzero(occur_tmp)[0]
            nnz_bin = np.random.choice(nnz_bins)
            co_occur_tmp[:, nnz_bin] = 0

            new_rf[nnz_bin] = 1
            # rf_bins = [nnz_bin]

            for tmp in range(rf_size):
                # greedy: add max co-occuring bin that we haven't added yet, with any in the rf so far
                rf_bins = np.nonzero(new_rf)[0]

                co_occur_with_rf = co_occur_tmp[rf_bins, :]

                ind = np.unravel_index(np.argmax(co_occur_with_rf, axis=None), co_occur_with_rf.shape)
                # ind is two-dim: (r, c)
                bin_to_add = ind[1]
                new_rf[bin_to_add] = 1

                co_occur_tmp[:, bin_to_add] = 0

            self.rfs[rf, :] = new_rf[:]

            # TODO this is an extreme, hack rule!
            # actually, just affects initial bin selection
            # occur_tmp[np.nonzero(new_rf)] = 0

        print(self.rfs.shape)
        self.rf_im = make_im(input_arrays=self.rfs, num_bins_per_pixel=self.bins_per_pixel, input_im_dim=self.input_im_dim, im_final_dim=2600)

    def get_table_ims(self):
        ims_list = []
        ims_names_list = []

        if self.rf_im is not None:
            ims_list.append(self.rf_im)
            ims_names_list.append('rfs')

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




class MultiLayerSeqNNBrain(object):
    def __init__(self, params):
        '''

        :param params:
        '''

        self.last_input_im = None
        self.bins_per_pixel = 1
        self.input_im_dim = params['input_im_dim']

        self.do_raster_plots_every_k_im = 10
        self.ims_since_raster = 0

        self.display_im_dim = 3000

        self.arbitrary_viz_constant = 20  # 60 for 3200 cuda table rows; 10 for 400 cuda table rows

        num_rfs_layer_0 = 400
        num_rfs_layer_1 = 50

        layer_0_paras = {'input_state_dim': self.input_im_dim * self.input_im_dim * self.bins_per_pixel,
                         'max_input_concat_timesteps': 2,  # UNUSED  # TODO if equals default, error: bug?
                         'default_concat_timesteps': 2,
                         'target_rf_sum': 60,  # UNUSED
                         'enable_target_rf': False,
                         'layer_name': '0',
                         'num_rfs': num_rfs_layer_0,
                         'rf_training_times': [np.inf],  # [10000],  # [20000, 40000, 80000],
                         'input_im_dim': params['input_im_dim'],
                         'num_bins_per_pixel': self.bins_per_pixel,
                         'display_im_dim': self.display_im_dim,
                         'learning_start_time': 0,
                         'learning_off_time': params['learning_off_time']}

        layer_1_paras = {'input_state_dim': num_rfs_layer_0,
                         'max_input_concat_timesteps': 4,  # UNUSED
                         'default_concat_timesteps': 4,
                         'target_rf_sum': 5,  # UNUSED
                         'enable_target_rf': False,
                         'layer_name': '1',
                         'num_rfs': num_rfs_layer_1,
                         'rf_training_times': [np.inf],  # [10000],  # [20000, 40000, 80000],
                         'input_im_dim': None,
                         'num_bins_per_pixel': None,
                         'display_im_dim': self.display_im_dim,
                         'learning_start_time': 20000,
                         'learning_off_time': params['learning_off_time']}

        self.layers = [SingleLayer(params=layer_0_paras), SingleLayer(params=layer_1_paras)]

    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        # pixels to bins
        input_pixels_1 = input_im
        input_pixels_flat = input_pixels_1.flatten()
        input_arr_1 = input_pixels_flat[np.newaxis, :]

        input_exp_1 = input_arr_1
        #input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

        self.last_input_im = input_pixels_1.copy()

        layer_output_0 = self.layers[0].step(layer_input=input_exp_1, change_to_diff_image=True)

        # TODO enable next layer!
        layer_output_1 = self.layers[1].step(layer_input=layer_output_0, change_to_diff_image=False)

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

        arbitrary_constant = self.arbitrary_viz_constant  # number per column or something like that

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

    def _get_rfs_im_composite(self, l0_rfs_ims_info):
        '''

        :param l0_rfs_ims_info: packaged in right way, from get_rfs_im() call of layer 0
        :return:
        '''

        # for second layer need something custom showing components of RF
        #   how to display? show (top?) component RFs and their relative weights

        top_k_components = 4

        l1_weights = self.layers[1].get_weights()

        tmp_all_im = None
        for l1_rf_ind in range(l1_weights.shape[0]):
            rf_weights = l1_weights[l1_rf_ind, :]
            l0_rf_index = np.argsort(rf_weights)[::-1]  # [::-1]: largest weights' indices first

            tmp_im = None

            for k in range(top_k_components):
                if tmp_im is None:
                    tmp_im = l0_rfs_ims_info[l0_rf_index[k]].copy()
                else:
                    tmp3 = 0.8 * np.ones((2, tmp_im.shape[1]))
                    tmp_im = np.vstack((tmp_im, tmp3, l0_rfs_ims_info[l0_rf_index[k]].copy()))

            if tmp_all_im is None:
                tmp_all_im = tmp_im.copy()
            else:
                tmp4 = 0.8 * np.ones((tmp_im.shape[0], 2))
                tmp_all_im = np.hstack((tmp_all_im, tmp4, tmp_im.copy()))

        return tmp_all_im

    def get_table_ims(self):
        ims_list = []
        ims_names_list = []

        l0_rfs_im, l0_rfs_ims_info = self.layers[0].get_rfs_im()
        if l0_rfs_im is not None:
            ims_list.append(l0_rfs_im)
            ims_names_list.append('layer_0_rfs')

        l1_rfs_im = self._get_rfs_im_composite(l0_rfs_ims_info)
        scale_l1_im = 2.0
        l1_rfs_im = cv2.resize(l1_rfs_im, dsize=(0, 0), fx=scale_l1_im, fy=scale_l1_im, interpolation=cv2.INTER_NEAREST)

        if l1_rfs_im is not None:
            ims_list.append(l1_rfs_im)
            ims_names_list.append('layer_1_rfs')

        if self.do_raster_plots_every_k_im is not None:
            self.ims_since_raster += 1
            if self.ims_since_raster > self.do_raster_plots_every_k_im:
                self.layers[0].do_plots()
                self.layers[1].do_plots()
                self.ims_since_raster = 0

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

        self.learning_off_time = params['learning_off_time']
        self.learning_start_time = params['learning_start_time']

        self.input_im_dim = params['input_im_dim']  # used for sparse features
        self.num_bins_per_pixel = params['num_bins_per_pixel']  # used for sparse features

        self.input_state_dim = params['input_state_dim']
        self.max_input_concat_timesteps = params['max_input_concat_timesteps']
        self.target_rf_sum = params['target_rf_sum']

        self.enable_target_rf = params['enable_target_rf']

        self.default_concat_timesteps = params['default_concat_timesteps']

        self.input_history = StatesLimitedHistory(params={'max_delay': self.max_input_concat_timesteps,
                                                          'states_dim_list': [self.input_state_dim],
                                                          'store_extra_data': False})

        self.layer_name = params['layer_name']

        self.display_im_dim = params['display_im_dim']

        self.t = 0

        self.rfs_im = None

        self.last_layer_input = None
        self.last_input_state = None
        self.last_last_input_state = None
        self.last_last_last_input_state = None

        # WTA experiment
        # 400
        num_rfs = params['num_rfs']  # 800
        self.weights = np.zeros((num_rfs, self.input_state_dim))
        self.num_rfs = num_rfs

        self.pixel_bin_counts = np.zeros(self.input_state_dim)
        self.pixel_bin_count_steps = 1000  # 10000

        self.predicted_pixel_bin = None

        # release for learning over time
        self.release_prop = 1.0  #0.1
        self.release_steps = 5000  # 5000

        # computed:
        self.release_num = int(self.release_prop * self.num_rfs)
        self.total_released = 0

        # target rf and current concat time steps
        self.per_rf_concat_timesteps = self.default_concat_timesteps * np.ones(self.num_rfs)

        # raster stuff
        max_time = 8000000
        self.input_raster_history = np.zeros((self.input_state_dim, max_time), np.uint8)
        self.rfs_0_raster_history = np.zeros((num_rfs, max_time), np.uint8)

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

        self.last_rfs_sums = None
        self.last_state_to_learn = None
        self.last_per_rf_time_index = None

        self.scaling_factor = np.ones(self.num_rfs)
        self.mean_rates = np.zeros(self.num_rfs)
        self.last_event_times = np.zeros(self.num_rfs)

        # adjust scaling factor based on firing rates
        self.target_rate = 1.0 / self.num_rfs  # since this is a single-WTA, we want even on average
        self.rate_calc_timescale = 800
        self.scale_factor_delta = 0.001

        self.lr = 0.01  #0.01 * 0.25  # * 0.05

        self.last_adjust_time = 0

    def get_weights(self):
        return self.weights.copy()

    def step(self, layer_input, change_to_diff_image):

        input_state = layer_input.flatten().astype(np.float32)

        if change_to_diff_image:
            if self.last_layer_input is not None:
                input_state = input_state - self.last_layer_input.flatten().astype(np.float32)
                input_state[input_state < 0] = 0

        nnz_input_state = np.nonzero(input_state)[0]
        self.input_raster_history[nnz_input_state, self.t] = 1

        self.input_history.store_new_states(newest_states_list=[input_state])

        input_states_seq = self.input_history.get_state_sequence(state_index=0,
                                                                 delay_short=0,
                                                                 delay_long=self.max_input_concat_timesteps - 1,
                                                                 oldest_first=False)
        #print(input_states_seq.shape)  # (10, 864)
        sum_input_states = np.cumsum(input_states_seq, axis=0)  # shape: (10, 864)

        # TODO this shouldn't be here
        # sum_input_states[sum_input_states > 1] = 1

        learning_on = self.learning_start_time <= self.t < self.learning_off_time
        layer_output = self._step_wta(sum_input_states, learning_on)

        self.t += 1
        self.last_layer_input = layer_input.copy()

        return layer_output

    def _step_wta(self, input_state, learning_on):
        '''

        :param input_state:
        :return:
        '''

        if (self.t == 0 or self.t % self.release_steps == 0) and self.total_released < self.num_rfs:
            self.total_released += self.release_num
            if self.total_released > self.num_rfs:
                self.total_released = self.num_rfs

        #state_to_learn = self.last_input_state.copy()
        state_to_learn = input_state.copy()

        # print(self.weights.shape, state_to_learn.shape)
        #             (400, 864)      (10, 864)
        # print(state_to_learn)

        # adjust per-rf-concat-timesteps based on rf size
        tmp_3_cpu = np.sum(self.weights[0:self.total_released, :], axis=1)

        self.last_rfs_sums = tmp_3_cpu.copy()

        self.last_state_to_learn = state_to_learn.copy()

        #nnz_below = np.nonzero(tmp_3_cpu < self.target_rf_sum)[0]
        #nnz_above = np.nonzero(tmp_3_cpu > self.target_rf_sum)[0]
        #self.weights[nnz_below, :] *= 1.00001
        #self.weights[nnz_above, :] *= (1.0 - 0.00001)

        #self.per_rf_concat_timesteps[nnz_below] += 0.0005 #*= (1.0 + 0.001)
        #self.per_rf_concat_timesteps[nnz_above] -= 0.0005 #*= (1.0 - 0.001)
        #self.per_rf_concat_timesteps[self.per_rf_concat_timesteps < 1] = 1
        #self.per_rf_concat_timesteps[self.per_rf_concat_timesteps >= self.max_input_concat_timesteps] = self.max_input_concat_timesteps - 1e-3

        per_rf_time_index = self.per_rf_concat_timesteps.astype(np.int) - 1

        self.last_per_rf_time_index = per_rf_time_index.copy()

        if True:
            tmp_1 = np.multiply(self.weights[0:self.total_released, :], state_to_learn[per_rf_time_index, :])
            tmp_2_cpu = np.sum(tmp_1, axis=1)
            eff_frame = np.divide(tmp_2_cpu, tmp_3_cpu)

            scaling_on = True
            if scaling_on:
                eff_frame = np.multiply(eff_frame, self.scaling_factor)

            best_rf = np.argmax(eff_frame)
        else:
            err_frame = np.sum(np.abs(self.weights - state_to_learn[per_rf_time_index, :]), axis=1)
            #err_frame = np.divide(err_frame, tmp_3_cpu + 1e-9)

            scaling_on = True  # TODO UNSURE IF NEED? PROBABLY. NEED TO MEASURE AND PLOT!!!
            if scaling_on:
                err_frame = np.multiply(err_frame, self.scaling_factor)

            best_rf = np.argmin(err_frame)




        # adjust scaling factor based on firing rates

        if self.t > self.rate_calc_timescale:

            if self.t - self.last_adjust_time > self.rate_calc_timescale:  # adjust all
                counts = np.sum(self.rfs_0_raster_history[:, self.t - self.rate_calc_timescale:self.t], axis=1)
                self.mean_rates[:] = counts * 1.0 / 200

                below = np.nonzero(self.mean_rates < self.target_rate)
                above = np.nonzero(self.mean_rates > self.target_rate)

                #self.scaling_factor[below] = self.scaling_factor[below] * (1.0 - self.scale_factor_delta)
                #self.scaling_factor[above] = self.scaling_factor[above] * (1.0 + self.scale_factor_delta)

                self.scaling_factor[below] = self.scaling_factor[below] + self.scale_factor_delta
                self.scaling_factor[above] = self.scaling_factor[above] - self.scale_factor_delta
                self.scaling_factor[self.scaling_factor < self.scale_factor_delta] = self.scale_factor_delta

                self.last_adjust_time = self.t

            if 0:  # only winner
                counts = np.sum(self.rfs_0_raster_history[:, self.t - 200:self.t], axis=1)
                self.mean_rates[:] = counts * 1.0/200

                if self.mean_rates[best_rf] > target_rate:
                    self.scaling_factor[best_rf] -= 0.001
                    self.scaling_factor[best_rf] = max(0.001, self.scaling_factor[best_rf])

                if self.mean_rates[best_rf] < target_rate:
                    self.scaling_factor[best_rf] += 0.001




        # adjust scaling factors, for those that haven't fired based on if they were to fire next step (so continuously scale up)
        #self.scaling_factor = np.ones(self.num_rfs)

        if learning_on:
            lr = self.lr  # 0.01

            self.weights[best_rf, :] = lr * state_to_learn[per_rf_time_index[best_rf], :] + (1.0 - lr) * self.weights[best_rf, :]

            if self.enable_target_rf:
                if tmp_3_cpu[best_rf] < self.target_rf_sum:
                    self.per_rf_concat_timesteps[best_rf] += 0.0005
                else:
                    self.per_rf_concat_timesteps[best_rf] -= 0.0005

                if self.per_rf_concat_timesteps[best_rf] >= self.max_input_concat_timesteps:
                    self.per_rf_concat_timesteps[best_rf] = self.max_input_concat_timesteps - 1e-3
                if self.per_rf_concat_timesteps[best_rf] < 0:
                    self.per_rf_concat_timesteps[best_rf] = 0

        self.rfs_0_raster_history[best_rf, self.t] = 1
        self.last_event_times[best_rf] = self.t

        # expected activity per RF: 1/N frames
        num_rf = self.weights.shape[0]

        slow_unlearn = True
        if slow_unlearn:
            pass
            #self.weights *= (1.0 - 0.0001 * 0.5)  # (1.0 - 0.00001 * 0.5)

        layer_output = np.zeros(self.num_rfs)
        layer_output[best_rf] = 1

        return layer_output

    def get_rfs_im(self):

        if False:  #self.layer_name == '0':
            print('************')
            print('mean_rates')
            print(1.0 / (1e-9 + self.mean_rates))
            print('scaling factor')
            print(self.scaling_factor)

        if False:  # self.layer_name == '0':
            print('****************')
            print('concat timesteps:')
            print(self.per_rf_concat_timesteps)
            print('per rf time index:')
            print(self.last_per_rf_time_index)
            print('sums:')
            print(self.last_rfs_sums)
            print('state to learn cumsum:')
            print(self.last_state_to_learn)

        if self.input_im_dim is not None and self.num_bins_per_pixel is not None:
            # this is a pixel-bin RF:
            rfs_im, rf_ims_dict = make_im(self.weights, num_bins_per_pixel=self.num_bins_per_pixel,
                             input_im_dim=self.input_im_dim,
                             im_final_dim=3000,  # 5000 for 1600 rfs
                             mod_for_disp=20,
                             normalize_weights=True)
        else:
            rfs_im = None
            rf_ims_dict = None
            # custom solution: needs to reference pixel-bin images; probably in superclass

        return rfs_im, rf_ims_dict

    def do_plots(self):
        self.ax.cla()
        num_rf = self.input_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.input_raster_history[0:num_rf, max(0, self.t - 200):self.t],
                                               np.arange(num_rf)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.fig.savefig("raster_" + self.layer_name + "_input.png", dpi=100)

        self.ax.cla()
        num_rf = self.rfs_0_raster_history.shape[0]
        raster_plot = np.transpose(np.multiply(self.rfs_0_raster_history[0:num_rf, max(0, self.t - 200):self.t],
                                               np.arange(num_rf)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.fig.savefig("raster_" + self.layer_name + "_output.png", dpi=100)


































