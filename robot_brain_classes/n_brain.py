import time
import cv2
import numpy as np
#np.set_printoptions(threshold=np.inf, linewidth=400)

import random
import pickle
from math import sqrt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.one_time_messages import OneTimeMessages
from brain_components_classes.states_history import StatesLimitedHistory
from cython_index import do_indexing, get_match_parallel


class NBrain(object):
    def __init__(self, params):
        '''

        'image_dim_NxN_pixels': IM_DIM,
            128
        'tile_dim_per_layer': TILE_DIM,
            [8, 16, 32, 64, 128]  # pixels, width & height
        'tile_tau_per_layer': TILE_TAU
            [1, 2,  4,  8,  16]  # spatiotemporal RF history steps
        'max_history_length': MAX_HISTORY_LENGTH,
            10600000 + 1
        'ims_scale_pixels': 800,

        :param params:


        IMPLEMENTATION
            initially, no overlap
                not strictly necessary for MVP
            ideally, minimize for-loops: minimize number of np.dot operations
                aim for: one dot product per layer
            each layer has two steps:
                collect input from previous layer
                dot product and normalize by sum

        '''

        self.im_dim = params['image_dim_NxN_pixels']
        self.tile_dim_per_layer = params['tile_dim_per_layer']
        self.tile_tau_per_layer = params['tile_tau_per_layer']
        self.ims_scale_pixels = params['ims_scale_pixels']
        self.max_time = params['max_history_length']

        self.num_bins_per_pixel = 6
        self.units_per_tile = 12 * 12  # units per WTA group, given one WTA per tile

        self.otm = OneTimeMessages()

        self.t = 0

        # input pixel history
        self.binned_input_image_history = StatesLimitedHistory(params={'max_delay': self.tile_tau_per_layer[0],
                                                                       'states_dim_list': [self.im_dim * self.im_dim * self.num_bins_per_pixel],
                                                                       'store_extra_data': False})

        self.n_layers = len(self.tile_dim_per_layer)

        self.layer_activity_histories = []
        self.layer_input_indices = []
        self.weights_per_layer = []
        self.gains_per_layer = []

        self.num_tiles_per_layer = []

        num_units_prev = self.im_dim * self.im_dim * self.num_bins_per_pixel

        self.input_features_arrs = []

        for layer_n in range(self.n_layers):
            print()
            print('Starting init of layer ', layer_n)
            print()
            # ? for layer 0, tile dim is directly input
            # ? for higher layer, need to match with previous layer tiles that are in the pixel dim space

            tile_dim = self.tile_dim_per_layer[layer_n]
            tile_tau = self.tile_tau_per_layer[layer_n]

            # assume non-overlap for now
            num_tiles_NxN = int(self.im_dim / tile_dim)

            # calculate total number of units
            num_tiles = num_tiles_NxN * num_tiles_NxN
            num_units = num_tiles * self.units_per_tile

            self.num_tiles_per_layer.append(num_tiles)

            if layer_n < self.n_layers - 1:
                # use next layer input params to determine how many steps of this layer activity to store
                max_delay = self.tile_tau_per_layer[layer_n + 1]
            else:
                # this is just for visualization, or prediction later, since no layer uses last layer for input:
                max_delay = 2 * self.tile_tau_per_layer[layer_n]  # TODO set this better

            # use StatesLimitedHistory, with store_extra_data: False (default is True)
            # needs to be initialized with max delay according to input time steps parameter of next layer

            layer_activity_history = StatesLimitedHistory(params={'max_delay': max_delay,
                                                                  'states_dim_list': [num_units],
                                                                  'store_extra_data': False})

            self.layer_activity_histories.append(layer_activity_history)

            # set self.layer_input_indices[layer_n] for use by _get_input_features
            input_indices = self._compute_input_indices(layer_n=layer_n)

            # quick check on shape of input_indices
            if layer_n == 0:
                scale_tmp = tile_dim * tile_dim
                input_features_len = int(scale_tmp * self.num_bins_per_pixel * tile_tau)
            else:
                # but easy check for now
                scale_tmp = (tile_dim / self.tile_dim_per_layer[layer_n - 1]) * (tile_dim / self.tile_dim_per_layer[layer_n - 1])
                assert scale_tmp == 4, 'consistency check for the moment; not necessary later'
                input_features_len = int(scale_tmp * self.units_per_tile * tile_tau)  # here, units per tile is referring to prev layer, tile tau is current layer

            assert input_indices.shape[0] == num_units
            assert input_indices.shape[1] == input_features_len, str((input_indices.shape[1], input_features_len))

            # also, make an assert on min and max of above to assure within expected bounds based on
            #   ... previous layer's init
            min_val = np.amin(input_indices)
            max_val = np.amax(input_indices)

            assert min_val == 0
            assert max_val == tile_tau * num_units_prev - 1, str((max_val, tile_tau * num_units_prev - 1))

            unique_input_indices = len(np.unique(input_indices))
            assert unique_input_indices == num_tiles * input_features_len, str((unique_input_indices, num_tiles * input_features_len))

            self.layer_input_indices.append(input_indices)

            # TODO still need a check on geometry; could be just displaying an example tile; to make sure right indices map, with delays, when actually doing the step

            # init weights, gains

            self.input_features_arrs.append(np.zeros((num_units, input_features_len), np.float32))

            self.weights_per_layer.append(np.random.random((num_units, input_features_len)).astype(np.float32))
            self.gains_per_layer.append(np.ones(num_units, np.float32))

            num_units_prev = num_units

    def _compute_input_indices(self, layer_n):
        '''

        :param layer_n:
        :return:

            input_indices
                shape: ()

                # input_indices.shape: (num_units, input_features_len)
                #       i.e. the shape is what we will expect input_features to be
                #  the indices are [0, len(tmp.flatten()))
                #       i.e. indices relative to flattened history of all previous layer activity
                # it already contains all the indexing so that:
                #    all units of the same tile in current layer, get the same set of inputs from previous layer
                #    and from the correct tiles in the previous layer
        '''

        tile_dim = self.tile_dim_per_layer[layer_n]
        tile_tau = self.tile_tau_per_layer[layer_n]

        # assume non-overlap for now
        num_tiles_NxN = int(self.im_dim / tile_dim)

        # calculate total number of units
        num_tiles = num_tiles_NxN * num_tiles_NxN
        num_units = num_tiles * self.units_per_tile

        if layer_n == 0:
            tile_dim_prev = 1  # one pixel
            num_tiles_NxN_prev = self.im_dim  # pixels

            # calculate total number of units prev
            num_tiles_prev = num_tiles_NxN_prev * num_tiles_NxN_prev
            num_units_prev = num_tiles_prev * self.num_bins_per_pixel
        else:
            tile_dim_prev = self.tile_dim_per_layer[layer_n - 1]
            num_tiles_NxN_prev = int(self.im_dim / tile_dim_prev)

            # calculate total number of units prev
            num_tiles_prev = num_tiles_NxN_prev * num_tiles_NxN_prev
            num_units_prev = num_tiles_prev * self.units_per_tile

        # for each tile of this unit, get input indices from previous layer

        input_indices = []

        for tile_row in range(num_tiles_NxN):
            for tile_col in range(num_tiles_NxN):
                tile_index = tile_row * num_tiles_NxN + tile_col

                # get tile indices of input (prev) layer, for this tile index
                # assuming no overlap

                # extend of this tile in terms of pixels
                r0 = tile_row * tile_dim
                r1 = (tile_row + 1) * tile_dim

                c0 = tile_col * tile_dim
                c1 = (tile_col + 1) * tile_dim

                assert r0 / tile_dim_prev == int(r0 / tile_dim_prev)
                assert r1 / tile_dim_prev == int(r1 / tile_dim_prev)
                assert c0 / tile_dim_prev == int(c0 / tile_dim_prev)
                assert c1 / tile_dim_prev == int(c1 / tile_dim_prev)

                tile_row_prev_first = int(r0 / tile_dim_prev)
                tile_row_prev_last = int(r1 / tile_dim_prev - 1)  # (.../... - 1) is last index!
                tile_col_prev_first = int(c0 / tile_dim_prev)
                tile_col_prev_last = int(c1 / tile_dim_prev - 1)  # (.../... - 1) is last index!

                prev_tile_indices = []
                prev_unit_indices = []

                if layer_n == 0:
                    n_tmp = self.num_bins_per_pixel
                else:
                    n_tmp = self.units_per_tile

                for prev_tile_r in range(tile_row_prev_first, tile_row_prev_last + 1):
                    for prev_tile_c in range(tile_col_prev_first, tile_col_prev_last + 1):
                        prev_tile_index = prev_tile_r * num_tiles_NxN_prev + prev_tile_c
                        prev_tile_indices.append(prev_tile_index)

                        for prev_unit_i in range(prev_tile_index * n_tmp, (prev_tile_index + 1) * n_tmp):
                            prev_unit_indices.append(prev_unit_i)

                # translate to indices in flattened history
                # need to know: what does prev layer flattened look like as far as indices?
                #   i.e. tmp.flatten() from _get_input_features
                #   where:
                #       tmp.shape: (time_steps_delay, state_dim)
                #       where state_dim is for previous layer
                #       all units of all tiles
                #       if prev layer is pixels: all bins for all pixels
                #       order: tmp[0,  :]: oldest
                #       tmp[-1, :]: newest

                # prev_tile_indices = np.array(prev_tile_indices)

                prev_unit_indices = np.array(prev_unit_indices)
                tile_input_indices = np.array([])

                for tau in range(tile_tau):
                    # is this right?
                    assert 0 <= np.amin(prev_unit_indices)
                    assert np.amax(prev_unit_indices) <= num_units_prev - 1, str((np.amax(prev_unit_indices), num_units_prev - 1))

                    tile_input_indices = np.concatenate((tile_input_indices, prev_unit_indices + tau * num_units_prev))

                for unit_in_tile in range(self.units_per_tile):
                    input_indices.append(tile_input_indices)

                debug_print = 0
                if debug_print:
                    print()
                    print('layer_n', layer_n)
                    print()
                    print('tile_dim', tile_dim)
                    print('num_tiles_NxN', num_tiles_NxN)
                    print()
                    print('tile_row', tile_row)
                    print('tile_col', tile_col)
                    print()
                    print('tile_row_prev_first', tile_row_prev_first)
                    print('tile_row_prev_last', tile_row_prev_last)
                    print('tile_col_prev_first', tile_col_prev_first)
                    print('tile_col_prev_last', tile_col_prev_last)
                    print()
                    print('prev_tile_indices')
                    print(prev_tile_indices)
                    print()

        input_indices = np.array(input_indices, np.int)

        debug_print_2 = 1
        if debug_print_2:
            print()
            print('layer_n', layer_n)
            print('input_indices', input_indices.shape)
            #print('input_indices')
            #print(input_indices)
            print()

        # input_indices.shape: (num_units, input_features_len)
        print(input_indices.dtype, np.amax(input_indices))
        input_indices = input_indices.astype(np.int32)
        print(input_indices.dtype, np.amax(input_indices))

        return input_indices

    #@profile
    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        # compute binning for whole image, and store
        # compute actual (binned pixel, per tile) input_features here for layer 0
        input_pixels = input_im.flatten()[np.newaxis, :]
        binned_pixels = self._bin_pixels_expand_columns(arr=input_pixels, num_bins_per_pixel=self.num_bins_per_pixel)

        # needs to be initialized with max delay according to input time steps parameter of next layer
        self.binned_input_image_history.process_new_states(newest_states_list=[binned_pixels])

        for layer_n in range(self.n_layers):

            layer_weights = self.weights_per_layer[layer_n]
            layer_gains = self.gains_per_layer[layer_n]

            num_units = layer_weights.shape[0]
            input_feature_len = layer_weights.shape[1]

            # complication: input features are a delayed sequence of prev layer activities
            # i.e. this should read from a limited states history
            # assumes always delta time step of 1; i.e. specify delay steps M, but assume D=1
            # this function also needs to be aware of the tile to input mapping

            # 22
            input_features = self._get_input_features(layer_index=layer_n, time_steps=self.tile_tau_per_layer[layer_n])

            # make sure input features are same shape as unit weights
            # each unit will have its own input; but units of the same tile will get the same input
            # number of units in the layer, total:
            assert input_features.shape[0] == num_units, str((layer_weights.shape, input_features.shape))
            # input feature length for all units in this layer:
            assert input_features.shape[1] == input_feature_len, str((layer_weights.shape, input_features.shape))

            # length: number of units in the layer total:
            # another way to think of this for speedup: many matrix-vector .dot operations in parallel (one per tile: weights of tile units * tile input vector)

            # commented, and compose to individual lines, for profiling:
            #match_no_gain = np.divide(np.sum(np.multiply(layer_weights, input_features), axis=1), np.sum(layer_weights, axis=1))

            use_cython = False
            if use_cython:
                # THIS IS BROKEN AND SLOW:

                num_threads = np.int32(8)
                rows_per_thread = np.int32(layer_weights.shape[0] / num_threads)
                assert rows_per_thread == layer_weights.shape[0] / num_threads

                match_no_gain = np.zeros(layer_weights.shape[0], np.float32)
                tmp_1 = np.zeros((match_no_gain.shape[0], 1), np.float32)
                tmp_2 = np.zeros((match_no_gain.shape[0], 1), np.float32)
                get_match_parallel(layer_weights, input_features, match_no_gain, tmp_1, tmp_2, rows_per_thread, num_threads)

            else:
                # 38
                tmp_1 = np.multiply(layer_weights, input_features)
                # 17
                tmp_2 = np.sum(tmp_1, axis=1)
                # 15
                tmp_3 = np.sum(layer_weights, axis=1)
                match_no_gain = np.divide(tmp_2, tmp_3)

            # length: number of units in the layer total:
            match = np.multiply(match_no_gain, layer_gains)

            # print(layer_weights.dtype, input_features.dtype, tmp_1.dtype, tmp_2.dtype, tmp_3.dtype, match_no_gain.dtype, layer_gains.dtype, match.dtype)

            # now we need to get win row, per tile (winner per WTA group)
            # win_row_indices will be absolute indices in the range [0, total units in layer over all tiles]
            win_row_indices = self._get_win_rows(layer_index=layer_n, num_tiles=self.num_tiles_per_layer[layer_n], match=match)  # this function will have to reshape to (n_tiles, n_rows_per_tile) to get argmax per tile

            # checks on win row indices shape, min/max, unique

            assert win_row_indices.shape[0] == self.num_tiles_per_layer[layer_n]

            # commented because might be slow:
            # assert len(np.unique(win_row_indices)) == len(win_row_indices), str(win_row_indices)

            learn_rate_p = 0.0025 * abs(1.0 - match[win_row_indices])

            # adjust weights for all tile winners based on their input features

            term_1 = (1.0 - learn_rate_p[:, np.newaxis]) * layer_weights[win_row_indices, :]
            term_2 = np.multiply(learn_rate_p[:, np.newaxis], input_features[win_row_indices, :])

            self.weights_per_layer[layer_n][win_row_indices, :] = term_1 + term_2

            # adjust gains: winners reset to 1.0, all others increase slowly
            self.gains_per_layer[layer_n] = layer_gains * 1.0005
            self.gains_per_layer[layer_n][win_row_indices] = 1.0

            # update layer activity history
            layer_activity = np.zeros(num_units, np.float32)
            layer_activity[win_row_indices] = 1.0

            # later, we can store this as sparse, if memory for this becomes a bottleneck:
            # needs to be initialized with max delay according to input time steps parameter of next layer
            self.layer_activity_histories[layer_n].process_new_states(newest_states_list=[layer_activity])

    #@profile
    def _get_input_features(self, layer_index, time_steps):
        '''

        # assumes always delta time step of 1; i.e. specify delay steps M, but assume D=1
        # this function also needs to be aware of the tile to input mapping

        :param prev_layer_index:
        :param time_steps:
        :return:

        input_fetures
            this is the input features to the next layer, based on activity history of prev_layer_index
            shape: (num_units, input_feature_len)



        '''

        prev_layer_index = layer_index - 1

        if layer_index == 0:
            tmp = self.binned_input_image_history.get_state_sequence(state_index=0, delay_long=time_steps - 1, delay_short=0)
        else:
            tmp = self.layer_activity_histories[prev_layer_index].get_state_sequence(state_index=0, delay_long=time_steps - 1, delay_short=0)

        # tmp.shape: (time_steps_delay, state_dim)
        #   where state_dim is for previous layer
        #   all units of all tiles
        #   if prev layer is pixels: all bins for all pixels

        # order: tmp[0,  :]: oldest
        #        tmp[-1, :]: newest

        # tmp is full activity of previous layer
        # now we need to take subsets of it, as input per tile of layer_index
        #   when to flatten over delays?
        #   how storing subset indices?

        input_indices = self.layer_input_indices[layer_index]

        # input_indices.shape: (num_units, input_features_len)
        #       i.e. the shape is what we will expect input_features to be
        #  the indices are [0, len(tmp.flatten()))
        #       i.e. indices relative to flattened history of all previous layer activity
        # it already contains all the indexing so that:
        #    all units of the same tile in current layer, get the same set of inputs from previous layer
        #    and from the correct tiles in the previous layer
        # there could be a way to do this without .flatten() being necessary...

        tmp_0 = tmp.flatten()

        # print(input_indices)

        # slow:
        # input_features = tmp_0[input_indices]

        use_cython = True
        if use_cython:
            # fast: confirmed the speedup on the desktop
            #input_features = np.zeros((input_indices.shape[0], input_indices.shape[1]), np.float32)
            input_features = self.input_features_arrs[layer_index]
            do_indexing(input_indices, input_features, tmp_0)
        else:
            input_features = tmp_0[input_indices]

        # print(tmp.shape, tmp_0.shape, input_indices.shape)
        # print(tmp.dtype, tmp_0.dtype, input_features.dtype)

        return input_features

    def _get_win_rows(self, layer_index, num_tiles, match):
        '''

        :param layer_index:
        :param match:
        :return:

        win_row_indices
            # now we need to get win row, per tile (winner per WTA group)
            # win_row_indices will be absolute indices in the range [0, total units in layer over all tiles]

        '''
        # print()
        # print('_get_win_rows')
        # print('layer_index', layer_index)
        # print(match.shape)

        # this function will have to reshape match to (n_tiles, units_per_tile) to get argmax per tile
        # is this the right ordering?
        # yes because: units for the same tile are sequentially arranged in activity array
        # TODO verify this via display or assert

        match_per_tile = match.reshape(num_tiles, self.units_per_tile)

        win_row_indices = np.argmax(match_per_tile, axis=1)
        win_row_indices = win_row_indices + self.units_per_tile * np.arange(num_tiles)

        return win_row_indices

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
        #print('***')
        #print('arr', arr.shape)

        bins = np.linspace(0, 1+1e-9, num_bins_per_pixel + 1)  # TODO to fix binning add 1e-9 to 1 in second argument!
        #print('bins', bins.shape, bins)
        bin_indices = np.digitize(arr, bins) - 1  # same shape as arr; which bin, per pixel
        #print('bin_indices', bin_indices.shape, bin_indices)
        arr_exp = np.zeros((arr.shape[0], arr.shape[1] * num_bins_per_pixel), np.float32)
        #print('arr_exp', arr_exp.shape)

        #term1 = np.tile(self.num_bins_per_pixel * np.arange(arr.shape[1]), 2)  # can't remember why this is np.tile(..., 2)
        term1 = num_bins_per_pixel * np.arange(arr.shape[1])  # only works if arr rows is 1?
        term2 = bin_indices.flatten()
        #print('term1', term1.shape, term1)
        #print('term2', term2.shape, term2)
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

        #print(max_vals.shape, weights.shape)

        # weighted mean val for display:
        # tmp1 = np.multiply(tmp, bins[np.newaxis, 0:num_bins_per_pixel])
        # tmp2 = np.sum(tmp1, axis=1)  # 65536
        # tmp3 = np.divide(tmp2, np.sum(tmp, axis=1))
        # max_vals = tmp3

        weights = weights.reshape(num_rows, num_pixels)
        arr = max_vals.reshape(num_rows, num_pixels)

        return arr, weights

    def get_table_ims(self):
        '''

        :return:
        '''


        # TODO
        # TODO
        # TODO: instead of distributing units per tile in this haphazard way,
        # TODO: make the image block rows: one row per tile, image block columns: one column per unit in the tile
        # TODO
        # TODO

        ims_list = []
        ims_names_list = []

        rows, weights = self._collapse_binned_columns_to_pixels(self.weights_per_layer[0], num_bins_per_pixel=self.num_bins_per_pixel)

        tile_r_c = int(sqrt(rows.shape[1]))
        N = int(sqrt(self.weights_per_layer[0].shape[0]))

        table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

        weights_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5
        tile_n = 0

        r_offset = 0
        for disp_r in range(N):

            c_offset = 0
            for disp_c in range(N):
                r0 = disp_r * tile_r_c + r_offset
                r1 = (disp_r + 1) * tile_r_c + r_offset
                c0 = disp_c * tile_r_c + c_offset
                c1 = (disp_c + 1) * tile_r_c + c_offset

                tile_im = rows[tile_n, :].reshape((tile_r_c, tile_r_c))
                tile_weights = weights[tile_n, :].reshape((tile_r_c, tile_r_c))

                table_im[r0:r1, c0:c1] = tile_im
                weights_im[r0:r1, c0:c1] = tile_weights

                # if tile_n == last_win_row:
                #     table_im[r0:r1, c0 - 1] = 1.0
                #     table_im[r0:r1, c1] = 1.0
                #     table_im[r0 - 1, c0:c1] = 1.0
                #     table_im[r1, c0:c1] = 1.0
                #
                #     weights_im[r0:r1, c0 - 1] = 1.0
                #     weights_im[r0:r1, c1] = 1.0
                #     weights_im[r0 - 1, c0:c1] = 1.0
                #     weights_im[r1, c0:c1] = 1.0

                tile_n += 1
                c_offset += 1

            r_offset += 1

        max_dim = max(table_im.shape[0], table_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list.append(table_im.copy())
        ims_names_list.append('table_im')

        return ims_list, ims_names_list
