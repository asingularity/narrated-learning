


import time
import cv2
import numpy as np
import random
import pickle
from math import sqrt

from cuda_dist_query import CudaTable
from utils.one_time_messages import OneTimeMessages


from brain_components_classes.perceptron import Perceptron


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt





from utils.one_time_messages import OneTimeMessages



class ExpBrain(object):
    '''

    first step:
        feedforward-only: test WTA mechanism + homeostatic gain adaptation
    next step:
        add predictive weights

    '''

    def __init__(self, params):

        self.im_dim = params['image_dim_NxN_pixels']

        self.ims_scale_pixels = 800
        self.otm = OneTimeMessages()

        self.t = 0

        self.last_im = None

        self.rows = 36

        self.NxN_output = 1  # predicted output square size
        self.NxN_input_pad = 4  # + pixels to pad on every side for input relative to output square

        self.bins_per_pixel = 20

        self.NxN_input = int(self.NxN_output + 2 * self.NxN_input_pad)

        # for now, table is not spatiotemporal history, just spatial, as a start
        self.input_history_steps = 1
        assert self.input_history_steps == 1, 'havent implement history for input yet'

        self.input_feature_len = self.bins_per_pixel * self.NxN_input * self.NxN_input * self.input_history_steps
        self.output_feature_len = self.bins_per_pixel * self.NxN_output * self.NxN_output

        random_init = True
        if random_init:
            self.table_i = np.random.random((self.rows, self.input_feature_len)).astype(np.float32)
        else:
            self.table_i = np.zeros((self.rows, self.input_feature_len), np.float32)

        # top left of input region
        self.in_r = 50
        self.in_c = 64

        # top left of input region
        # self.in_r = 30
        # self.in_c = 34

        print()
        print('Initialized:')
        print('    (tl) in_r:', self.in_r)
        print('    (tl) in_c:', self.in_c)
        print('    NxN_input:', self.NxN_input)
        print()

        # gain

        self.gains = np.ones(self.rows, np.float32)
        # self.last_win_t = np.zeros()

        self.MAX_TIME = 1000000
        self.WTA_winner_history = np.zeros(self.MAX_TIME, np.int)
        self.input_match_history = np.zeros(self.MAX_TIME, np.float32)
        self.mean_input_match_history = np.zeros(self.MAX_TIME, np.float32)
        self.mean_gain_history = np.zeros(self.MAX_TIME, np.float32)
        self.mean_mean_gain_history = np.zeros(self.MAX_TIME, np.float32)

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

    def process_input(self, input_im):

        if self.last_im is not None:

            # get input subset from image

            assert self.in_r + self.NxN_input < self.im_dim, str((self.in_r + self.NxN_input, self.im_dim))
            assert self.in_c + self.NxN_input < self.im_dim, str((self.in_c + self.NxN_input, self.im_dim))

            input_pixels = self.last_im[self.in_r:self.in_r + self.NxN_input, self.in_c:self.in_c + self.NxN_input]
            input_arr = input_pixels.flatten()[np.newaxis, :]
            input_exp = self._bin_pixels_expand_columns(arr=input_arr, num_bins_per_pixel=self.bins_per_pixel)

            assert input_exp.shape[0] == 1
            assert input_exp.shape[1] == self.input_feature_len, str((input_exp.shape[1], self.input_feature_len))

            all_mp = np.multiply(self.table_i, input_exp)
            match = np.divide(np.sum(all_mp, axis=1), np.sum(self.table_i, axis=1))

            match = np.multiply(match, self.gains)

            argsort_match = np.argsort(match)[::-1]  # largest match first
            win_row = argsort_match[0]

            learn_rate_p = 0.005 * abs(1.0 - match[argsort_match[0]])
            # this is too extreme (forces synchrony):
            # * self.gains[win_row]

            self.table_i[win_row, :] = (1.0 - learn_rate_p) * self.table_i[win_row, :] + learn_rate_p * input_exp

            self.gains = self.gains * 1.004
            self.gains[win_row] = 1.0

            self.WTA_winner_history[self.t] = win_row
            self.input_match_history[self.t] = match[win_row]
            self.mean_input_match_history[self.t] = np.mean(self.input_match_history[max(0, self.t - 1000):self.t])
            self.mean_gain_history[self.t] = np.mean(self.gains)
            self.mean_mean_gain_history[self.t] = np.mean(self.mean_gain_history[max(0, self.t - 1000):self.t])
            self.t += 1

        self.last_im = input_im.copy()

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

        # weighted mean val for display:
        # tmp1 = np.multiply(tmp, bins[np.newaxis, 0:num_bins_per_pixel])
        # tmp2 = np.sum(tmp1, axis=1)  # 65536
        # tmp3 = np.divide(tmp2, np.sum(tmp, axis=1))
        # max_vals = tmp3

        arr = max_vals.reshape(num_rows, num_pixels)

        return arr

    def get_table_ims(self):
        rows = self._collapse_binned_columns_to_pixels(self.table_i, num_bins_per_pixel=self.bins_per_pixel)

        tile_r_c = int(sqrt(rows.shape[1]))
        N = int(sqrt(self.rows))

        table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

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

                table_im[r0:r1, c0:c1] = tile_im

                tile_n += 1
                c_offset += 1

            r_offset += 1

        # selection image
        select_im = None

        # scale all

        max_dim = max(table_im.shape[0], table_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list = [table_im.copy()]
        ims_names_list = ['table_i']

        # plot mat plot lib
        print()
        print('Making plot of state vars...')
        self.ax.cla()
        self.ax.plot(self.WTA_winner_history[0:self.t], 'r.')
        self.fig.savefig("WTA_winner_history.png", dpi=100)
        self.ax.cla()
        self.ax.plot(self.mean_input_match_history[0:self.t], 'r-')
        self.fig.savefig("mean_input_match_history.png", dpi=100)
        self.ax.cla()
        self.ax.plot(self.mean_gain_history[0:self.t], 'r-')
        self.fig.savefig("mean_gain_history.png", dpi=100)
        self.ax.cla()
        self.ax.plot(self.mean_mean_gain_history[0:self.t], 'r-')
        self.fig.savefig("mean_mean_gain_history.png", dpi=100)
        print('Done.')
        print()


        return ims_list, ims_names_list


class ExpBrainSinglePixelBin(object):
    def __init__(self, params):

        im_dim = params['image_dim_NxN_pixels']

        self.ims_scale_pixels = 1200
        self.otm = OneTimeMessages()

        self.t = 0

        self.count = 0
        self.last_im = None

        self.rows = 9

        self.NxN_output = 1  # predicted output square size
        self.NxN_input_pad = 8  # + pixels to pad on every side for input relative to output square

        self.bins_per_pixel = 20

        self.NxN_input = int(self.NxN_output + 2 * self.NxN_input_pad)

        # for now, table is not spatiotemporal history, just spatial, as a start
        self.input_history_steps = 1
        assert self.input_history_steps == 1, 'havent implement history for input yet'

        self.input_feature_len = self.bins_per_pixel * self.NxN_input * self.NxN_input * self.input_history_steps
        self.output_feature_len = self.bins_per_pixel * self.NxN_output * self.NxN_output

        random_init = True
        if random_init:
            self.table_i = np.random.random((self.rows, self.input_feature_len)).astype(np.float32)
        else:
            self.table_i = np.zeros((self.rows, self.input_feature_len), np.float32)

        self.table_o = np.zeros((self.rows, self.output_feature_len))

        # top left of input region
        self.in_r = 50
        self.in_c = 64

        # top left of prediction region
        self.out_r = self.in_r + self.NxN_input_pad
        self.out_c = self.in_c + self.NxN_input_pad

        # middle of region
        self.mid_r = (2 * self.in_r + self.NxN_input - 1) / 2
        assert self.mid_r == int(self.mid_r)
        self.mid_r = int(self.mid_r)

        self.mid_c = (2 * self.in_c + self.NxN_input - 1) / 2
        assert self.mid_c == int(self.mid_c)
        self.mid_c = int(self.mid_c)

        print()
        print('Initialized:')
        print('    (tl) in_r:', self.in_r)
        print('    (tl) in_c:', self.in_c)
        print('    NxN_input:', self.NxN_input)
        print('    (tl) out_r:', self.out_r)
        print('    (tl) out_c:', self.out_c)
        print('    NxN_output:', self.NxN_output)
        print('    mid_r:', self.mid_r)
        print('    mid_c:', self.mid_c)
        print()

    def process_input(self, input_im):

        if self.last_im is not None:

            # get input subset from image
            # get output subset from image

            input_pixels = self.last_im[self.in_r:self.in_r + self.NxN_input, self.in_c:self.in_c + self.NxN_input]
            output_pixels = input_im[self.out_r:self.out_r + self.NxN_output, self.out_c:self.out_c + self.NxN_output]

            # compute pixel bins: copy from file where we already wrote this
            input_arr = input_pixels.flatten()[np.newaxis, :]
            output_arr = output_pixels.flatten()[np.newaxis, :]

            input_exp = self._bin_pixels_expand_columns(arr=input_arr, num_bins_per_pixel=self.bins_per_pixel)
            output_exp = self._bin_pixels_expand_columns(arr=output_arr, num_bins_per_pixel=self.bins_per_pixel)

            assert input_exp.shape[0] == 1
            assert input_exp.shape[1] == self.input_feature_len

            assert output_exp.shape[0] == 1
            assert output_exp.shape[1] == self.output_feature_len

            all_mp = np.multiply(self.table_i, input_exp)
            match = np.divide(np.sum(all_mp, axis=1), np.sum(self.table_i, axis=1))
            argsort_match = np.argsort(match)[::-1]  # largest match first
            #print(argsort_match[0])

            assert 0.0 <= np.amin(match) <= 1.0
            assert 0.0 <= np.amax(match) <= 1.0

            # what we want when learning is done:
            #   if output pixel-bin was HIGH in the future:
            #       one and only one row was 1.0 match, others were 0.0 match
            #   if output pixel-bin was LOW in the future:
            #       all rows were 0.0 match
            #   (implied)
            #       all rows of input table are used
            #       all rows of input table are different
            #       learning rate 0 if no error (conditions above are met)

            # learning rule for this:
            #   if output pixel-bin was HIGH in training output:
            #       winner learns: (large learning rate) * abs(1.0 - input_match)
            #       other rows: (learning rate) * abs(0.0 - input_match)
            #   if output pixel-bin was LOW in training output:
            #       all rows: (learning rate) * abs(0.0 - input_match)
            #   + slow drift up?

            # for now, we are predicted only one pixel-bin:

            pixel_bin = range(2, 3)  # [0, self.bins_per_pixel]
            output_val = np.amax(output_exp[0, pixel_bin])

            if output_val == 1:
                self.count += 1

                #print(match[argsort_match[0]], match[argsort_match[1::]])

                learn_rate_p = 0.0005*4 * abs(1.0 - match[argsort_match[0]])

                self.table_i[argsort_match[0], :] = (1.0 - learn_rate_p) * self.table_i[argsort_match[0], :] + learn_rate_p * input_exp
            else:
                learn_rate_p = -0.0005 * abs(0.0 - match[argsort_match[0]])

                self.table_i[argsort_match[0], :] = (1.0 - learn_rate_p) * self.table_i[argsort_match[0], :] + learn_rate_p * input_exp

            self.table_i[self.table_i < 0.0] = 0.0
            self.table_i[self.table_i > 1.0] = 1.0

            #self.table_i += 0.001


        self.last_im = input_im.copy()

    def process_input_1(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        # TODO TODO !!!!!!!!! TODO set init_fast back to true in visualizer !!!!!!!

        if self.last_im is not None:

            # get input subset from image
            # get output subset from image

            input_pixels = self.last_im[self.in_r:self.in_r + self.NxN_input, self.in_c:self.in_c + self.NxN_input]
            output_pixels = input_im[self.out_r:self.out_r + self.NxN_output, self.out_c:self.out_c + self.NxN_output]

            # compute pixel bins: copy from file where we already wrote this
            input_arr = input_pixels.flatten()[np.newaxis, :]
            output_arr = output_pixels.flatten()[np.newaxis, :]

            input_exp = self._bin_pixels_expand_columns(arr=input_arr, num_bins_per_pixel=self.bins_per_pixel)
            output_exp = self._bin_pixels_expand_columns(arr=output_arr, num_bins_per_pixel=self.bins_per_pixel)

            assert input_exp.shape[0] == 1
            assert input_exp.shape[1] == self.input_feature_len

            assert output_exp.shape[0] == 1
            assert output_exp.shape[1] == self.output_feature_len

            # compute input matches, threshold? k-WTA? k dynamic based on prediction match threshold? until min remainder of input? every input pixel explained or max?
            # call units "active" in rank order of input match, until:
            #       (option 1) remainder of input unexplained < threshold OR exceed number
            #       (option 2) per-pixel remainder < threshold OR exceed number

            # select which are active

            all_mp = np.multiply(self.table_i, input_exp)
            #print(all_mp.shape, np.amin(all_mp), np.amax(all_mp))
            match = np.divide(np.sum(all_mp, axis=1), np.sum(self.table_i, axis=1))
            argsort_match = np.argsort(match)[::-1]  # largest match first

            unexplained = input_exp.copy()
            active_rows = []

            for k in range(argsort_match.shape[0]):

                row_i = argsort_match[k]
                unexplained = unexplained - self.table_i[row_i, :]
                unexplained[unexplained < 0] = 0
                val = np.sum(unexplained) / np.sum(input_exp)
                #print(k, row_i, val)
                active_rows.append(row_i)
                if val < 0.05:
                    break
            active_rows = np.array(active_rows, np.int)

            # update table_o based on output probabilities and active input rows

            self.table_o[active_rows, :]  = 0.95 * self.table_o[active_rows, :] + 0.05 * output_exp

            # update table_i to evolve towards goal:
            #   for a particular input,
            #   only one row predicted, with "1.0" prediction, each pixel-bin, and all other rows 0.0 for that pixel-bin
            #   i.e. pick which row should win, for each 1.0 value in output_exp

            # TODO realized this doesn't make sense:
            # print(output_exp)
            # nnz_out = np.nonzero(output_exp)[1]  # len 9, which pixel-bins overall were active in training output
            # for each of the pixel-bins above, find which of the active rows in table_o had max value
            # table_o_active = self.table_o[active_rows, :]

            self.table_i[active_rows, :] = 0.95 * self.table_i[active_rows, :] + 0.05 * input_exp

            # match_out = np.multiply(self.table_o[active_rows, :], output_exp)


        self.last_im = input_im.copy()

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

        # weighted mean val for display:
        # tmp1 = np.multiply(tmp, bins[np.newaxis, 0:num_bins_per_pixel])
        # tmp2 = np.sum(tmp1, axis=1)  # 65536
        # tmp3 = np.divide(tmp2, np.sum(tmp, axis=1))
        # max_vals = tmp3

        arr = max_vals.reshape(num_rows, num_pixels)

        return arr

    def get_table_ims(self):
        rows = self._collapse_binned_columns_to_pixels(self.table_i, num_bins_per_pixel=self.bins_per_pixel)

        tile_r_c = int(sqrt(rows.shape[1]))
        N = int(sqrt(self.rows))

        table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

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

                table_im[r0:r1, c0:c1] = tile_im

                tile_n += 1
                c_offset += 1

            r_offset += 1

        # selection image
        select_im = None

        # scale all

        max_dim = max(table_im.shape[0], table_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list = [table_im.copy()]
        ims_names_list = ['table_i']


        rows = self._collapse_binned_columns_to_pixels(self.table_o, num_bins_per_pixel=self.bins_per_pixel)

        tile_r_c = int(sqrt(rows.shape[1]))
        N = int(sqrt(self.rows))

        table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

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

                table_im[r0:r1, c0:c1] = tile_im

                tile_n += 1
                c_offset += 1

            r_offset += 1

        # selection image
        select_im = None

        # scale all

        max_dim = max(table_im.shape[0], table_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        #ims_list.append(table_im.copy())
        #ims_names_list.append('table_o')

        return ims_list, ims_names_list


    def get_table_ims_simple(self):

        ims_list = []
        ims_names_list = []

        input_table_im = self.table_i.copy()

        max_dim = max(input_table_im.shape[0], input_table_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        input_table_im = cv2.resize(input_table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list.append(input_table_im)
        ims_names_list.append('table_i')

        output_table_im = self.table_o.copy()

        max_dim = max(output_table_im.shape[0], output_table_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        output_table_im = cv2.resize(output_table_im, dsize=(0, 0), fx=imscale, fy=imscale,
                                    interpolation=cv2.INTER_NEAREST)

        ims_list.append(output_table_im)
        ims_names_list.append('table_o')

        return ims_list, ims_names_list







    def process_input_OLD_one_pixel_value_position(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        px_r = self.px_r
        px_c = self.px_c
        px_v = self.px_v

        # don't learn after repeat
        if self.t < 32300:
            if px_v[0] <= int(255 * input_im[px_r, px_c]) <= px_v[1]:
                if self.last_im is not None:
                    query_in = self.last_im .flatten()[np.newaxis, :].copy()
                    self._learn_table(query_inputs=query_in)
        else:
            self.otm.print_once('learning stopped!')

        self.t += 1
        self.last_im = input_im.copy()

    def _learn_table(self, query_inputs):

        dists, argmin_dists = self.table.query_multiple_rows(query_inputs=query_inputs)

        dists = dists.flatten()
        argmin_dist = argmin_dists[0]

        if self.init_I_row_num < self.table.get_num_rows():
            self.table.set_matrix_row(row_index=self.init_I_row_num,
                                      row_input=query_inputs[0, :],
                                      row_weights=None,
                                      row_to_table_dists=dists,
                                      fast_init=True)
            self.init_I_row_num += 1
        else:
            if not self.table.post_init_done:
                print()
                print(self.t, 'Table init done! doing post-init...')
                print()

                self.table.post_init()

        if self.table.post_init_done:

            table_min_dist, table_min_dist_r, table_min_dist_c = self.table.get_min_dist()  # table <-> table

            new_min_dist = np.amin(dists)  # row <-> table

            # TODO hack
            if self.t < 10000 and new_min_dist > table_min_dist:
                # print(self.t, 'replacing ', new_min_dist, table_min_dist)
                dists[table_min_dist_r] = np.inf

                # replace the current min dist row, with the new row
                self.table.set_matrix_row(row_index=table_min_dist_r,
                                          row_input=query_inputs[0, :],
                                          row_to_table_dists=dists)
                self.W[table_min_dist_r, :] = 0.5
            else:
                row_index = argmin_dist
                # learn weights
                table_row, _, _ = self.table.get_matrix_row(row_index=row_index)
                tile_input_vect = query_inputs[0, :]

                per_pixel_error = np.abs(tile_input_vect - table_row)
                per_pixel_match = 1.0 - per_pixel_error

                self.win_count[row_index] += 1

                self.W[row_index, :] = 0.8 * self.W[row_index, :] + 0.2 * per_pixel_match

    def save_model(self, models_save_folder):
        '''

        :param models_save_folder:
        :return:
        '''

    def OLD__init__(self, params):

        im_dim = params['image_dim_NxN_pixels']

        self.ims_scale_pixels = 1200

        rows = 9
        self.rows = rows

        self.table = CudaTable(num_entries=rows,
                               input_dim=im_dim * im_dim,
                               disable_row_row_dist=False,
                               enable_weight_bias=False)
        self.init_I_row_num = 0

        self.W = np.ones((rows, im_dim * im_dim), np.float32) * 0.5
        self.otm = OneTimeMessages()

        self.t = 0

        self.count = 0
        self.last_im = None

        self.win_count = np.zeros(rows, np.int)

        self.px_r = 60
        self.px_c = 70
        self.px_v = [10, 40]


    def get_table_ims_OLD(self):

        N = int(sqrt(self.rows))

        rows = self.table.table_i

        # print('***', np.amin(rows), np.amax(rows))
        # print('---', np.amin(cuda_table.table_i), np.amax(cuda_table.table_i))

        tile_r_c = int(sqrt(rows.shape[1]))

        table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5
        weight_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

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
                tile_im[self.px_r, self.px_c] = 0
                tile_im[self.px_r + 1, self.px_c] = 0
                tile_im[self.px_r - 1, self.px_c] = 0
                tile_im[self.px_r, self.px_c + 1] = 0
                tile_im[self.px_r, self.px_c - 1] = 0

                table_im[r0:r1, c0:c1] = tile_im

                tile_weights = self.W[tile_n, :].reshape((tile_r_c, tile_r_c))
                tile_weights[self.px_r, self.px_c] = 0
                tile_weights[self.px_r + 1, self.px_c] = 0
                tile_weights[self.px_r - 1, self.px_c] = 0
                tile_weights[self.px_r, self.px_c + 1] = 0
                tile_weights[self.px_r, self.px_c - 1] = 0

                weight_im[r0:r1, c0:c1] = tile_weights

                tile_n += 1
                c_offset += 1

            r_offset += 1

        #max_dim = max(table_im.shape[0], table_im.shape[1])
        #imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        #table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list = [table_im, weight_im]
        ims_names_list = ['table', 'weights']

        print(self.win_count)

        return ims_list, ims_names_list





class ExpBrainPerceptron(object):
    '''
    single layer perceptron pool-to-pool (first pool-to-one) with output gain adaptation mechanism
    '''

    def __init__(self, params):
        im_dim = params['image_dim_NxN_pixels']

        self.ims_scale_pixels = 200
        self.otm = OneTimeMessages()

        self.last_im = None

        # input bins

        self.NxN_output = 1  # predicted output square size
        self.NxN_input_pad = 4  # + pixels to pad on every side for input relative to output square

        self.bins_per_pixel = 20

        self.NxN_input = int(self.NxN_output + 2 * self.NxN_input_pad)

        # for now, table is not spatiotemporal history, just spatial, as a start
        self.input_history_steps = 1
        assert self.input_history_steps == 1, 'havent implement history for input yet'

        self.input_feature_len = self.bins_per_pixel * self.NxN_input * self.NxN_input * self.input_history_steps
        self.output_feature_len = self.bins_per_pixel * self.NxN_output * self.NxN_output

        # predictors
        self.N = 4

        self.predictors = []
        for k in range(self.N):
            self.predictors.append(Perceptron(no_of_inputs=self.input_feature_len))

        # input / output regions

        # top left of input region
        self.in_r = 50
        self.in_c = 64

        # top left of prediction region
        self.out_r = self.in_r + self.NxN_input_pad
        self.out_c = self.in_c + self.NxN_input_pad

        # middle of region
        self.mid_r = (2 * self.in_r + self.NxN_input - 1) / 2
        assert self.mid_r == int(self.mid_r)
        self.mid_r = int(self.mid_r)

        self.mid_c = (2 * self.in_c + self.NxN_input - 1) / 2
        assert self.mid_c == int(self.mid_c)
        self.mid_c = int(self.mid_c)

        print()
        print('Initialized:')
        print('    (tl) in_r:', self.in_r)
        print('    (tl) in_c:', self.in_c)
        print('    NxN_input:', self.NxN_input)
        print('    (tl) out_r:', self.out_r)
        print('    (tl) out_c:', self.out_c)
        print('    NxN_output:', self.NxN_output)
        print('    mid_r:', self.mid_r)
        print('    mid_c:', self.mid_c)
        print()

        # measure error over time
        self.t = 0
        self.MAX_TIME = 1000000
        self.predict_error_history = np.zeros(self.MAX_TIME)
        self.mean_error_history = np.zeros(self.MAX_TIME)
        self.error_t = 0

        self.winner_output_history = np.zeros(self.MAX_TIME)
        self.WTA_winner_history = np.zeros(self.MAX_TIME, np.int)
        self.average_gain_history = np.zeros(self.MAX_TIME)

        #self.mean_error_history_when_predict_one
        #self.t_when_predict_one
        #self.t_when_predict_zero

        # other

        self.count = 0

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

    def process_input(self, input_im):

        if self.last_im is not None:
            # get input subset from image
            # get output subset from image

            input_pixels = self.last_im[self.in_r:self.in_r + self.NxN_input, self.in_c:self.in_c + self.NxN_input]
            output_pixels = input_im[self.out_r:self.out_r + self.NxN_output, self.out_c:self.out_c + self.NxN_output]

            # compute pixel bins: copy from file where we already wrote this
            input_arr = input_pixels.flatten()[np.newaxis, :]
            output_arr = output_pixels.flatten()[np.newaxis, :]

            input_exp = self._bin_pixels_expand_columns(arr=input_arr, num_bins_per_pixel=self.bins_per_pixel)
            output_exp = self._bin_pixels_expand_columns(arr=output_arr, num_bins_per_pixel=self.bins_per_pixel)

            assert input_exp.shape[0] == 1
            assert input_exp.shape[1] == self.input_feature_len

            assert output_exp.shape[0] == 1
            assert output_exp.shape[1] == self.output_feature_len

            # for now, we are predicted only one pixel-bin:

            pixel_bin = range(2, 3)  # [0, self.bins_per_pixel]
            output_val = np.amax(output_exp[0, pixel_bin])

            # get outputs of perceptrons
            predictors_out = np.zeros(self.N, np.float32)
            for k in range(self.N):
                predictors_out[k] = self.predictors[k].result(inputs=input_exp.flatten())

            # learn the max predictor
            win_i = np.argmax(predictors_out)
            error_i = output_val - predictors_out[win_i]

            self.predictors[win_i].weight_adjustment(inputs=input_exp.flatten(), error=error_i)

            if output_val == 0:
                self.count += 1

                self.predict_error_history[self.error_t] = error_i
                self.mean_error_history[self.error_t] = np.mean(self.predict_error_history[max(0, self.error_t - 1000):self.error_t])

                self.error_t += 1

            # self.predict_error_history[self.t] = error_i
            # self.mean_error_history[self.] = np.mean(self.predict_error_history[max(0, self.t - 1000):self.t])
            # self.error_t += 1

            self.WTA_winner_history[self.t] = win_i
            self.average_gain_history[self.t] = 0.0
            self.winner_output_history[self.t] = predictors_out[win_i]

            self.t += 1

        self.last_im = input_im.copy()


    def get_table_ims(self):

        # display normalized perceptron weights

        table_i = np.zeros((self.N, self.input_feature_len))
        for k in range(self.N):
            row = self.predictors[k].w[0:-1]
            row = (row - np.amin(row)) * 1.0 / (np.amax(row) - np.amin(row))
            table_i[k, :] = row

        rows = self._collapse_binned_columns_to_pixels(table_i, num_bins_per_pixel=self.bins_per_pixel)

        tile_r_c = int(sqrt(rows.shape[1]))
        N = int(sqrt(self.N))

        table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

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

                table_im[r0:r1, c0:c1] = tile_im

                tile_n += 1
                c_offset += 1

            r_offset += 1

        # selection image
        select_im = None

        # scale all

        max_dim = max(table_im.shape[0], table_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list = [table_im.copy()]
        ims_names_list = ['table_i']

        print_large_arrays = False
        if print_large_arrays:
            print()
            print('TIME: ', self.t)
            print()
            print('predict error history:')
            print()
            print(self.predict_error_history[0:self.t])
            print()
            print('WTA_winner_history:')
            print()
            print(self.WTA_winner_history[0:self.t])
            print()
            print('average_gain_history:')
            print()
            print(self.average_gain_history[0:self.t])
            print()
            print('winner_output_history:')
            print()
            print(self.winner_output_history[0:self.t])
            print()
        else:
            # plot mat plot lib
            print()
            print('Making plot of state vars...')
            self.ax.cla()
            self.ax.plot(self.mean_error_history[0:self.error_t])
            #self.ax.plot(self.predict_error_history[0:self.error_t])

            self.fig.savefig("predict_error_history.png", dpi=100)

            self.ax.cla()
            #self.ax.plot(self.mean_error_history[0:self.error_t])
            self.ax.plot(self.WTA_winner_history[0:self.t], 'r.')
            self.fig.savefig("WTA_winner_history.png", dpi=100)

            # self.ax.plot(self.mean_error_history_when_predict_one[0:self.t])
            # self.ax.plot(self.mean_error_history_when_predict_zero[0:self.t])

            print('Done.')
            print()

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

        # weighted mean val for display:
        # tmp1 = np.multiply(tmp, bins[np.newaxis, 0:num_bins_per_pixel])
        # tmp2 = np.sum(tmp1, axis=1)  # 65536
        # tmp3 = np.divide(tmp2, np.sum(tmp, axis=1))
        # max_vals = tmp3

        arr = max_vals.reshape(num_rows, num_pixels)

        return arr


class ExpBrainOldPatternMatch(object):
    def __init__(self, params):

        im_dim = params['image_dim_NxN_pixels']

        self.ims_scale_pixels = 1200
        self.otm = OneTimeMessages()

        self.t = 0

        self.count = 0
        self.last_im = None

        self.rows = 9

        self.NxN_output = 1  # predicted output square size
        self.NxN_input_pad = 8  # + pixels to pad on every side for input relative to output square

        self.bins_per_pixel = 20

        self.NxN_input = int(self.NxN_output + 2 * self.NxN_input_pad)

        # for now, table is not spatiotemporal history, just spatial, as a start
        self.input_history_steps = 1
        assert self.input_history_steps == 1, 'havent implement history for input yet'

        self.input_feature_len = self.bins_per_pixel * self.NxN_input * self.NxN_input * self.input_history_steps
        self.output_feature_len = self.bins_per_pixel * self.NxN_output * self.NxN_output

        random_init = True
        if random_init:
            self.table_i = np.random.random((self.rows, self.input_feature_len)).astype(np.float32)
        else:
            self.table_i = np.zeros((self.rows, self.input_feature_len), np.float32)

        self.table_o = np.zeros((self.rows, self.output_feature_len))

        # top left of input region
        self.in_r = 50
        self.in_c = 64

        # top left of prediction region
        self.out_r = self.in_r + self.NxN_input_pad
        self.out_c = self.in_c + self.NxN_input_pad

        # middle of region
        self.mid_r = (2 * self.in_r + self.NxN_input - 1) / 2
        assert self.mid_r == int(self.mid_r)
        self.mid_r = int(self.mid_r)

        self.mid_c = (2 * self.in_c + self.NxN_input - 1) / 2
        assert self.mid_c == int(self.mid_c)
        self.mid_c = int(self.mid_c)

        print()
        print('Initialized:')
        print('    (tl) in_r:', self.in_r)
        print('    (tl) in_c:', self.in_c)
        print('    NxN_input:', self.NxN_input)
        print('    (tl) out_r:', self.out_r)
        print('    (tl) out_c:', self.out_c)
        print('    NxN_output:', self.NxN_output)
        print('    mid_r:', self.mid_r)
        print('    mid_c:', self.mid_c)
        print()

    def process_input(self, input_im):

        if self.last_im is not None:

            # get input subset from image
            # get output subset from image

            input_pixels = self.last_im[self.in_r:self.in_r + self.NxN_input, self.in_c:self.in_c + self.NxN_input]
            output_pixels = input_im[self.out_r:self.out_r + self.NxN_output, self.out_c:self.out_c + self.NxN_output]

            # compute pixel bins: copy from file where we already wrote this
            input_arr = input_pixels.flatten()[np.newaxis, :]
            output_arr = output_pixels.flatten()[np.newaxis, :]

            input_exp = self._bin_pixels_expand_columns(arr=input_arr, num_bins_per_pixel=self.bins_per_pixel)
            output_exp = self._bin_pixels_expand_columns(arr=output_arr, num_bins_per_pixel=self.bins_per_pixel)

            assert input_exp.shape[0] == 1
            assert input_exp.shape[1] == self.input_feature_len

            assert output_exp.shape[0] == 1
            assert output_exp.shape[1] == self.output_feature_len

            all_mp = np.multiply(self.table_i, input_exp)
            match = np.divide(np.sum(all_mp, axis=1), np.sum(self.table_i, axis=1))
            argsort_match = np.argsort(match)[::-1]  # largest match first

            assert 0.0 <= np.amin(match) <= 1.0
            assert 0.0 <= np.amax(match) <= 1.0

            # what we want when learning is done:
            #   if output pixel-bin was HIGH in the future:
            #       one and only one row was 1.0 match, others were 0.0 match
            #   if output pixel-bin was LOW in the future:
            #       all rows were 0.0 match
            #   (implied)
            #       all rows of input table are used
            #       all rows of input table are different
            #       learning rate 0 if no error (conditions above are met)

            # learning rule for this:
            #   if output pixel-bin was HIGH in training output:
            #       winner learns: (large learning rate) * abs(1.0 - input_match)
            #       other rows: (learning rate) * abs(0.0 - input_match)
            #   if output pixel-bin was LOW in training output:
            #       all rows: (learning rate) * abs(0.0 - input_match)
            #   + slow drift up?

            # for now, we are predicted only one pixel-bin:

            pixel_bin = range(2, 3)  # [0, self.bins_per_pixel]
            output_val = np.amax(output_exp[0, pixel_bin])

            if output_val == 1:
                self.count += 1

                #print(match[argsort_match[0]], match[argsort_match[1::]])

                learn_rate_p = 0.05 * abs(1.0 - match[argsort_match[0]])

                self.table_i[argsort_match[0], :] = (1.0 - learn_rate_p) * self.table_i[argsort_match[0], :] + learn_rate_p * input_exp

                #learn_rate_p_2 = learn_rate_p * 0.1
                #self.table_i[argsort_match[1::], :] = (1.0 - learn_rate_p_2) * self.table_i[argsort_match[1::], :] + learn_rate_p_2 * input_exp

                learn_rate_n = (0.05 * abs(0.0 - match[argsort_match[1::]]))[:, np.newaxis]
                self.table_i[argsort_match[1::], :] = np.multiply((1.0 - learn_rate_n), self.table_i[argsort_match[1::], :]) - np.multiply(learn_rate_n, input_exp)

                # additional thing: maybe, others should learn + for this input IF other was not 1.0

                # TODO maybe:
                #   there is a certain finite amount of "total match" for which we learn +
                #   i.e.
                #       if winner match was low, more rows learn + for this sample
                #           others unlearn it
                #       if winner match was high, less rows (or only winner) learn for this sample
                #           others unlearn it
                #   this is equivalent to: "activity" for an input is dependent on input match
                # TODO also:
                #   we need to visualize "weight"; the RF are not all equal in confidence/weight
            else:
                learn_rate_n = (0.01 * abs(0.0 - match))[:, np.newaxis]

                self.table_i = np.multiply((1.0 - learn_rate_n), self.table_i) - np.multiply(learn_rate_n, input_exp)

            #self.table_i[argsort_match[1::]] += 0.01

            self.table_i[self.table_i < 0.0] = 0.0
            self.table_i[self.table_i > 1.0] = 1.0

            #self.table_i += 0.001


        self.last_im = input_im.copy()

    def process_input_1(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        # TODO TODO !!!!!!!!! TODO set init_fast back to true in visualizer !!!!!!!

        if self.last_im is not None:

            # get input subset from image
            # get output subset from image

            input_pixels = self.last_im[self.in_r:self.in_r + self.NxN_input, self.in_c:self.in_c + self.NxN_input]
            output_pixels = input_im[self.out_r:self.out_r + self.NxN_output, self.out_c:self.out_c + self.NxN_output]

            # compute pixel bins: copy from file where we already wrote this
            input_arr = input_pixels.flatten()[np.newaxis, :]
            output_arr = output_pixels.flatten()[np.newaxis, :]

            input_exp = self._bin_pixels_expand_columns(arr=input_arr, num_bins_per_pixel=self.bins_per_pixel)
            output_exp = self._bin_pixels_expand_columns(arr=output_arr, num_bins_per_pixel=self.bins_per_pixel)

            assert input_exp.shape[0] == 1
            assert input_exp.shape[1] == self.input_feature_len

            assert output_exp.shape[0] == 1
            assert output_exp.shape[1] == self.output_feature_len

            # compute input matches, threshold? k-WTA? k dynamic based on prediction match threshold? until min remainder of input? every input pixel explained or max?
            # call units "active" in rank order of input match, until:
            #       (option 1) remainder of input unexplained < threshold OR exceed number
            #       (option 2) per-pixel remainder < threshold OR exceed number

            # select which are active

            all_mp = np.multiply(self.table_i, input_exp)
            #print(all_mp.shape, np.amin(all_mp), np.amax(all_mp))
            match = np.divide(np.sum(all_mp, axis=1), np.sum(self.table_i, axis=1))
            argsort_match = np.argsort(match)[::-1]  # largest match first

            unexplained = input_exp.copy()
            active_rows = []

            for k in range(argsort_match.shape[0]):

                row_i = argsort_match[k]
                unexplained = unexplained - self.table_i[row_i, :]
                unexplained[unexplained < 0] = 0
                val = np.sum(unexplained) / np.sum(input_exp)
                #print(k, row_i, val)
                active_rows.append(row_i)
                if val < 0.05:
                    break
            active_rows = np.array(active_rows, np.int)

            # update table_o based on output probabilities and active input rows

            self.table_o[active_rows, :]  = 0.95 * self.table_o[active_rows, :] + 0.05 * output_exp

            # update table_i to evolve towards goal:
            #   for a particular input,
            #   only one row predicted, with "1.0" prediction, each pixel-bin, and all other rows 0.0 for that pixel-bin
            #   i.e. pick which row should win, for each 1.0 value in output_exp

            # TODO realized this doesn't make sense:
            # print(output_exp)
            # nnz_out = np.nonzero(output_exp)[1]  # len 9, which pixel-bins overall were active in training output
            # for each of the pixel-bins above, find which of the active rows in table_o had max value
            # table_o_active = self.table_o[active_rows, :]

            self.table_i[active_rows, :] = 0.95 * self.table_i[active_rows, :] + 0.05 * input_exp

            # match_out = np.multiply(self.table_o[active_rows, :], output_exp)


        self.last_im = input_im.copy()

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

        # weighted mean val for display:
        # tmp1 = np.multiply(tmp, bins[np.newaxis, 0:num_bins_per_pixel])
        # tmp2 = np.sum(tmp1, axis=1)  # 65536
        # tmp3 = np.divide(tmp2, np.sum(tmp, axis=1))
        # max_vals = tmp3

        arr = max_vals.reshape(num_rows, num_pixels)

        return arr

    def get_table_ims(self):
        rows = self._collapse_binned_columns_to_pixels(self.table_i, num_bins_per_pixel=self.bins_per_pixel)

        tile_r_c = int(sqrt(rows.shape[1]))
        N = int(sqrt(self.rows))

        table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

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

                table_im[r0:r1, c0:c1] = tile_im

                tile_n += 1
                c_offset += 1

            r_offset += 1

        # selection image
        select_im = None

        # scale all

        max_dim = max(table_im.shape[0], table_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list = [table_im.copy()]
        ims_names_list = ['table_i']


        rows = self._collapse_binned_columns_to_pixels(self.table_o, num_bins_per_pixel=self.bins_per_pixel)

        tile_r_c = int(sqrt(rows.shape[1]))
        N = int(sqrt(self.rows))

        table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

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

                table_im[r0:r1, c0:c1] = tile_im

                tile_n += 1
                c_offset += 1

            r_offset += 1

        # selection image
        select_im = None

        # scale all

        max_dim = max(table_im.shape[0], table_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        #ims_list.append(table_im.copy())
        #ims_names_list.append('table_o')

        return ims_list, ims_names_list


    def get_table_ims_simple(self):

        ims_list = []
        ims_names_list = []

        input_table_im = self.table_i.copy()

        max_dim = max(input_table_im.shape[0], input_table_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        input_table_im = cv2.resize(input_table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list.append(input_table_im)
        ims_names_list.append('table_i')

        output_table_im = self.table_o.copy()

        max_dim = max(output_table_im.shape[0], output_table_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        output_table_im = cv2.resize(output_table_im, dsize=(0, 0), fx=imscale, fy=imscale,
                                    interpolation=cv2.INTER_NEAREST)

        ims_list.append(output_table_im)
        ims_names_list.append('table_o')

        return ims_list, ims_names_list







    def process_input_OLD_one_pixel_value_position(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        px_r = self.px_r
        px_c = self.px_c
        px_v = self.px_v

        # don't learn after repeat
        if self.t < 32300:
            if px_v[0] <= int(255 * input_im[px_r, px_c]) <= px_v[1]:
                if self.last_im is not None:
                    query_in = self.last_im .flatten()[np.newaxis, :].copy()
                    self._learn_table(query_inputs=query_in)
        else:
            self.otm.print_once('learning stopped!')

        self.t += 1
        self.last_im = input_im.copy()

    def _learn_table(self, query_inputs):

        dists, argmin_dists = self.table.query_multiple_rows(query_inputs=query_inputs)

        dists = dists.flatten()
        argmin_dist = argmin_dists[0]

        if self.init_I_row_num < self.table.get_num_rows():
            self.table.set_matrix_row(row_index=self.init_I_row_num,
                                      row_input=query_inputs[0, :],
                                      row_weights=None,
                                      row_to_table_dists=dists,
                                      fast_init=True)
            self.init_I_row_num += 1
        else:
            if not self.table.post_init_done:
                print()
                print(self.t, 'Table init done! doing post-init...')
                print()

                self.table.post_init()

        if self.table.post_init_done:

            table_min_dist, table_min_dist_r, table_min_dist_c = self.table.get_min_dist()  # table <-> table

            new_min_dist = np.amin(dists)  # row <-> table

            # TODO hack
            if self.t < 10000 and new_min_dist > table_min_dist:
                # print(self.t, 'replacing ', new_min_dist, table_min_dist)
                dists[table_min_dist_r] = np.inf

                # replace the current min dist row, with the new row
                self.table.set_matrix_row(row_index=table_min_dist_r,
                                          row_input=query_inputs[0, :],
                                          row_to_table_dists=dists)
                self.W[table_min_dist_r, :] = 0.5
            else:
                row_index = argmin_dist
                # learn weights
                table_row, _, _ = self.table.get_matrix_row(row_index=row_index)
                tile_input_vect = query_inputs[0, :]

                per_pixel_error = np.abs(tile_input_vect - table_row)
                per_pixel_match = 1.0 - per_pixel_error

                self.win_count[row_index] += 1

                self.W[row_index, :] = 0.8 * self.W[row_index, :] + 0.2 * per_pixel_match

    def save_model(self, models_save_folder):
        '''

        :param models_save_folder:
        :return:
        '''

    def OLD__init__(self, params):

        im_dim = params['image_dim_NxN_pixels']

        self.ims_scale_pixels = 1200

        rows = 9
        self.rows = rows

        self.table = CudaTable(num_entries=rows,
                               input_dim=im_dim * im_dim,
                               disable_row_row_dist=False,
                               enable_weight_bias=False)
        self.init_I_row_num = 0

        self.W = np.ones((rows, im_dim * im_dim), np.float32) * 0.5
        self.otm = OneTimeMessages()

        self.t = 0

        self.count = 0
        self.last_im = None

        self.win_count = np.zeros(rows, np.int)

        self.px_r = 60
        self.px_c = 70
        self.px_v = [10, 40]


    def get_table_ims_OLD(self):

        N = int(sqrt(self.rows))

        rows = self.table.table_i

        # print('***', np.amin(rows), np.amax(rows))
        # print('---', np.amin(cuda_table.table_i), np.amax(cuda_table.table_i))

        tile_r_c = int(sqrt(rows.shape[1]))

        table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5
        weight_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

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
                tile_im[self.px_r, self.px_c] = 0
                tile_im[self.px_r + 1, self.px_c] = 0
                tile_im[self.px_r - 1, self.px_c] = 0
                tile_im[self.px_r, self.px_c + 1] = 0
                tile_im[self.px_r, self.px_c - 1] = 0

                table_im[r0:r1, c0:c1] = tile_im

                tile_weights = self.W[tile_n, :].reshape((tile_r_c, tile_r_c))
                tile_weights[self.px_r, self.px_c] = 0
                tile_weights[self.px_r + 1, self.px_c] = 0
                tile_weights[self.px_r - 1, self.px_c] = 0
                tile_weights[self.px_r, self.px_c + 1] = 0
                tile_weights[self.px_r, self.px_c - 1] = 0

                weight_im[r0:r1, c0:c1] = tile_weights

                tile_n += 1
                c_offset += 1

            r_offset += 1

        #max_dim = max(table_im.shape[0], table_im.shape[1])
        #imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        #table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list = [table_im, weight_im]
        ims_names_list = ['table', 'weights']

        print(self.win_count)

        return ims_list, ims_names_list
