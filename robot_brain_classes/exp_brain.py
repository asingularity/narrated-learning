


import time
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt

from cuda_dist_query import CudaTable
from utils.one_time_messages import OneTimeMessages


from brain_components_classes.perceptron import Perceptron


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import log




from utils.one_time_messages import OneTimeMessages



class ExpBrain(object):
    '''
    kWTA, where k is adaptive and based on remainder
    activations and learning are based on remainder
    '''

    def __init__(self, params):
        #self.im_dim = params['image_dim_NxN_pixels']
        self.use_full_input = params['use_full_input']

        self.im_dim = 1500
        #self.ims_scale_pixels = self.im_dim

        self.otm = OneTimeMessages()

        self.t = 0

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

        self.in_r = 50  # 40 # 65 #50 # 60
        self.in_c = 64  # 84 # 76 #64 # 64

        self.rf_dim = 16

        self.bins_per_pixel = 6

        # self.num_rf = 80
        # self.error_threshold = 50 / 8.

        self.num_rf = 40  # per layer
        self.layer_start_times = np.array([0, 1, 2, 3]) * 50000
        #self.layer_start_times = np.array([0, 1]) * 50000
        self.num_layers = self.layer_start_times.shape[0]

        # no more error threshold
        # self.error_threshold = 50  # 50  # 12.5 / 2

        assert self.in_r + self.rf_dim < self.im_dim, str((self.in_r + self.rf_dim, self.im_dim))
        assert self.in_c + self.rf_dim < self.im_dim, str((self.in_c + self.rf_dim, self.im_dim))

        self.input_feature_len = self.rf_dim * self.rf_dim * self.bins_per_pixel

        self.probs = []
        for k in range(self.num_layers):
            self.probs.append(np.zeros((self.num_rf, self.input_feature_len)))

        self.lr = 0.001

        max_time = 100000000

        self.mean_fr = 10000
        self.rf_counts = np.zeros(max_time)
        self.mean_rf_counts = np.zeros(max_time)
        self.rec_error = np.zeros(max_time)
        self.mean_rec_error = np.zeros(max_time)

        self.im_error = np.zeros(max_time)
        self.mean_im_error = np.zeros(max_time)

        self.last_input_im = None
        self.last_reconstruction_im = None
        self.last_remainder_im = None

        # plot error
        self.last_plot_time = time.time()
        self.plot_interval = 30

        self.last_info = None
        self.last_eff_frames = np.zeros(self.num_rf)

    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        # TODO make this a param; this is for the full-frame input 16x16 that matches RFs size

        # if self.t == 40000:
        #     print()
        #     print('REDUCING ERROR THRESHOLD')
        #     print()
        #
        #     self.error_threshold = self.error_threshold / 2.0

        if self.use_full_input:
            input_pixels_1 = input_im
        else:
            input_pixels_1 = input_im[self.in_r:self.in_r + self.rf_dim, self.in_c:self.in_c + self.rf_dim]

        input_arr_1 = input_pixels_1.flatten()[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

        self.last_input_im = input_pixels_1.copy()

        error = np.sum(input_exp_1)

        remainder = input_exp_1.copy()
        reconstruction = np.zeros((1, self.input_feature_len))
        rfs_active = np.zeros(self.num_rf)

        self.last_eff_frames[:] = np.nan

        loop = 0
        #while (error > error_threshold) and np.count_nonzero(rfs_active) < self.num_rf:
        for layer_n in range(self.num_layers):
            if self.t >= self.layer_start_times[layer_n]:
                # do WTA over all RFs that have no event yet this frame
                #   use prob, applied on remainder
                #   include subtraction that we do to count EV

                #print(np.amin(self.probs), np.amax(self.probs), np.amin(remainder), np.amax(remainder))

                mp_p = np.multiply(self.probs[layer_n], remainder)
                mp_n = np.multiply(self.probs[layer_n], 1.0 - remainder)

                eff_frame_p = np.sum(mp_p, axis=1)
                eff_frame_n = np.sum(mp_n, axis=1)
                eff_frame = eff_frame_p - eff_frame_n

                inactive_rfs = np.nonzero(rfs_active==0)[0]
                win_rf_index = inactive_rfs[np.argmax(eff_frame[inactive_rfs])]

                self.last_eff_frames[win_rf_index] = eff_frame[win_rf_index]

                #print(win_rf_index)

                # winner learns (updates prob) on remainder
                #   alternative: learns on whole image, not remainder
                self.probs[layer_n][win_rf_index, :] = (1.0 - self.lr) * self.probs[layer_n][win_rf_index, :] + self.lr * remainder[0, :]
                #self.probs[win_rf_index, :] = (1.0 - self.lr) * self.probs[win_rf_index, :] + self.lr * input_exp_1[0, :]

                # recalculate new remainder
                # TODO this is the part that is problematic:
                remainder = remainder - mp_p[win_rf_index, :]  # results in strange k-WTA, but suboptimal, and requiring higher error threshold

                #remainder = remainder - (mp_p[win_rf_index, :] > 0)  # results in single-WTA; doesnt work for bouncing balls at all (single rf gray)
                #remainder = remainder - self.probs[win_rf_index, :]  # k-wta also, same as mp_p one
                #remainder = remainder - (self.probs[win_rf_index, :] > 0)  # also results in single-WTA


                #remainder = np.minimum(remainder, match[win_rf_index, :])

                remainder[remainder < 0] = 0

                # reconstruction
                #reconstruction = reconstruction + self.probs[win_rf_index, :]
                reconstruction = np.maximum(reconstruction, self.probs[layer_n][win_rf_index, :])

                # set error for next loop
                # error = np.sum(remainder)

                #print(error)

                # update rfs_active
                rfs_active[win_rf_index] = 1

                # if loop == 0:
                #     self.last_remainder_im = remainder.copy()
                loop += 1

        self.last_remainder_im = remainder.copy()

        reconstruction[reconstruction > 1] = 1
        self.last_reconstruction_im = reconstruction.copy()

        # None of these rec_error values converge / reduce:

        #rec_error = np.sum(np.abs(reconstruction - input_exp_1))

        #rec_im, rec_w = self._collapse_binned_columns_to_pixels(arr_exp=self.last_reconstruction_im, num_bins_per_pixel=self.bins_per_pixel)
        #im0 = rec_im[0, :].reshape((self.rf_dim, self.rf_dim))
        #rec_error = np.sum(np.abs(im0 - input_pixels_1))

        rec_error = np.sum(remainder)

        self.rec_error[self.t] = rec_error
        self.mean_rec_error[self.t] = np.mean(self.rec_error[max(0, self.t - self.mean_fr):self.t])

        self.rf_counts[self.t] = np.count_nonzero(rfs_active)
        self.mean_rf_counts[self.t] = np.mean(self.rf_counts[max(0, self.t - self.mean_fr):self.t])
        self.t += 1

    def get_table_ims(self):

        ims_list = []
        ims_names_list = []

        # print (self.last_info)
        #print('mean rf active: ', np.mean(self.rf_counts[max(0, self.t - 200):self.t]))
        # print(self.last_eff_frames)

        if time.time() > self.last_plot_time + self.plot_interval:
            print()
            print('Making plot of state vars...')
            self.ax.cla()
            self.ax.plot(self.rec_error[0:self.t], color='r')
            self.ax.plot(self.mean_rec_error[0:self.t], color='b')
            self.fig.savefig("rec_error.png", dpi=100)

            self.ax.cla()
            self.ax.plot(self.rf_counts[0:self.t], color='r')
            self.ax.plot(self.mean_rf_counts[0:self.t], color='b')
            self.fig.savefig("rf_counts.png", dpi=100)

            self.last_plot_time = time.time()

        tmp_all_im = None
        for layer_n in range(self.num_layers):
            tmp_im = None
            arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.probs[layer_n], num_bins_per_pixel=self.bins_per_pixel)

            for r_tmp in range(weights.shape[0]):
                #weights[r_tmp, :] = (weights[r_tmp, :] - np.amin(weights[r_tmp, :])) * 1.0 / (np.amax(weights[r_tmp, :]) - np.amin(weights[r_tmp, :]))

                im0 = arr[r_tmp, :].reshape((self.rf_dim, self.rf_dim))
                im1 = weights[r_tmp, :].reshape((self.rf_dim, self.rf_dim))

                tmp2 = np.hstack((im0, im1))

                if tmp_im is None:
                    tmp_im = tmp2.copy()
                else:
                    tmp3 = np.zeros((2, tmp_im.shape[1]))
                    tmp_im = np.vstack((tmp_im, tmp3, tmp2))

            if tmp_all_im is None:
                tmp_all_im = tmp_im.copy()
            else:
                tmp4 = np.zeros((tmp_im.shape[0], 2))
                tmp_all_im = np.hstack((tmp_all_im, tmp4, tmp_im))

        max_dim = max(tmp_all_im.shape[0], tmp_all_im.shape[1])
        imscale = self.im_dim / max_dim  # 0.2: full table, 2.0
        tmp_im = cv2.resize(tmp_all_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list.append(tmp_im)
        ims_names_list.append('probs_v, probs_w')

        rec_im, rec_w = self._collapse_binned_columns_to_pixels(arr_exp=self.last_reconstruction_im, num_bins_per_pixel=self.bins_per_pixel)
        im0 = rec_im[0, :].reshape((self.rf_dim, self.rf_dim))
        im1 = rec_w[0, :].reshape((self.rf_dim, self.rf_dim))

        rem_im, rem_w = self._collapse_binned_columns_to_pixels(arr_exp=self.last_remainder_im, num_bins_per_pixel=self.bins_per_pixel)
        imr0 = rem_im[0, :].reshape((self.rf_dim, self.rf_dim))
        imr1 = rem_w[0, :].reshape((self.rf_dim, self.rf_dim))

        ims_list.append(np.hstack((self.last_input_im,
                                   np.zeros((self.rf_dim, 2)), im0, np.zeros((self.rf_dim, 2)), im1,
                                   np.zeros((self.rf_dim, 2)), imr0, np.zeros((self.rf_dim, 2)), imr1,)))
        ims_names_list.append('input, reconstruct, remainder')

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
        arr_exp = np.zeros((arr.shape[0], arr.shape[1] * num_bins_per_pixel), np.float32)
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


class ExpBrainMultiWeightedPartialWithReconstruction(object):
    def __init__(self, params):
        self.im_dim = params['image_dim_NxN_pixels']

        self.ims_scale_pixels = 300

        self.otm = OneTimeMessages()

        self.t = 0

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

        self.in_r = 50  # 40 # 65 #50 # 60
        self.in_c = 64  # 84 # 76 #64 # 64

        self.rf_dim = 16

        self.bins_per_pixel = 6

        self.num_rf = 6
        self.im_dim = 500

        assert self.in_r + self.rf_dim < self.im_dim, str((self.in_r + self.rf_dim, self.im_dim))
        assert self.in_c + self.rf_dim < self.im_dim, str((self.in_c + self.rf_dim, self.im_dim))

        self.input_feature_len = self.rf_dim * self.rf_dim * self.bins_per_pixel

        self.rfs = np.random.random((self.num_rf, self.input_feature_len))
        #self.rfs = np.zeros((self.num_rf, self.input_feature_len)) + 1e-3

        self.probs = np.zeros((self.num_rf, self.input_feature_len))

        self.total_events_disp = np.zeros(self.num_rf)
        self.total_time_disp = 0

        self.mean_effs = np.zeros(self.num_rf)

        self.last_match_ims = []
        self.last_remainder_ims = []
        for k in range(self.num_rf):
            self.last_match_ims.append(None)
            self.last_remainder_ims.append(None)

        lr_factor = 5.0

        self.prob_lr = 0.0001 * lr_factor
        self.mean_eff_lr = 0.00001 * lr_factor
        self.weights_lr = 0.0001 * lr_factor * 1  # 0.00001
        #self.inhib_weights_lr = 1 * self.weights_lr
        #self.prob_corr_lr = 0.0001 * lr_factor * 100
        #self.prob_corr_lr = 0.7

        self.gain_lr = 0.01  #0.00001 * lr_factor

        self.threshold = 127.0

        # competition
        self.prob_corr = np.zeros((self.num_rf, self.num_rf))
        self.inhib_w = np.zeros((self.num_rf, self.num_rf))  # from, to
        self.gains = np.ones(self.num_rf)
        self.last_event_time = np.zeros(self.num_rf)

        # reconstruction
        self.last_input_im = None
        self.last_reconstruction_im = None

    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''


        input_pixels_1 = input_im[self.in_r:self.in_r + self.rf_dim, self.in_c:self.in_c + self.rf_dim]
        input_arr_1 = input_pixels_1.flatten()[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

        event_rfs = []
        self.rf_inputs = []

        if 1:
            remainder = input_exp_1

            self.last_input_im = input_pixels_1.copy()

            for rf_index in range(self.num_rf):

                if rf_index == 0 or (rf_index == 1 and self.t > 50000) or (rf_index == 2 and self.t > 100000) or (rf_index == 3 and self.t > 150000) or (rf_index == 4 and self.t > 200000) or (rf_index == 5 and self.t > 250000):
                    self.rf_inputs.append(remainder.copy())
                    s_mp = np.multiply(self.rfs[rf_index, :], remainder)
                    s_mp_sum = np.sum(s_mp)
                    match = s_mp_sum   # * self.gains[rf_index]

                    if match >= self.threshold:
                        event_rfs.append(rf_index)
                        remainder = remainder - np.multiply(self.probs[rf_index, :], input_exp_1)

                    self.last_remainder_ims[rf_index] = remainder.copy()

        else:
            all_mp = np.multiply(self.rfs, input_exp_1)
            mp_sum = np.sum(all_mp, axis=1)
            match = np.multiply(mp_sum, self.gains)
            event_rfs = np.nonzero(match >= self.threshold)[0]

        if 0:

            all_mp = np.multiply(self.rfs, input_exp_1)
            mp_sum = np.sum(all_mp, axis=1)
            match = np.multiply(mp_sum, self.gains)
            argsort_match = np.argsort(match)

            input_tmp = input_exp_1.copy()

            for k in range(argsort_match.shape[0]):
                rf_index = argsort_match[k]

                s_mp = np.multiply(self.rfs[rf_index, :], input_tmp)
                s_mp_sum = np.sum(s_mp)
                match = s_mp_sum * self.gains[rf_index]

                if match >= self.threshold:
                    event_rfs.append(rf_index)
                    input_tmp -= s_mp
                    input_tmp[input_tmp < 0] = 0

        elif 0:
            all_mp = np.multiply(self.rfs, input_exp_1)

            assert all_mp.shape[0] == self.num_rf
            assert all_mp.shape[1] == self.input_feature_len

            mp_sum = np.sum(all_mp, axis=1)
            assert mp_sum.shape[0] == self.num_rf

            #match = np.divide(mp_sum, np.sum(self.rfs, axis=1))
            #match = mp_sum
            match = np.multiply(mp_sum, self.gains)

            assert match.shape[0] == self.num_rf

            argsort_match = np.argsort(match)
            # print('********')
            # print(np.amax(self.prob_corr), np.amin(self.prob_corr))
            # print('***')

            event_rfs_before_inhibit = len(np.nonzero(match >= self.threshold)[0])

            for k in range(argsort_match.shape[0]):
                rf_index = argsort_match[k]

                inhib = 0.0
                self.inhib_w[:, rf_index] -= self.inhib_weights_lr
                for k2 in range(k):
                    rf_index_pre = argsort_match[k2]
                    inhib += self.inhib_w[rf_index_pre, rf_index]
                    self.inhib_w[rf_index_pre, rf_index] += (self.inhib_weights_lr * 2)

                match -= inhib

            # compute effectiveness for each rf that had an event:
            event_rfs = np.nonzero(match >= self.threshold)[0]
            no_event_rfs = np.nonzero(match < self.threshold)[0]

        event_rfs = np.array(event_rfs, np.int)
        # if len(event_rfs) > 0:
        #     print('(((((((')
        #     print(self.inhib_w)
        #     print(self.t, '------', event_rfs_before_inhibit, '->', len(event_rfs))

        assert event_rfs.shape[0] <= self.num_rf

        self.last_event_time[event_rfs] = self.t
        self.gains[np.nonzero(self.t - self.last_event_time > 100)] *= (1.0 + self.gain_lr)
        self.gains[event_rfs] = 1.0
        self.gains[self.gains > 1000] = 1000

        eff_frame_p = np.sum(np.multiply(self.probs[event_rfs], input_exp_1), axis=1)
        eff_frame_n = np.sum(np.multiply(self.probs[event_rfs], input_exp_1 == 0), axis=1)

        eff_frame = eff_frame_p - eff_frame_n
        eff_frame_all = np.zeros(self.num_rf)
        eff_frame_all[event_rfs] = eff_frame[:]

        assert eff_frame.shape[0] == event_rfs.shape[0]

        argsort_event_rf = np.argsort(eff_frame)[::-1]
        sorted_event_rfs = event_rfs[argsort_event_rf]

        lr = self.weights_lr

        input_tmp = input_exp_1.copy()
        nnz_input = np.nonzero(input_tmp)[1]
        nz_input = np.nonzero(input_tmp == 0)[1]

        # tmp_p = np.multiply(self.probs, input_exp_1)

        reconstruction = np.zeros(self.input_feature_len)

        for event_rf in sorted_event_rfs:
            if event_rf == 0 or (event_rf == 1 and self.t > 50000) or (event_rf == 2 and self.t > 100000) or (rf_index == 3 and self.t > 150000) or (rf_index == 4 and self.t > 200000) or (rf_index == 5 and self.t > 250000):
                rf_input = self.rf_inputs[event_rf]

                nnz_rf_input = np.nonzero(rf_input)[1]
                nz_rf_input = np.nonzero(rf_input==0)[1]

                self.rfs[event_rf, nnz_input] = self.rfs[event_rf, nnz_rf_input] + lr * self.gains[event_rf]
                self.rfs[event_rf, nz_input] = self.rfs[event_rf, nz_rf_input] - lr * self.gains[event_rf]

                reconstruction = np.maximum(reconstruction, self.probs[event_rf, :])


        self.last_reconstruction_im = reconstruction

        # for rf_index in range(self.num_rf):
        #     if rf_index in event_rfs:
        #         self.prob_corr[rf_index, event_rfs] = (1.0 - self.prob_corr_lr) * self.prob_corr[event_rf, event_rfs] + self.prob_corr_lr * 1.0
        #         self.prob_corr[rf_index, no_event_rfs] = (1.0 - self.prob_corr_lr) * self.prob_corr[event_rf, no_event_rfs] + self.prob_corr_lr * 0.0
        #     else:
        #         pass
        #
        # self.prob_corr[range(self.num_rf), range(self.num_rf)] = 0.0

        self.rfs[np.nonzero(self.rfs < 0)] = 0
        self.rfs[np.nonzero(self.rfs > self.threshold)] = self.threshold

        self.probs[event_rfs, :] = self.probs[event_rfs, :] * (1.0 - self.prob_lr) + self.prob_lr * input_exp_1

        self.total_events_disp[event_rfs] += 1
        self.total_time_disp += 1

        #if len(event_rfs) > 0:
        #    print(event_rfs)
        for rf_index in event_rfs:
            self.last_match_ims[rf_index] = input_pixels_1.copy()

        self.mean_effs = (1.0 - self.mean_eff_lr) * self.mean_effs + self.mean_eff_lr * eff_frame_all

        self.t += 1

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
        arr_exp = np.zeros((arr.shape[0], arr.shape[1] * num_bins_per_pixel), np.float32)
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

    def get_table_ims(self):
        print('mean_effs:', self.mean_effs)
        print(np.amax(self.prob_corr), np.amin(self.prob_corr))
        print(self.gains)
        #print('prob corr:')
        #print(self.prob_corr)

        ims_list = []
        ims_names_list = []

        if self.last_match_ims[0] is not None:
            arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.rfs, num_bins_per_pixel=self.bins_per_pixel)

            tmp_im = None

            for r_tmp in range(weights.shape[0]):
                weights[r_tmp, :] = (weights[r_tmp, :] - np.amin(weights[r_tmp, :])) * 1.0 / (np.amax(weights[r_tmp, :]) - np.amin(weights[r_tmp, :]))

                im0 = arr[r_tmp, :].reshape((self.rf_dim, self.rf_dim))
                im1 = weights[r_tmp, :].reshape((self.rf_dim, self.rf_dim))

                if self.last_match_ims[r_tmp] is None:
                    im2 = np.zeros((self.rf_dim, self.rf_dim)) + 0.5
                else:
                    im2 = self.last_match_ims[r_tmp]

                tmp2 = np.hstack((im0, im1, im2))
                if tmp_im is None:
                    tmp_im = tmp2.copy()
                else:
                    tmp3 = np.zeros((2, tmp_im.shape[1]))
                    tmp_im = np.vstack((tmp_im, tmp3, tmp2))

            max_dim = max(tmp_im.shape[0], tmp_im.shape[1])
            imscale = self.im_dim / max_dim  # 0.2: full table, 2.0
            tmp_im = cv2.resize(tmp_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

            ims_list.append(tmp_im)
            ims_names_list.append('rf_im')


            arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.probs, num_bins_per_pixel=self.bins_per_pixel)

            tmp_im = None

            for r_tmp in range(weights.shape[0]):
                weights[r_tmp, :] = (weights[r_tmp, :] - np.amin(weights[r_tmp, :])) * 1.0 / (np.amax(weights[r_tmp, :]) - np.amin(weights[r_tmp, :]))

                im0 = arr[r_tmp, :].reshape((self.rf_dim, self.rf_dim))
                im1 = weights[r_tmp, :].reshape((self.rf_dim, self.rf_dim))

                if self.last_match_ims[r_tmp] is None:
                    im2 = np.zeros((self.rf_dim, self.rf_dim)) + 0.5
                    im4 = im2.copy()
                    im5 = im5.copy()
                else:
                    im2 = self.last_match_ims[r_tmp]

                    arr4, weights4 = self._collapse_binned_columns_to_pixels(arr_exp=self.last_remainder_ims[r_tmp], num_bins_per_pixel=self.bins_per_pixel)

                    im4 = arr4.reshape((self.rf_dim, self.rf_dim))
                    im5 = weights4.reshape((self.rf_dim, self.rf_dim))

                tmp2 = np.hstack((im0, im1, im2, im4, im5))
                if tmp_im is None:
                    tmp_im = tmp2.copy()
                else:
                    tmp3 = np.zeros((2, tmp_im.shape[1]))
                    tmp_im = np.vstack((tmp_im, tmp3, tmp2))

            max_dim = max(tmp_im.shape[0], tmp_im.shape[1])
            imscale = self.im_dim / max_dim  # 0.2: full table, 2.0
            tmp_im = cv2.resize(tmp_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

            ims_list.append(tmp_im)
            ims_names_list.append('probs_v, probs_w, last_match, rem_v, rem_w')


            arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.last_reconstruction_im[np.newaxis, :], num_bins_per_pixel=self.bins_per_pixel)
            imr0 = arr[0, :].reshape((self.rf_dim, self.rf_dim))
            imr1 = weights[0, :].reshape((self.rf_dim, self.rf_dim))

            tmp_imr = np.hstack((self.last_input_im, imr0, imr1))

            max_dim = max(tmp_imr.shape[0], tmp_imr.shape[1])
            imscale = self.im_dim / max_dim  # 0.2: full table, 2.0
            tmp_imr = cv2.resize(tmp_imr, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

            ims_list.append(tmp_imr)
            ims_names_list.append('reconstruction')

        return ims_list, ims_names_list

class ExpBrainSingleWeightedPartial(object):
#class ExpBrain(object):
    def __init__(self, params):
        self.im_dim = params['image_dim_NxN_pixels']

        self.ims_scale_pixels = 300

        self.otm = OneTimeMessages()

        self.t = 0

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

        self.in_r = 50  # 40 # 65 #50 # 60
        self.in_c = 64  # 84 # 76 #64 # 64

        self.rf_dim = 16

        self.bins_per_pixel = 6

        self.rf = np.random.random((self.rf_dim, self.rf_dim, self.bins_per_pixel))

        self.prob = np.zeros((self.rf_dim, self.rf_dim, self.bins_per_pixel))
        self.mean_eff = 0.0

        self.prob_lr = 0.0001
        self.mean_eff_lr = 0.00001

        self.weights_lr = 0.0001 # 0.00001

        self.threshold = 0.165

        self.last_match_im = None

        self.total_events_disp = 0
        self.total_time_disp = 0


    def process_input(self, input_im):
        input_pixels_1 = input_im[self.in_r:self.in_r + self.rf_dim, self.in_c:self.in_c + self.rf_dim]
        input_arr_1 = input_pixels_1.flatten()[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

        all_mp = np.multiply(self.rf.flatten()[np.newaxis, :], input_exp_1)
        mp_sum = np.sum(all_mp)
        #print(mp_sum)

        match = mp_sum / np.sum(self.rf)

        if match >= self.threshold:
            #print('Event!', self.t)
            self.last_match_im = input_im[self.in_r:self.in_r + self.rf_dim, self.in_c:self.in_c + self.rf_dim].copy()
            tmp = input_exp_1.flatten().reshape((self.rf_dim, self.rf_dim, self.bins_per_pixel))
            self.prob = self.prob * (1.0 - self.prob_lr) + self.prob_lr * tmp

            eff_frame_p = np.sum(np.multiply(self.prob.flatten()[np.newaxis, :], input_exp_1))
            eff_frame_n = np.sum(np.multiply(self.prob.flatten()[np.newaxis, :], input_exp_1==0))

            eff_frame = eff_frame_p - eff_frame_n

            # TODO stabilize this rule
            self.rf[np.nonzero(tmp)] += self.weights_lr
            self.rf[np.nonzero(tmp == 0)] -= self.weights_lr

            # if this isn't here, weights can go negative... very bizarre results
            self.rf[np.nonzero(self.rf < 0)] = 0

            # TODO max_threshold, if later we change the threshold
            # self.rf[np.nonzero(self.rf > self.threshold)] = self.threshold

            self.total_events_disp += 1

        else:
            # no match

            eff_frame = 0.0

        self.mean_eff = (1.0 - self.mean_eff_lr) * self.mean_eff + self.mean_eff_lr * eff_frame

        # TODO now thing we want to know is: would increase in prob for any given pixel-bin, help mean_eff or not?

        self.t += 1
        self.total_time_disp += 1

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
        arr_exp = np.zeros((arr.shape[0], arr.shape[1] * num_bins_per_pixel), np.float32)
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

    def get_table_ims(self):
        # print(self.t)
        # print((np.amin(self.prob), np.amax(self.prob)))
        # print(np.nonzero(self.prob==0))

        print('t', self.t, 'mean_eff:', self.mean_eff, 'prop frames with event:', self.total_events_disp / self.total_time_disp)

        ims_list = []
        ims_names_list = []

        if self.last_match_im is not None:
            arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.rf.flatten()[np.newaxis, :], num_bins_per_pixel=self.bins_per_pixel)

            # normalize weights to [0, 1]
            print('    weights:', np.amin(weights), np.amax(weights))

            weights = (weights - np.amin(weights)) * 1.0 / (np.amax(weights) - np.amin(weights))

            im0 = arr.reshape((self.rf_dim, self.rf_dim))
            im1 = weights.reshape((self.rf_dim, self.rf_dim))
            im2 = self.last_match_im
            #cv2.imshow('asd', np.hstack((im0, im1)))
            #cv2.waitKey(1)

            ims_list.append(np.hstack((im0, im1, im2)))
            ims_names_list.append('rf_im')

            arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.prob.flatten()[np.newaxis, :], num_bins_per_pixel=self.bins_per_pixel)

            im0 = arr.reshape((self.rf_dim, self.rf_dim))
            im1 = weights.reshape((self.rf_dim, self.rf_dim))
            im2 = self.last_match_im
            #cv2.imshow('asd', np.hstack((im0, im1)))
            #cv2.waitKey(1)

            ims_list.append(np.hstack((im0, im1, im2)))
            ims_names_list.append('prob_im')

            # input image with box
            in_region_im = self.last_match_im.copy()
            cv2.rectangle(in_region_im, (self.in_c, self.in_r), (self.in_c + self.rf_dim, self.in_r + self.rf_dim), 255, 2)

            ims_list.append(in_region_im.copy())
            ims_names_list.append('input_region')

        return ims_list, ims_names_list




class ExpBrainMultiPartialPatternUnstable(object):
    '''
    new greedy algorithm for growing/contracting RF over time

    multi-unit rule
    '''

    def __init__(self, params):
        '''

        :param params:
        '''

        random.seed(1124)
        np.random.seed(1312)

        self.im_dim = params['image_dim_NxN_pixels']

        self.ims_scale_pixels = 1400

        self.otm = OneTimeMessages()

        self.t = 0

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

        # top left of input region

        self.in_r = 50  # 40 # 65 #50 # 60
        self.in_c = 64  # 84 # 76 #64 # 64

        # parameters

        self.num_rf = 32
        self.rf_dim = 16
        self.bins_per_pixel = 6
        self.input_history_steps = 1
        self.tau = 0.001  # 0.001 # 0.0001
        self.wait_learn_until = 4 * (log(2) / self.tau)

        self.plot_graphs = False

        print('start learning at: ', self.wait_learn_until)

        assert self.in_r + self.rf_dim < self.im_dim, str((self.in_r + self.rf_dim, self.im_dim))
        assert self.in_c + self.rf_dim < self.im_dim, str((self.in_c + self.rf_dim, self.im_dim))

        # arrays, per rf

        self.rf_list = []
        self.add_candidates_list = []
        self.remove_candidates_list = []

        # numbers, per rf

        self.rf_px_bins_exp = []

        # init per rf

        # TODO should be smarter than this
        self.init_done = []

        for k in range(self.num_rf):
            rf_px_bins_exp = 0.0
            rf = np.zeros((self.rf_dim, self.rf_dim, self.bins_per_pixel))
            add_candidates = np.zeros((self.rf_dim, self.rf_dim, self.bins_per_pixel))
            remove_candidates = np.zeros((self.rf_dim, self.rf_dim, self.bins_per_pixel))

            self.rf_px_bins_exp.append(rf_px_bins_exp)
            self.rf_list.append(rf)
            self.add_candidates_list.append(add_candidates)
            self.remove_candidates_list.append(remove_candidates)

            self.init_done.append(False)

        # other

        self.rf_ages = np.zeros(self.num_rf)
        self.rf_num_bad = np.zeros(self.num_rf)

        self.last_im = None
        self.last_match_im = None
        self.whats_left_im = None
        self.last_input_sub_im = None

        self.t = 0
        self.MAX_TIME = 1000000
        self.im_match_prop_history = np.zeros(self.MAX_TIME)
        self.mean_im_match_prop_history = np.zeros(self.MAX_TIME)  # time-averaged

        self.last_print_time = 0


        # TODO track here: a *new* hypothetical RF, made of any one of the pixel-bins, how effective would it be?
        # TODO - computed on "whats left" after all RFs detect and subtract from image

        self.single_pixel_bin_RF_exp = np.zeros((self.rf_dim, self.rf_dim, self.bins_per_pixel))

        # TODO questions:
        # TODO how RF initialized, if we dont choose an initial pixel-bin? should just work with zero-rf as a start
        #       for now we did hack: random choice of an input pixel-bin
        # TODO where is shared array of what others have already explained
        # TODO need to verify that will also converge to less-rare features

    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        self.last_im = input_im.copy()

        input_pixels_1 = input_im[self.in_r:self.in_r + self.rf_dim, self.in_c:self.in_c + self.rf_dim]

        self.last_input_sub_im = input_pixels_1.copy()

        input_arr_1 = input_pixels_1.flatten()[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

        input_exp_1_orig = input_exp_1.copy()
        input_exp_1_tmp = input_exp_1.copy()

        count_nnz_org = np.count_nonzero(input_exp_1)
        DO_COMPETITION = 1

        # if self.t < 100000:
        #     range_end = int(self.num_rf / 2)
        # else:
        range_end = self.num_rf

        for rf_ind in range(range_end):
            if self.init_done[rf_ind]:
                if DO_COMPETITION:
                    input_exp_1 = self._process_rf(rf_index=rf_ind, input_exp_1=input_exp_1)
                else:
                    # (Done?) fix this; logic is wrong; should be able to measure total explained without the logic to remove at every step

                    input_exp_1 = self._process_rf(rf_index=rf_ind, input_exp_1=input_exp_1)
                    input_exp_1_tmp = np.minimum(input_exp_1_tmp, input_exp_1)
                    input_exp_1[:, :] = input_exp_1_orig[:, :]

            else:
                if random.random() < 0.01:
                    nnz_input = np.nonzero(input_exp_1)[1]
                    rand_nnz = np.random.choice(nnz_input)
                    rand_inds = np.unravel_index(rand_nnz, self.rf_list[rf_ind].shape)
                    self.rf_list[rf_ind][rand_inds] = 1.0
                    self.init_done[rf_ind] = True

        if DO_COMPETITION:
            count_nnz_left = np.count_nonzero(input_exp_1)
            # arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.rf_list[rf_index].flatten()[np.newaxis, :], num_bins_per_pixel=self.bins_per_pixel)

            _, whats_left = self._collapse_binned_columns_to_pixels(arr_exp=input_exp_1, num_bins_per_pixel=self.bins_per_pixel)

            tmp = input_exp_1.flatten().reshape((self.rf_dim, self.rf_dim, self.bins_per_pixel))
            nnz_tmp = np.nonzero(tmp)
            nz_tmp = np.nonzero(tmp==0)
            self.single_pixel_bin_RF_exp[nnz_tmp] = (1.0 - self.tau) * self.single_pixel_bin_RF_exp[nnz_tmp] + self.tau * 1.0
            self.single_pixel_bin_RF_exp[nz_tmp] = (1.0 - self.tau) * self.single_pixel_bin_RF_exp[nz_tmp] + self.tau * 0.0
        else:
            count_nnz_left = np.count_nonzero(input_exp_1_tmp)
            _, whats_left = self._collapse_binned_columns_to_pixels(arr_exp=input_exp_1_tmp, num_bins_per_pixel=self.bins_per_pixel)

            # TODO update single_pixel_bin_RF_exp here


        self.whats_left_im = whats_left.reshape((self.rf_dim, self.rf_dim))

        self.im_match_prop_history[self.t] = (count_nnz_org - count_nnz_left) * 1.0 / count_nnz_org
        self.mean_im_match_prop_history[self.t] = np.mean(self.im_match_prop_history[max(0, self.t - 4000):self.t])

        self.t += 1

    def _process_rf(self, rf_index, input_exp_1):

        rf = self.rf_list[rf_index]
        rf_px_bins_exp = self.rf_px_bins_exp[rf_index]
        add_candidates = self.add_candidates_list[rf_index]
        remove_candidates = self.remove_candidates_list[rf_index]

        # nonzero elements of RF:
        nnz_rf_flat = np.nonzero(rf.flatten())  # verified: flatten expands the same as _bin_pixels_expand_columns

        # input values for nonzero elements of RF:
        tmp = input_exp_1[0, nnz_rf_flat[0]]

        rf_size = np.size(tmp)
        input_match_num = np.count_nonzero(tmp)
        assert input_match_num <= rf_size

        if input_match_num == rf_size:
            match = True
        else:
            match = False

        tau = self.tau
        nnz_rf = np.nonzero(rf)
        nz_rf = np.nonzero(rf==0)

        if match:

            # self.count_matches += 1
            # print('match: ', self.t)
            # self.last_match_im = input_im.copy()

            rf_px_bins_exp = (1.0 - tau) * rf_px_bins_exp + tau * rf_size

            # hypothetical additions to expand to:

            # all pixel-bins activity values (1 or 0) on this frame:
            tmp = input_exp_1.flatten().reshape((self.rf_dim, self.rf_dim, self.bins_per_pixel))
            nnz_tmp = np.nonzero(tmp)
            add_candidates[nnz_tmp] = (1.0 - tau) * add_candidates[nnz_tmp] + tau * (rf_size + 1)

            z_tmp = np.nonzero(tmp == 0)
            add_candidates[z_tmp] = (1.0 - tau) * add_candidates[z_tmp] + tau * 0

            # modify input_exp_1 to remove what matched
            input_exp_1[0, nnz_rf_flat] = 0

            # assert that only one value is 1 per pixel-bin, in input and in the RF (sanity check)
            do_debug = 0
            if do_debug:
                tmp_sum = np.sum(tmp, axis=2)
                assert tmp_sum.size == np.sum(tmp_sum==1)
                tmp_sum2 = np.sum(rf, axis=2)

                assert np.amax(tmp_sum2) <= 1

            # there is a current RF match: update self.remove_hyp_px_bin_exp_per_frame:
            #   for all pixels within RF:
            #       update towards hypothetical smaller RF value, since all sub-patterns are present
            remove_candidates[nnz_rf] = (1.0 - tau) * remove_candidates[nnz_rf] + tau * (rf_size - 1)

        else:
            rf_px_bins_exp = (1.0 - tau) * rf_px_bins_exp + tau * 0.0

            add_candidates = (1.0 - tau) * add_candidates + tau * 0.0

            # TODO
            # no match: update self.remove_hyp_px_bin_exp_per_frame:
            #   for all pixel-bin within RF:
            #       if would have been exact pattern match without this pixel
            #           update pixels-explained-per-frame towards hypothetical smaller RF value
            #       else
            #           update pixels-explained-per-frame towards zero

            # way to do this: was there a partial match that missed by one pixel-bin?
            #   then increment pixel-bin that was missed towards RF_size - 1, all others towards zero
            if input_match_num == rf_size - 1 and input_match_num > 0:
                # print('A', self.t, input_match_num)
                # what if input_match_num is zero? still valid technically, but excluded anyways

                # find which pixel-bin was the not-match:
                # tmp = input_exp_1[0, nnz_rf_flat[0]]
                # one element of tmp is zero here
                flat_element_nz = np.nonzero(tmp==0)[0][0]  # before index, it is a tuple of one array of one element, which is the index

                # flat_element_nz: index of np.nonzero(rf.flatten()) that was a not match
                # now find (r, c, bin) for this index

                flat_rf_index = nnz_rf_flat[0][flat_element_nz]
                rf_indices = np.unravel_index(flat_rf_index, rf.shape)
                # print(rf_indices)

                old_val = remove_candidates[rf_indices]
                remove_candidates[nnz_rf] = (1.0 - tau) * remove_candidates[nnz_rf] + tau * 0
                remove_candidates[rf_indices] = (1.0 - tau) * old_val + tau * (rf_size - 1)

            else:
                remove_candidates[nnz_rf] = (1.0 - tau) * remove_candidates[nnz_rf] + tau * 0

        add_candidates[nnz_rf] = 0.0
        remove_candidates[nz_rf] = 0.0

        max_hyp_ind = np.unravel_index(np.argmax(add_candidates, axis=None), add_candidates.shape)
        max_hyp = add_candidates[max_hyp_ind]

        did_add = False
        if max_hyp > rf_px_bins_exp and self.t > self.wait_learn_until:

            # print()
            # print()
            # print('ADDING ONE')
            # print('max_hyp', max_hyp)
            # print('rf_px_bins_exp', rf_px_bins_exp)
            # print('max_hyp_ind', max_hyp_ind)
            # print('t', self.t, self.wait_learn_until, 'of num candidates: ', np.count_nonzero(add_candidates > rf_px_bins_exp))
            # print()
            # print()

            rf[max_hyp_ind] = 1.0

            add_candidates = add_candidates * 0.95
            remove_candidates = remove_candidates * 0.95
            #add_candidates[:, :, :] = 0.0
            #remove_candidates[:, :, :] = 0.0

            did_add = True
            #self.num_additions += 1

        if not did_add:
            # see if remove should be done
            max_hyp_ind = np.unravel_index(np.argmax(remove_candidates, axis=None), remove_candidates.shape)
            max_hyp = remove_candidates[max_hyp_ind]

            if max_hyp > rf_px_bins_exp and self.t > self.wait_learn_until:
                # print()
                # print()
                # print('REMOVING ONE PERHAPS')
                #
                # print('max_hyp', max_hyp)
                # print('rf_px_bins_exp', rf_px_bins_exp)
                # print('max_hyp_ind', max_hyp_ind)
                # print('t', self.t, self.wait_learn_until, 'of num candidates: ', np.count_nonzero(remove_candidates> rf_px_bins_exp))
                # print()
                # print()

                rf[max_hyp_ind] = 0.0
                add_candidates = add_candidates * 0.95
                remove_candidates = remove_candidates * 0.95
                #add_candidates[:, :, :] = 0.0
                #remove_candidates[:, :, :] = 0.0

                #self.num_removals += 1

        did_replace = False
        # TODO here, replace bad RFs with best single-pixel ones from self.single_pixel_bin_RF_exp
        max_tmp = np.amax(self.single_pixel_bin_RF_exp)
        if rf_px_bins_exp < max_tmp and self.rf_ages[rf_index] > self.wait_learn_until:

            self.rf_num_bad[rf_index] += 1

            if self.rf_num_bad[rf_index] > 10000:

                print('REPLACING AN RF COMPLETELY RF:', rf_index, ', time: ', self.t, rf_px_bins_exp, max_tmp, self.rf_ages[rf_index], self.wait_learn_until * 4)

                new_rf_ind = np.unravel_index(np.argmax(self.single_pixel_bin_RF_exp, axis=None), self.single_pixel_bin_RF_exp.shape)

                rf[:, :, :] = 0.0
                rf[new_rf_ind] = 1.0
                self.single_pixel_bin_RF_exp[new_rf_ind] = 0.0

                rf_px_bins_exp = max_tmp
                add_candidates[:, :, :] = 0.0
                remove_candidates[:, :, :]  = 0.0

                self.rf_ages[rf_index] = 0
                self.rf_num_bad[rf_index] = 0
                print(np.sum(rf))
                did_replace = True
        else:

            self.rf_num_bad[rf_index] -= 1
            if self.rf_num_bad[rf_index] < 0:
                self.rf_num_bad[rf_index] = 0

            self.rf_ages[rf_index] = self.rf_ages[rf_index] + 1

        self.rf_list[rf_index] = rf

        if did_replace:
            print(np.sum(self.rf_list[rf_index]))

        self.rf_px_bins_exp[rf_index] = rf_px_bins_exp
        self.add_candidates_list[rf_index] = add_candidates
        self.remove_candidates_list[rf_index] = remove_candidates

        return input_exp_1

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
        arr_exp = np.zeros((arr.shape[0], arr.shape[1] * num_bins_per_pixel), np.float32)
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

    def get_table_ims(self):
        '''

        :return:
        '''

        ims_list = []
        ims_names_list = []

        rf_im = None
        weights_im = None
        for rf_index in range(self.num_rf):
            arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.rf_list[rf_index].flatten()[np.newaxis, :], num_bins_per_pixel=self.bins_per_pixel)
            if rf_im is None:
                rf_im = arr.reshape((self.rf_dim, self.rf_dim))
                weights_im = weights.reshape((self.rf_dim, self.rf_dim))
            else:
                rf_im = np.hstack((rf_im, 1.0 * np.ones((self.rf_dim, 2))))
                rf_im = np.hstack((rf_im, arr.reshape((self.rf_dim, self.rf_dim))))

                weights_im = np.hstack((weights_im, 1.0 * np.ones((self.rf_dim, 2))))
                weights_im = np.hstack((weights_im, weights.reshape((self.rf_dim, self.rf_dim))))

        max_dim = max(rf_im.shape[0], rf_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        rf_im = cv2.resize(rf_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        max_dim = max(weights_im.shape[0], weights_im.shape[1])
        imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        weights_im = cv2.resize(weights_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list.append(rf_im)
        ims_names_list.append('rfs')

        ims_list.append(weights_im)
        ims_names_list.append('weights')

        tmp_im = np.hstack((self.last_input_sub_im, 1.0 * np.ones((self.last_input_sub_im.shape[0], 2)), self.whats_left_im))
        max_dim = max(tmp_im.shape[0], tmp_im.shape[1])
        imscale = 600 / max_dim  # 0.2: full table, 2.0
        tmp_im__2 = cv2.resize(tmp_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list.append(tmp_im__2)
        ims_names_list.append('whats-left-mask')

        if self.plot_graphs:
            print()
            print('Making plot of state vars...')
            self.ax.cla()
            self.ax.plot(self.im_match_prop_history[0:self.t])
            self.fig.savefig("match_prop.png", dpi=100)
            self.ax.cla()
            self.ax.plot(self.mean_im_match_prop_history[0:self.t])
            self.fig.savefig("mean_match_prop.png", dpi=100)

        if time.time() > 5 + self.last_print_time:
            print()
            print('how good are RFs?')
            print(np.sort(self.rf_px_bins_exp))
            print('how good would be single pixel-bin RFs? min, max, mean, median')
            print(np.amin(self.single_pixel_bin_RF_exp), np.amax(self.single_pixel_bin_RF_exp), np.mean(self.single_pixel_bin_RF_exp), np.median(self.single_pixel_bin_RF_exp))
            print()
            self.last_print_time = time.time()

        return ims_list, ims_names_list


class ExpBrainSinglePartialPattern(object):
    '''

    new greedy algorithm for growing/contracting RF over time

    single RF test

    '''

    def __init__(self, params):
        '''

        :param params:
        '''

        self.im_dim = params['image_dim_NxN_pixels']

        self.ims_scale_pixels = 300

        self.otm = OneTimeMessages()

        self.t = 0

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

        # top left of input region

        # 50, 64
        # 65, 76

        self.in_r = 50  # 40 # 65 #50 # 60
        self.in_c = 64  # 84 # 76 #64 # 64

        self.in_bin = 2  # 2

        self.rf_dim = 16

        self.bins_per_pixel = 6
        self.input_history_steps = 1

        assert self.in_r + self.rf_dim < self.im_dim, str((self.in_r + self.rf_dim, self.im_dim))
        assert self.in_c + self.rf_dim < self.im_dim, str((self.in_c + self.rf_dim, self.im_dim))

        # TODO later introduce input history
        #  self.input_feature_len = self.bins_per_pixel * self.rf_dim * self.rf_dim  # * self.input_history_steps

        self.rf = np.zeros((self.rf_dim, self.rf_dim, self.bins_per_pixel))

        # measure of average pixel bins explained per frame, for RF as it currently stands
        self.rf_px_bins_exp_per_frame = 0.0

        # for pixel-bins outside RF: if pixel-bin were added, how much would be explained per frame by new larger RF
        self.hyp_px_bin_exp_per_frame = np.zeros((self.rf_dim, self.rf_dim, self.bins_per_pixel))

        # for pixel-bins inside RF: if pixel-bin were removed, how much would be explained per frame by new smaller RF
        self.remove_hyp_px_bin_exp_per_frame = np.zeros((self.rf_dim, self.rf_dim, self.bins_per_pixel))

        # set self.rf current pixel to 1: considering it the top left
        self.rf[int(self.rf_dim/2), int(self.rf_dim/2), self.in_bin] = 1

        self.count_steps = 0
        self.count_matches = 0

        self.last_im = None
        self.last_match_im = None

        self.tau = 0.0001  # 0.0001
        self.wait_learn_until = 4 * (log(2) / self.tau)

        self.num_additions = 0
        self.num_removals = 0

    # def _r_c_bin_to_index(self, r, c, bin_n):
    #     '''
    #
    #     :return:
    #     '''
    #
    #     return index
    #
    # def _index_to_r_c_bin(self, index):
    #     return r, c, bin_n

    def process_input(self, input_im):

        self.t += 1
        self.count_steps += 1
        self.last_im = input_im.copy()

        input_pixels_1 = input_im[self.in_r:self.in_r + self.rf_dim, self.in_c:self.in_c + self.rf_dim]
        input_arr_1 = input_pixels_1.flatten()[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

        # decide if RF match (all of it must be inside the current rf_im

        # nonzero elements of RF:
        nnz_rf_flat = np.nonzero(self.rf.flatten())  # verified: flatten expands the same as _bin_pixels_expand_columns

        # input values for nonzero elements of RF:
        tmp = input_exp_1[0, nnz_rf_flat[0]]

        rf_size = np.size(tmp)
        input_match_num = np.count_nonzero(tmp)
        assert input_match_num <= rf_size

        if input_match_num == rf_size:
            match = True
        else:
            match = False

        tau = self.tau
        nnz_rf = np.nonzero(self.rf)
        nz_rf = np.nonzero(self.rf==0)

        if match:

            self.count_matches += 1
            # print('match: ', self.t)
            self.last_match_im = input_im.copy()

            self.rf_px_bins_exp_per_frame = (1.0 - tau) * self.rf_px_bins_exp_per_frame + tau * rf_size

            # hypothetical additions to expand to:

            # all pixel-bins activity values (1 or 0) on this frame:
            tmp = input_exp_1.flatten().reshape((self.rf_dim, self.rf_dim, self.bins_per_pixel))
            nnz_tmp = np.nonzero(tmp)
            self.hyp_px_bin_exp_per_frame[nnz_tmp] = (1.0 - tau) * self.hyp_px_bin_exp_per_frame[nnz_tmp] + tau * (rf_size + 1)

            z_tmp = np.nonzero(tmp == 0)
            self.hyp_px_bin_exp_per_frame[z_tmp] = (1.0 - tau) * self.hyp_px_bin_exp_per_frame[z_tmp] + tau * 0

            # assert that only one value is 1 per pixel-bin, in input and in the RF (sanity check)
            do_debug = 0
            if do_debug:
                tmp_sum = np.sum(tmp, axis=2)
                assert tmp_sum.size == np.sum(tmp_sum==1)
                tmp_sum2 = np.sum(self.rf, axis=2)

                assert np.amax(tmp_sum2) <= 1

            # there is a current RF match: update self.remove_hyp_px_bin_exp_per_frame:
            #   for all pixels within RF:
            #       update towards hypothetical smaller RF value, since all sub-patterns are present
            self.remove_hyp_px_bin_exp_per_frame[nnz_rf] = (1.0 - tau) * self.remove_hyp_px_bin_exp_per_frame[nnz_rf] + tau * (rf_size - 1)

        else:
            self.rf_px_bins_exp_per_frame = (1.0 - tau) * self.rf_px_bins_exp_per_frame + tau * 0.0

            self.hyp_px_bin_exp_per_frame = (1.0 - tau) * self.hyp_px_bin_exp_per_frame + tau * 0.0

            # TODO
            # no match: update self.remove_hyp_px_bin_exp_per_frame:
            #   for all pixel-bin within RF:
            #       if would have been exact pattern match without this pixel
            #           update pixels-explained-per-frame towards hypothetical smaller RF value
            #       else
            #           update pixels-explained-per-frame towards zero

            # way to do this: was there a partial match that missed by one pixel-bin?
            #   then increment pixel-bin that was missed towards RF_size - 1, all others towards zero
            if input_match_num == rf_size - 1 and input_match_num > 0:
                # print('A', self.t, input_match_num)
                # what if input_match_num is zero? still valid technically, but excluded anyways

                # find which pixel-bin was the not-match:
                # tmp = input_exp_1[0, nnz_rf_flat[0]]
                # one element of tmp is zero here
                flat_element_nz = np.nonzero(tmp==0)[0][0]  # before index, it is a tuple of one array of one element, which is the index

                # flat_element_nz: index of np.nonzero(rf.flatten()) that was a not match
                # now find (r, c, bin) for this index

                flat_rf_index = nnz_rf_flat[0][flat_element_nz]
                rf_indices = np.unravel_index(flat_rf_index, self.rf.shape)
                # print(rf_indices)

                old_val = self.remove_hyp_px_bin_exp_per_frame[rf_indices]
                self.remove_hyp_px_bin_exp_per_frame[nnz_rf] = (1.0 - tau) * self.remove_hyp_px_bin_exp_per_frame[nnz_rf] + tau * 0
                self.remove_hyp_px_bin_exp_per_frame[rf_indices] = (1.0 - tau) * old_val + tau * (rf_size - 1)

            else:
                self.remove_hyp_px_bin_exp_per_frame[nnz_rf] = (1.0 - tau) * self.remove_hyp_px_bin_exp_per_frame[nnz_rf] + tau * 0

        self.hyp_px_bin_exp_per_frame[nnz_rf] = 0.0
        self.remove_hyp_px_bin_exp_per_frame[nz_rf] = 0.0

        max_hyp_ind = np.unravel_index(np.argmax(self.hyp_px_bin_exp_per_frame, axis=None), self.hyp_px_bin_exp_per_frame.shape)
        max_hyp = self.hyp_px_bin_exp_per_frame[max_hyp_ind]

        did_add = False
        if max_hyp > self.rf_px_bins_exp_per_frame and self.t > self.wait_learn_until:
            print()
            print()
            print('ADDING ONE')
            print('max_hyp', max_hyp)
            print('self.rf_px_bins_exp_per_frame', self.rf_px_bins_exp_per_frame)
            print('max_hyp_ind', max_hyp_ind)
            print('t', self.t, self.wait_learn_until, 'of num candidates: ', np.count_nonzero(self.hyp_px_bin_exp_per_frame > self.rf_px_bins_exp_per_frame))
            print()
            print()

            self.rf[max_hyp_ind] = 1.0
            self.hyp_px_bin_exp_per_frame[:, :, :] = 0.0
            self.remove_hyp_px_bin_exp_per_frame[:, :, :] = 0.0

            did_add = True
            self.num_additions += 1

        if not did_add:
            # see if remove should be done
            max_hyp_ind = np.unravel_index(np.argmax(self.remove_hyp_px_bin_exp_per_frame, axis=None), self.remove_hyp_px_bin_exp_per_frame.shape)
            max_hyp = self.remove_hyp_px_bin_exp_per_frame[max_hyp_ind]

            if max_hyp > self.rf_px_bins_exp_per_frame and self.t > self.wait_learn_until:
                print()
                print()
                print('REMOVING ONE PERHAPS')

                print('max_hyp', max_hyp)
                print('self.rf_px_bins_exp_per_frame', self.rf_px_bins_exp_per_frame)
                print('max_hyp_ind', max_hyp_ind)
                print('t', self.t, self.wait_learn_until, 'of num candidates: ', np.count_nonzero(self.remove_hyp_px_bin_exp_per_frame > self.rf_px_bins_exp_per_frame))
                print()
                print()

                self.rf[max_hyp_ind] = 0.0
                self.hyp_px_bin_exp_per_frame[:, :, :] = 0.0
                self.remove_hyp_px_bin_exp_per_frame[:, :, :] = 0.0

                self.num_removals += 1

        if self.count_steps > 10000:
            print()
            print('******************************************************************')
            print('Match Prop: ', self.count_matches / self.count_steps)
            print('pixel-bins explained per frame: ', self.rf_px_bins_exp_per_frame)
            print('num_additions', self.num_additions)
            print('num_removals', self.num_removals)
            print('******************************************************************')
            print()
            self.count_steps = 0
            self.count_matches = 0

            #print()
            #print(self.hyp_px_bin_exp_per_frame)
            #print()

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
        arr_exp = np.zeros((arr.shape[0], arr.shape[1] * num_bins_per_pixel), np.float32)
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

    def get_table_ims(self):
        # print(self.t)
        # print((np.amin(self.prob), np.amax(self.prob)))
        # print(np.nonzero(self.prob==0))

        ims_list = []
        ims_names_list = []

        arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.rf.flatten()[np.newaxis, :], num_bins_per_pixel=self.bins_per_pixel)

        if self.last_match_im is not None:
            im0 = arr.reshape((self.rf_dim, self.rf_dim))
            im1 = weights.reshape((self.rf_dim, self.rf_dim))
            im2 = self.last_match_im[self.in_r:self.in_r+self.rf_dim, self.in_c:self.in_c+self.rf_dim]
            #cv2.imshow('asd', np.hstack((im0, im1)))
            #cv2.waitKey(1)

            ims_list.append(np.hstack((im0, im1, im2)))
            ims_names_list.append('im')

            # input image with box
            in_region_im = self.last_match_im.copy()
            cv2.rectangle(in_region_im, (self.in_c, self.in_r), (self.in_c + self.rf_dim, self.in_r + self.rf_dim), 255, 2)

            ims_list.append(in_region_im.copy())
            ims_names_list.append('input_region')

        return ims_list, ims_names_list


class ExpBrainWTA(object):
    '''

    first step:
        feedforward-only: test WTA mechanism + homeostatic gain adaptation
    next step:
        add predictive weights

    '''

    def __init__(self, params):

        self.im_dim = params['image_dim_NxN_pixels']

        self.ims_scale_pixels = 300

        self.otm = OneTimeMessages()

        self.t = 0

        self.last_im = None

        self.last_2_im = None
        self.last_3_im = None
        self.last_4_im = None
        self.last_5_im = None
        self.last_6_im = None
        self.last_7_im = None

        self.last_8_im = None

        self.last_9_im = None
        self.last_10_im = None
        self.last_11_im = None
        self.last_12_im = None
        self.last_13_im = None
        self.last_14_im = None

        self.last_15_im = None

        self.rows = 12 * 12  # 6 * 6

        self.prob = np.zeros((self.rows, self.rows), np.float32)

        self.NxN_output = 1  # predicted output square size
        self.NxN_input_pad = 4  # 8  # + pixels to pad on every side for input relative to output square

        self.bins_per_pixel = 6

        self.NxN_input = int(self.NxN_output + 2 * self.NxN_input_pad)

        # for now, table is not spatiotemporal history, just spatial, as a start
        self.input_history_steps = 3
        assert self.input_history_steps == 3, 'havent implement history for input yet for more than 3'

        self.input_feature_len = self.bins_per_pixel * self.NxN_input * self.NxN_input * self.input_history_steps
        self.output_feature_len = self.bins_per_pixel * self.NxN_output * self.NxN_output

        random_init = True
        if random_init:
            self.table_i = np.random.random((self.rows, self.input_feature_len)).astype(np.float32)
        else:
            self.table_i = np.zeros((self.rows, self.input_feature_len), np.float32)

        # top left of input region
        self.in_r = 50 #65 #50
        self.in_c = 64 #76 #64

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
        self.threshold = 0.9 * np.ones(self.rows, np.float32)
        # self.last_win_t = np.zeros()

        self.MAX_TIME = 100000000
        self.WTA_winner_history = np.zeros(self.MAX_TIME, np.int)
        self.mean_WTA_winner_history = np.zeros(self.MAX_TIME, np.float32)

        self.input_match_history = np.zeros(self.MAX_TIME, np.float32)
        self.mean_input_match_history = np.zeros(self.MAX_TIME, np.float32)
        self.mean_gain_history = np.zeros(self.MAX_TIME, np.float32)
        self.mean_mean_gain_history = np.zeros(self.MAX_TIME, np.float32)

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

    def process_input_EGADS(self, input_im):

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
            match_no_gain = np.divide(np.sum(all_mp, axis=1), np.sum(self.table_i, axis=1))

            match = np.multiply(match_no_gain, self.gains)

            argsort_match = np.argsort(match)[::-1]  # largest match first

            multi_wta = True
            if multi_wta:

                unexplained = input_exp.copy()
                active_rows = []

                for k in range(argsort_match.shape[0]):

                    row_i = argsort_match[k]
                    unexplained = unexplained - self.table_i[row_i, :]
                    unexplained[unexplained < 0] = 0
                    val = np.sum(unexplained) / np.sum(input_exp)
                    #print(k, row_i, val)
                    active_rows.append(row_i)

                    # TODO we are trying to compute match only to what is "unexplained"
                    # BUT this is all wrong
                    # already ranked them by overall match; that is where to look first; this would have to be integrated into that process, to work
                    # learn_rate_p = 0.005 * abs(1.0 - match[row_i])

                    if val < 0.4:
                        break

                active_rows = np.array(active_rows, np.int)

                self.WTA_winner_history[self.t] = active_rows.shape[0]

                # unstable
                learn_rate_p = (0.001 * abs(1.0 - match[active_rows]))[:, np.newaxis]
                self.table_i[active_rows, :] = (1.0 - learn_rate_p) * self.table_i[active_rows, :] + learn_rate_p * input_exp
                self.gains = self.gains * 1.004
                self.gains[active_rows] = 1.0

            else:

                win_row = argsort_match[0]

                learn_rate_p = 0.005 * abs(1.0 - match[argsort_match[0]])
                # this is too extreme (forces synchrony):
                # * self.gains[win_row]
                # this forces synchrony as well:
                # - match_no_gain[...

                self.learn_stop_time = np.inf  # 200000

                if self.t < self.learn_stop_time:
                    self.table_i[win_row, :] = (1.0 - learn_rate_p) * self.table_i[win_row, :] + learn_rate_p * input_exp

                    self.gains = self.gains * 1.004
                    self.gains[win_row] = 1.0
                else:
                    self.gains[:] = 1.0

                self.WTA_winner_history[self.t] = win_row
                self.input_match_history[self.t] = match[win_row]
                self.mean_input_match_history[self.t] = np.mean(self.input_match_history[max(0, self.t - 1000):self.t])

            self.mean_gain_history[self.t] = np.mean(self.gains)
            self.mean_mean_gain_history[self.t] = np.mean(self.mean_gain_history[max(0, self.t - 1000):self.t])
            self.t += 1

        self.last_im = input_im.copy()

    def process_input_DYNAMIC_IN_PROGRESS(self, input_im):

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
            match_no_gain = np.divide(np.sum(all_mp, axis=1), np.sum(self.table_i, axis=1))

            match = np.multiply(match_no_gain, self.gains)

            argsort_match = np.argsort(match)[::-1]  # largest match first

            tmp = 1.0
            for k in range(argsort_match.shape[0]):
                row_i = argsort_match[k]
                if match[row_i] > self.threshold[row_i]:
                    self.threshold[row_i] = self.threshold[row_i] * tmp
                    tmp = tmp * 1.1

            active_rows = np.nonzero(np.greater(match, self.threshold))[0]
            inactive_rows = np.nonzero(np.less_equal(match, self.threshold))[0]

            learn_rate_p = (0.0025 * abs(1.0 - match[active_rows]))[:, np.newaxis]
            self.table_i[active_rows, :] = (1.0 - learn_rate_p) * self.table_i[active_rows, :] + learn_rate_p * input_exp

            self.threshold[inactive_rows] = self.threshold[inactive_rows] * 0.5
            self.threshold[active_rows] = 0.2

            #self.gains = self.gains * 1.004
            #self.gains[active_rows] = 1.0

            # n_tmp = 0
            # g = 1.0
            # for k in range(argsort_match.shape[0]):
            #     row_i = argsort_match[k]
            #     if row_i in list(active_rows):
            #         lr_probs = 0.005
            #         print((np.amin(self.prob), np.amax(self.prob)))
            #         self.prob[row_i, active_rows] = (1 - lr_probs) * self.prob[row_i, active_rows] + lr_probs * 1.0
            #         print((np.amin(self.prob), np.amax(self.prob)))
            #         print(' updated prob for ', row_i, active_rows)
            #         n_tmp += 1
            # assert n_tmp == active_rows.shape[0]

            if active_rows.shape[0] > 0:
                for row_i in list(active_rows):
                    for row_j in list(active_rows):
                        if row_i != row_j:
                            lr_probs = 0.1
                            self.prob[row_i, row_j] = (1 - lr_probs) * self.prob[row_i, row_j] + lr_probs * 1.0

                #print(self.t)
                #print((np.amin(self.prob), np.amax(self.prob)))
                # print(active_rows)

            #print(np.nonzero(self.prob==0.0))
            self.WTA_winner_history[self.t] = active_rows.shape[0]
            self.mean_WTA_winner_history[self.t] = np.mean(self.WTA_winner_history[max(0, self.t - 1000):self.t])

            self.mean_gain_history[self.t] = np.mean(self.threshold)
            self.mean_mean_gain_history[self.t] = np.mean(self.mean_gain_history[max(0, self.t - 1000):self.t])
            self.t += 1

        self.last_im = input_im.copy()

    def process_input(self, input_im):

        if self.last_15_im is not None:

            # get input subset from image

            assert self.in_r + self.NxN_input < self.im_dim, str((self.in_r + self.NxN_input, self.im_dim))
            assert self.in_c + self.NxN_input < self.im_dim, str((self.in_c + self.NxN_input, self.im_dim))

            input_pixels_1 = self.last_im[self.in_r:self.in_r + self.NxN_input, self.in_c:self.in_c + self.NxN_input]
            input_arr_1 = input_pixels_1.flatten()[np.newaxis, :]
            input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

            input_pixels_2 = self.last_8_im[self.in_r:self.in_r + self.NxN_input, self.in_c:self.in_c + self.NxN_input]
            input_arr_2 = input_pixels_2.flatten()[np.newaxis, :]
            input_exp_2 = self._bin_pixels_expand_columns(arr=input_arr_2, num_bins_per_pixel=self.bins_per_pixel)

            input_pixels_3 = self.last_15_im[self.in_r:self.in_r + self.NxN_input, self.in_c:self.in_c + self.NxN_input]
            input_arr_3 = input_pixels_3.flatten()[np.newaxis, :]
            input_exp_3 = self._bin_pixels_expand_columns(arr=input_arr_3, num_bins_per_pixel=self.bins_per_pixel)

            input_exp = np.hstack((input_exp_1, input_exp_2, input_exp_3))

            # print(input_exp_1.shape, input_exp.shape)

            assert input_exp.shape[0] == 1
            assert input_exp.shape[1] == self.input_feature_len, str((input_exp.shape[1], self.input_feature_len))

            all_mp = np.multiply(self.table_i, input_exp)
            match_no_gain = np.divide(np.sum(all_mp, axis=1), np.sum(self.table_i, axis=1))

            match = np.multiply(match_no_gain, self.gains)

            argsort_match = np.argsort(match)[::-1]  # largest match first
            win_row = argsort_match[0]

            learn_rate_p = 0.0025 * abs(1.0 - match[argsort_match[0]])
            # this is too extreme (forces synchrony):
            # * self.gains[win_row]
            # this forces synchrony as well:
            # - match_no_gain[...

            self.learn_stop_time = np.inf  # 200000

            if self.t < self.learn_stop_time:
                self.table_i[win_row, :] = (1.0 - learn_rate_p) * self.table_i[win_row, :] + learn_rate_p * input_exp

                self.gains = self.gains * 1.0005
                self.gains[win_row] = 1.0
            else:
                self.gains[:] = 1.0

            self.WTA_winner_history[self.t] = win_row
            self.input_match_history[self.t] = match[win_row]
            self.mean_input_match_history[self.t] = np.mean(self.input_match_history[max(0, self.t - 1000):self.t])
            self.mean_gain_history[self.t] = np.mean(self.gains)
            self.mean_mean_gain_history[self.t] = np.mean(self.mean_gain_history[max(0, self.t - 1000):self.t])
            self.t += 1

        if self.last_14_im is not None:
            self.last_15_im = self.last_14_im.copy()
        if self.last_13_im is not None:
            self.last_14_im = self.last_13_im.copy()
        if self.last_12_im is not None:
            self.last_13_im = self.last_12_im.copy()
        if self.last_11_im is not None:
            self.last_12_im = self.last_11_im.copy()
        if self.last_10_im is not None:
            self.last_11_im = self.last_10_im.copy()
        if self.last_9_im is not None:
            self.last_10_im = self.last_9_im.copy()
        if self.last_8_im is not None:
            self.last_9_im = self.last_8_im.copy()
        if self.last_7_im is not None:
            self.last_8_im = self.last_7_im.copy()

        if self.last_6_im is not None:
            self.last_7_im = self.last_6_im.copy()

        if self.last_5_im is not None:
            self.last_6_im = self.last_5_im.copy()

        if self.last_4_im is not None:
            self.last_5_im = self.last_4_im.copy()

        if self.last_3_im is not None:
            self.last_4_im = self.last_3_im.copy()

        if self.last_2_im is not None:
            self.last_3_im = self.last_2_im.copy()

        if self.last_im is not None:
            self.last_2_im = self.last_im.copy()

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

        # print(self.t)
        # print((np.amin(self.prob), np.amax(self.prob)))
        # print(np.nonzero(self.prob==0))

        ims_list = []
        ims_names_list = []

        table_im_list = []
        weights_im_list = []
        tmp = int(self.input_feature_len / self.input_history_steps)

        for k in range(self.input_history_steps):
            rows, weights = self._collapse_binned_columns_to_pixels(self.table_i[:, k * tmp:(k+1) * tmp], num_bins_per_pixel=self.bins_per_pixel)

            tile_r_c = int(sqrt(rows.shape[1]))
            N = int(sqrt(self.rows))

            table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

            weights_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5
            tile_n = 0

            last_win_row = self.WTA_winner_history[self.t - 1]

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

                    # RF-based / RGC-type view (one circle per pixel) in progress:
                    # tile_r = 0
                    # for im_r in range(r0, r1):
                    #     tile_c = 0
                    #     for im_c in range(c0, c1):
                    #
                    #         val =
                    #
                    #         tile_c += 1
                    #
                    #     tile_r += 1

                    table_im[r0:r1, c0:c1] = tile_im
                    weights_im[r0:r1, c0:c1] = tile_weights

                    if tile_n == last_win_row:
                        table_im[r0:r1, c0 - 1] = 1.0
                        table_im[r0:r1, c1] = 1.0
                        table_im[r0 - 1, c0:c1] = 1.0
                        table_im[r1, c0:c1] = 1.0

                        weights_im[r0:r1, c0 - 1] = 1.0
                        weights_im[r0:r1, c1] = 1.0
                        weights_im[r0 - 1, c0:c1] = 1.0
                        weights_im[r1, c0:c1] = 1.0


                    tile_n += 1
                    c_offset += 1

                r_offset += 1

            table_im_list.append(table_im.copy())
            weights_im_list.append(weights_im.copy())



        # scale all
        for k in range(len(table_im_list)):
            table_im = table_im_list[k]
            weights_im = weights_im_list[k]

            max_dim = max(table_im.shape[0], table_im.shape[1])
            imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
            table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

            ims_list.append(table_im.copy())
            ims_names_list.append('table_im_' + str(k))

            max_dim = max(weights_im.shape[0], weights_im.shape[1])
            imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
            weights_im = cv2.resize(weights_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

            ims_list.append(weights_im.copy())
            ims_names_list.append('weights_im_' + str(k))

        # input image with box
        in_region_im = self.last_im.copy()
        cv2.rectangle(in_region_im, (self.in_c, self.in_r), (self.in_c + self.NxN_input, self.in_r + self.NxN_input), 255, 2)

        ims_list.append(in_region_im.copy())
        ims_names_list.append('input_region')

        ims_scale_input_by_itself = int(self.ims_scale_pixels / N)
        input_itself_im = self.last_im[self.in_r:self.in_r + self.NxN_input, self.in_c:self.in_c+self.NxN_input]

        max_dim = max(input_itself_im.shape[0], input_itself_im.shape[1])
        imscale = ims_scale_input_by_itself / max_dim  # 0.2: full table, 2.0
        input_itself_im = cv2.resize(input_itself_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list.append(input_itself_im.copy())
        ims_names_list.append('input_by_itself')

        # plot mat plot lib
        plot_on = False
        if plot_on:
            print()
            print('Making plot of state vars...')

            self.ax.cla()
            self.ax.plot(self.WTA_winner_history[0:self.t], 'r.')
            self.fig.savefig("WTA_winner_history.png", dpi=100)

            self.ax.cla()
            self.ax.plot(self.mean_WTA_winner_history[0:self.t], 'r.')
            self.fig.savefig("mean_WTA_winner_history.png", dpi=100)

            # self.ax.cla()
            # self.ax.plot(self.mean_input_match_history[0:self.t], 'r-')
            # self.fig.savefig("mean_input_match_history.png", dpi=100)
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
