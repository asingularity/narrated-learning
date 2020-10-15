import time
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt
from brain_components_classes.states_history import StatesLimitedHistory

from utils.one_time_messages import OneTimeMessages

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log

class WTAIterativeBrain(object):
    def __init__(self, params):
        '''

        :param params:
        '''

        self.im_dim = params['image_dim_display']  # TODO 1500

        # only applies to first hyperlayer (hl==0)
        self.input_im_dim = params['image_dim_NxN_pixels']  # TODO 16

        self.lr_base = params['learning_rate']  # TODO 0.0001 * 0.25

        # only applies to first hyperlayer (hl==0)
        self.bins_per_pixel = params['bins_per_pixel']  # TODO 6

        # this is a list of num_rf in each WTA layer, one number per hyperlayer
        #   i.e [20, 10], assuming 3 WTA layers per hyperlayer, means:
        #   hyperlayer_0 has WTA layers with num_rf: [20, 20, 20]
        #   hyperlayer_1 has WTA layers with num_rf: [10, 10, 10]
        self.num_rf_per_hl = params['num_rf_per_hl']  # TODO np.array([20, 10])

        # number of iterations per input (fixed iterative k in k-WTA)
        self.num_iter = params['num_iter_per_input']  # TODO one number (all hyperlayers); can theoretically be more than RFs but probably want less

        # how many time steps, i.e. delay values, of the previous hl's output to use as input to the next hl
        self.input_time_steps_per_hl = params['input_time_steps_per_hl']  # TODO  np.array([1, 10])
        assert self.input_time_steps_per_hl[0] == 1, 'we do not support more than 1 input time step for first hl; see process_input function'

        # when to start each hyperlayer
        self.start_time_per_hl = params['start_time_per_hl']  # TODO np.array([0, 300000])

        # this enables turning the learning off after the logical time (based on layer start time offset per hl)
        #       at implied last+1 "start" time
        self.learning_off_time = params['learning_off_time']  # TODO sum of above + some more

        # how often in real seconds to print the plots (at a get_table_ims call)
        self.plot_interval = params['plot_interval_seconds']  # TODO 30

        # this one basically disables the "background" in the reconstruction / prediction images
        # also shows weights instead of values in prediction image
        self.enable_hack_skip_first_layer = params['enable_hack_skip_first_layer']  # TODO True only for balls input

        # enable "generic" weights viz; overall not very useful
        self.enable_generic_weights_viz = params['enable_generic_weights_viz']
        assert self.enable_generic_weights_viz is False, 'generic weights viz not implemented for this one (only WTAMultiLayerBrain)'

        # TODO make these parameters

        # for initializing arrays
        max_time = 10000000

        # predictive parameters:
        # how long to predict ahead
        self.predict_time_steps = 1

        # how many delayed lateral input time steps to use
        self.predict_lateral_input_time_steps = 4

        # how many delayed feedback input time steps to use
        self.predict_feedback_input_time_steps = 1

        # ************ derived parameters ************

        assert self.learning_off_time < max_time
        assert self.learning_off_time > self.start_time_per_hl[-1]

        self.num_hl = len(self.num_rf_per_hl)
        for hl in range(1, self.num_hl):
            assert self.start_time_per_hl[hl] > self.start_time_per_hl[hl - 1]
            assert self.num_rf_per_hl[hl] % 10 == 0, 'num rfs must be divisible by 10! for plotting purposes'

        assert len(self.input_time_steps_per_hl) == self.num_hl
        assert len(self.start_time_per_hl) == self.num_hl

        # ************ initialize hyperlayers ************
        self.hl_output_histories = []
        self.rec_error = []
        self.mean_rec_error = []
        self.weights = []
        self.predict_weights = []

        hl_start_time = 0
        hl_input_state_dim = self.input_im_dim * self.input_im_dim * self.bins_per_pixel

        print()

        for hl in range(self.num_hl):
            print()
            print('Initializing hyperlayer:', hl)
            print()

            # *** history for spatiotemporal RFs, input to next hl ***

            if hl == self.num_hl - 1:
                steps_to_store = max(self.predict_time_steps, 1) + self.predict_lateral_input_time_steps + self.predict_feedback_input_time_steps # last hyperlayer; we don't have a use for storing this actually, currently
            else:
                steps_to_store = max(self.predict_time_steps, self.input_time_steps_per_hl[hl + 1]) + self.predict_lateral_input_time_steps + self.predict_feedback_input_time_steps

            output_state_dim = self.num_rf_per_hl[hl]

            print('    output history: max_delay', steps_to_store, 'states_dim_list:', [output_state_dim])
            print()
            hl_output_history = StatesLimitedHistory(params={'max_delay': steps_to_store,
                                                             'states_dim_list': [output_state_dim],
                                                             'store_extra_data': False})

            self.hl_output_histories.append(hl_output_history)

            self.rec_error.append(np.zeros(max_time))
            self.mean_rec_error.append(np.zeros(max_time))

            hl_weights = np.zeros((self.num_rf_per_hl[hl], hl_input_state_dim))
            print('    adding feedforward weights for hyperlayer:', hl, ', of shape (num rf per hl, input dim):', hl_weights.shape)

            self.weights.append(hl_weights)

            if hl < self.num_hl - 1:
                # next layer's input state dim
                hl_input_state_dim = output_state_dim * self.input_time_steps_per_hl[hl + 1]

        print()
        print('Initialized!')
        print()
        print('start times per hl:')
        print(self.start_time_per_hl)
        print()
        print('learning off time: ', self.learning_off_time)
        print()

        # ************ other ************

        self.t = 0
        self.last_input_im = None
        self.mean_fr = 10000
        self.last_plot_time = 0

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)

        # ************ predictive stuff ************

        # all RFs in HL==1 -> all RFs in HL==0

        num_rfs_hl_0 = self.num_rf_per_hl[0]
        num_rfs_hl_1 = self.num_rf_per_hl[1]

        self.pm = np.zeros((num_rfs_hl_1, num_rfs_hl_0))  # shape: (from RFs, to RFs) i.e. (HL==1, HL==0)
        self.pm_lr = self.lr_base

        self.predicted_hl_0_rf_activies_history = StatesLimitedHistory(params={'max_delay': self.predict_time_steps,
                                                                               'states_dim_list': [num_rfs_hl_0],
                                                                               'store_extra_data': False})

        self.input_im_history = StatesLimitedHistory(params={'max_delay': self.predict_time_steps,
                                                             'states_dim_list': [self.input_im_dim * self.input_im_dim],
                                                             'store_extra_data': False})

    def process_input(self, input_im):
        input_pixels_1 = input_im
        input_pixels_flat = input_pixels_1.flatten()

        self.input_im_history.store_new_states(newest_states_list=[input_pixels_flat])

        input_arr_1 = input_pixels_flat[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

        self.last_input_im = input_pixels_1.copy()

        # this assumes that first hyperlayer only uses 1 time step!
        # we may need to change this if needed later
        hl_input = input_exp_1.copy()

        for hl in range(self.num_hl):
            if self.t >= self.start_time_per_hl[hl]:
                hl_output = self._process_hyperlayer(hl_input=hl_input, hl=hl)

                assert hl_output.shape[0] > 1, hl_output.shape
                tmp1 = np.nonzero(hl_output)[0]

                # self.activity_rasters[hl][self.t, tmp1] = 1

                # history update with hl_output

                try:
                    self.hl_output_histories[hl].store_new_states(newest_states_list=[hl_output],
                                                                  extra_data_list=[None])
                except ValueError:
                    print()
                    print('error storing states!')
                    print('    hl', hl, ', hl_output.shape: ', hl_output.shape)
                    print()
                    raise

                if hl < self.num_hl - 1:
                    # input for next layer: get history based on spatiotemporal history time steps
                    # set hl_input
                    hl_input = self.hl_output_histories[hl].get_state_sequence(state_index=0,
                                                                               delay_long=self.input_time_steps_per_hl[hl + 1] - 1,
                                                                               delay_short=0)
                    hl_input = hl_input.flatten()

                    assert hl_input.shape[0] > 1, hl_input.shape

                    hl_input = hl_input[np.newaxis, :]

                # predictive
                # OLD METHOD: DISABLED
                if True:
                    if hl == 1 and self.t > 1:

                        hl_0_rf_activities_current = self.hl_output_histories[0].get_state(state_index=0, delay=0)
                        hl_1_rf_activities_past = self.hl_output_histories[1].get_state(state_index=0,
                                                                                        delay=self.predict_time_steps)

                        hl_1_rf_activities_current = self.hl_output_histories[1].get_state(state_index=0, delay=0)

                        assert hl_0_rf_activities_current is not None
                        assert hl_1_rf_activities_past is not None

                        if np.count_nonzero(hl_0_rf_activities_current) > 0 and np.count_nonzero(
                                hl_1_rf_activities_past) > 0:
                            rows = np.nonzero(hl_1_rf_activities_past)[0].astype(np.int)
                            cols = np.nonzero(hl_0_rf_activities_current)[0].astype(np.int)

                            rows_exp = np.tile(rows, cols.shape[0])
                            cols_exp = np.repeat(cols, rows.shape[0])

                            # fix this; this is probably not computing the right thing
                            #   should be better now
                            # also, we are probably gonna have to look at prediction image to make sense of this
                            # for that, we will need to compute the actual prediction; not just training input asabove
                            #       meaning: use the prediction matrix on current hl_1 activities to make hl_0 activities prediction
                            # also need to define prediction time steps ahead

                            # what we want is: prob(l==0, present), |given| (l==1, past)

                            pm_old = self.pm.copy()

                            self.pm[rows, :] = (1.0 - self.pm_lr) * pm_old[rows, :] + self.pm_lr * 0.0
                            self.pm[rows_exp, cols_exp] = (1.0 - self.pm_lr) * pm_old[
                                rows_exp, cols_exp] + self.pm_lr * 1.0

                            # TODO compute actual next-frame prediction

                            # self.predict_time_steps use here!!!

                            # max is taken over columns; such that for every hl-0 "post", you store the max "pre" prob:
                            # TODO maybe this should actually be a sum, instead; to counter the multiple wta-layers where one post can make *multiple* valid / strong predictions in different wta-layers, that shouldn't be ignored
                            #       i.e. we have order-dependence...
                            hl_0_probs = np.amax(self.pm[np.nonzero(hl_1_rf_activities_current)[0], :], axis=0)

                            assert hl_0_probs.shape[0] == hl_0_rf_activities_current.shape[0], str(
                                (hl_0_probs.shape, hl_0_rf_activities_current.shape))

                            self.predicted_hl_0_rf_activies_history.store_new_states(newest_states_list=[hl_0_probs])

        # we do prediction and prediction learning on newest data, after processing and storing based on new input above

        self.t += 1

    def _process_hyperlayer(self, hl_input, hl):
        '''

        :param hl_input:
        :param hl: index of hyperlayer
        :return:
        '''

        assert hl_input.shape[0] == 1
        assert hl_input.shape[1] > 0

        input_feature_len = hl_input.shape[1]

        remainder = hl_input.copy()
        reconstruction = np.zeros((1, input_feature_len))

        reconstruction_no_first = np.zeros((1, input_feature_len))

        hl_output = np.zeros(self.num_rf_per_hl[hl])

        rfs_valid = np.ones(self.num_rf_per_hl[hl])

        for iter_i in range(self.num_iter):
            valid_indices = np.nonzero(rfs_valid)[0]

            eff_frame = np.sum(np.abs(self.weights[hl][valid_indices, :] - remainder), axis=1)
            win_rf_index = valid_indices[np.argmin(eff_frame)]

            lr = self.lr_base

            if self.t < self.learning_off_time:
                self.weights[hl][win_rf_index, :] = (1.0 - lr) * self.weights[hl][win_rf_index, :] + lr * remainder[0, :]

            remainder = remainder - self.weights[hl][win_rf_index, :]

            reconstruction = reconstruction + self.weights[hl][win_rf_index, :]

            # TODO get rid of this; there is no "hack first layer" for this model
            reconstruction_no_first = reconstruction

            rfs_valid[win_rf_index] = 0
            hl_output[win_rf_index] = 1.0

        reconstruction[reconstruction > 1] = 1
        reconstruction[reconstruction < 0] = 0

        if self.enable_hack_skip_first_layer:
            reconstruction_no_first[reconstruction_no_first > 1] = 1
            reconstruction_no_first[reconstruction_no_first < 0] = 0

        if hl == 0:
            self.last_remainder_im = remainder.copy()

            if self.enable_hack_skip_first_layer:
                self.last_reconstruction_im = reconstruction_no_first.copy()
            else:
                self.last_reconstruction_im = reconstruction.copy()

        rec_error = np.sum(np.abs(reconstruction - hl_input))
        # sum_remainder = np.sum(np.abs(remainder))

        self.rec_error[hl][self.t] = rec_error
        self.mean_rec_error[hl][self.t] = np.mean(self.rec_error[hl][max(0, self.t - self.mean_fr):self.t])

        return hl_output

        # TODO return an array with binary output per RF: did it appear at least once
        # TODO we may not allow one RF to be active more than once; no longer technically correct reconstruction then...
        #   without counting the "number"
        # TODO or we could return not a binary, but a number, like a "rate"

    def get_table_ims(self):

        ims_list = []
        ims_names_list = []

        # TODO uncomment and continue!
        if time.time() > self.last_plot_time + self.plot_interval:

            self._do_plots()

            self.last_plot_time = time.time()

        for hl in range(self.num_hl):

            ims_list_hl, ims_names_list_hl = self._do_images_for_hyperlayer(hl=hl)

            ims_list.extend(ims_list_hl)
            ims_names_list.extend(ims_names_list_hl)

        return ims_list, ims_names_list

    def _do_images_for_hyperlayer(self, hl):
        '''

        :param hl:
        :return:
        '''

        ims_list = []
        ims_names_list = []

        if hl == 0:
            tmp_im = None

            arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.weights[hl],
                                                                   num_bins_per_pixel=self.bins_per_pixel)

            for r_tmp in range(weights.shape[0]):

                im0 = arr[r_tmp, :].reshape((self.input_im_dim, self.input_im_dim))  # rf_dim
                im1 = weights[r_tmp, :].reshape((self.input_im_dim, self.input_im_dim))

                # if np.amin(im1) < 0:
                #     print(np.amin(im1), np.amax(im1))

                tmp2 = np.hstack((im0, im1))

                if tmp_im is None:
                    tmp_im = tmp2.copy()
                else:
                    tmp3 = np.zeros((2, tmp_im.shape[1]))
                    tmp_im = np.vstack((tmp_im, tmp3, tmp2))

            tmp_all_im = tmp_im

            max_dim = max(tmp_all_im.shape[0], tmp_all_im.shape[1])
            imscale = self.im_dim / max_dim  # 0.2: full table, 2.0
            tmp_im = cv2.resize(tmp_all_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

            ims_list.append(tmp_im)
            ims_names_list.append('weights_v, weights_w_' + str(hl))

            rec_im, rec_w = self._collapse_binned_columns_to_pixels(arr_exp=self.last_reconstruction_im, num_bins_per_pixel=self.bins_per_pixel)
            im0 = rec_im[0, :].reshape((self.input_im_dim, self.input_im_dim))
            im1 = rec_w[0, :].reshape((self.input_im_dim, self.input_im_dim))

            rem_im, rem_w = self._collapse_binned_columns_to_pixels(arr_exp=self.last_remainder_im, num_bins_per_pixel=self.bins_per_pixel)
            imr0 = rem_im[0, :].reshape((self.input_im_dim, self.input_im_dim))
            imr1 = rem_w[0, :].reshape((self.input_im_dim, self.input_im_dim))

            ims_list.append(np.hstack((self.last_input_im,
                                       np.zeros((self.input_im_dim, 2)), im0, np.zeros((self.input_im_dim, 2)), im1,
                                       np.zeros((self.input_im_dim, 2)), imr0, np.zeros((self.input_im_dim, 2)), imr1,)))
            ims_names_list.append('input, reconstruct, remainder_' + str(hl))

            # TODO need prediction image!!!

        # TODO need hl==1 image!!!

        # no generic for now

        return ims_list, ims_names_list

    def _do_plots(self):
        print()
        print('making plot')
        print()

        print('prediction info:')
        print('    min pm:', np.amin(self.pm))
        print('    max pm:', np.amax(self.pm))
        print('    mean pm:', np.mean(self.pm))
        print('    sum pm:', np.sum(self.pm))
        print()
        #print(self.pm)
        #print()

        for hl in range(self.num_hl):
            try:
                self.ax.cla()
                self.ax.plot(self.rec_error[hl][0:self.t], color='r')
                self.ax.plot(self.mean_rec_error[hl][0:self.t], color='b')

                for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]:
                    self.ax.axhline(y=k, color='g')

                self.ax.axvline(x=self.start_time_per_hl[hl], color='g')

                self.ax.axvline(x=self.learning_off_time, color='r')
                self.fig.savefig("rec_error_hyperlayer_" + str(hl) + ".png", dpi=100)

                # TODO plot prediction error (also vs. current frame for reference!) same plot!
                # TODO also maybe hack visualization to manually remove background RF for now (since multi-predict active anyways!!)

                # TODO fix this it isn't right!!!
                #
                # thing_to_plot = self.activity_rasters[hl][max(0, self.t-5000):self.t, :]
                #
                # #print('THING TO PLOT min', np.amin(thing_to_plot), ', max: ', np.amax(thing_to_plot), ', shape: ', thing_to_plot.shape)
                #
                # self.ax.cla()
                # self.ax.plot(thing_to_plot, color='b', marker='.', linestyle='')
                # self.ax.set_ylim([-1, self.num_wta_layers_per_hl[hl] * self.num_rf_per_wta_layer_per_hl[hl]])
                # self.fig.savefig("activities_" + str(hl) + ".png", dpi=100)

            except:
                print()
                print('!!!!!!!! Error !!!!!!!!')
                print()
                print('Plot Failed!')
                print()
                print('!!!!!!!! Error !!!!!!!!')
                print()

                raise

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

















































