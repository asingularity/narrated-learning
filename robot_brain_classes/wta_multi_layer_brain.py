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


class WTAMultiLayerBrain(object):
    def __init__(self, params):
        '''

        This class will, for now, simply instantiate multiple WTALayerBrain objects; one per hyperlayer

        TODO
            next hyperlayer means WTALayerBrain needs option to not assume input is pixels and needs binning
            need parameters per hyperlayer i.e. layer_start_times
            need to add input time steps parameter so that second hyperlayer can learn spatiotemporal

        :param params:
        '''

        # ************ supplied parameters ************

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
        self.num_rf_per_wta_layer_per_hl = params['num_rf_per_wta_layer_per_hl']  # TODO np.array([20, 10])

        # how many WTA layers, per hyperlayer
        #   i.e. [6, 4]
        self.num_wta_layers_per_hl = params['num_wta_layer_per_hl']  # TODO  np.array([6, 4])

        # how many time steps, i.e. delay values, of the previous hl's output to use as input to the next hl
        self.input_time_steps_per_hl = params['input_time_steps_per_hl']  # TODO  np.array([1, 10])

        # number (per hl) will apply to all WTA layers of that hl, like num_rf above
        self.layer_start_time_offset_per_hl = params['layer_start_time_offset_per_hl']  # TODO np.array([300000, 300000])

        # this enables turning the learning off after the logical time (based on layer start time offset per hl)
        #       at implied last+1 "start" time
        self.enable_learning_off = params['enable_learning_off']  # TODO True

        # how often in real seconds to print the plots (at a get_table_ims call)
        self.plot_interval = params['plot_interval_seconds']  # TODO 30

        # ************ derived parameters ************

        self.num_hl = len(self.num_wta_layers_per_hl)

        assert len(self.num_rf_per_wta_layer_per_hl) == self.num_hl
        assert len(self.num_wta_layers_per_hl) == self.num_hl
        assert len(self.layer_start_time_offset_per_hl) == self.num_hl

        max_time = 10000000

        # ************ initialize hyperlayers ************

        self.layer_start_times_per_hl = []
        self.hl_output_histories = []
        self.rec_error = []
        self.mean_rec_error = []
        self.weights = []

        hl_start_time = 0
        hl_input_state_dim = self.input_im_dim * self.input_im_dim * self.bins_per_pixel

        print()
        for hl in range(self.num_hl):
            print()
            print('Initializing hypelayer:', hl)
            # *** layer start times ***
            print()
            layer_start_times = hl_start_time + np.arange(self.num_wta_layers_per_hl[hl]) * self.layer_start_time_offset_per_hl[hl]
            self.layer_start_times_per_hl.append(layer_start_times)
            hl_start_time = np.amax(layer_start_times) + self.layer_start_time_offset_per_hl[hl]

            # *** history for spatiotemporal RFs, input to next hl ***

            if hl == self.num_hl - 1:
                steps_to_store = 1  # last hyperlayer; we don't have a use for storing this actually, currently
            else:
                steps_to_store = self.input_time_steps_per_hl[hl + 1]

            output_state_dim = self.num_rf_per_wta_layer_per_hl[hl] * self.num_wta_layers_per_hl[hl]

            print('    output history: max_delay', steps_to_store, 'states_dim_list:', [output_state_dim])
            print()
            hl_output_history = StatesLimitedHistory(params={'max_delay': steps_to_store,
                                                             'states_dim_list': [output_state_dim]})

            self.hl_output_histories.append(hl_output_history)

            # num_rf_prev = self.num_rf_per_wta_layer_per_hl[hl]
            # num_wta_layers_prev = len(hyperlayer_layer_start_times[hl - 1])
            # temporal_steps_input_current = self.hyperlayer_rf_temporal_steps[hl]
            #
            # input_dim = num_rf_prev * num_wta_layers_prev * temporal_steps_input_current
            # input_history = StatesLimitedHistory(params={'max_delay': max_predict_time,
            #                                              'states_dim_list': [self.input_dim]})

            self.rec_error.append(np.zeros(max_time))
            self.mean_rec_error.append(np.zeros(max_time))

            hl_weights = []
            for k in range(self.num_wta_layers_per_hl[hl]):
                hl_weights.append(np.zeros((self.num_rf_per_wta_layer_per_hl[hl], hl_input_state_dim)))

            self.weights.append(hl_weights)

            if hl < self.num_hl - 1:
                # next layer's input state dim
                hl_input_state_dim = output_state_dim * self.input_time_steps_per_hl[hl + 1]

        assert hl_start_time < max_time, str((hl_start_time, max_time))

        if self.enable_learning_off:
            self.learning_off_time = hl_start_time

        print()
        print('Initialized!')
        print()
        print('layer start times per hl:')
        print(self.layer_start_times_per_hl)
        print()
        print('learning off time: ', self.learning_off_time)

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


        if False:
            # ***************** OLD BAD *****************
            layer_start_times = params['wta_layer_start_time_offsets']

            self.hyperlayer_rf_temporal_steps = params['rf_temporal_steps_list']

            self.num_hyperlayers = len(hyperlayer_layer_start_times)

            self.hyperlayers = []

            for hl in range(self.num_hyperlayers):
                hl_params = {}

                hl_params['image_dim_display'] = params['image_dim_display']
                hl_params['learning_rate'] = params['learning_rate']

                hl_params['num_rf'] = params['num_rf']
                hl_params['max_time'] = params['max_time']
                hl_params['learning_off_time'] = params['learning_off_time']
                hl_params['plot_interval_seconds'] = params['plot_interval_seconds']

                hl_params['layer_start_times'] = hyperlayer_layer_start_times[hl]

                if hl == 0:
                    hl_params['bins_per_pixel'] = params['bins_per_pixel']

                    # TODO change WTALayerBrain to take in an "input dim" instead of an input_dim_NxN; no longer the square root, since prev layer will sent activations that are not an "image"
                    # TODO if it is a perfect square, should assume image, otherwise not
                    hl_params['input_dim'] = params['image_dim_NxN_pixels'] * params['image_dim_NxN_pixels']
                else:
                    # if this is zero: means assume input already binned, as it will be for all but first hyperlayer
                    # TODO make this change in WTALayerBrain
                    hl_params['bins_per_pixel'] = 0

                    # input dim of next hyperlayer ; assumes "num_rf" is same in every hyperlayer
                    # this superclass will be providing the requisite delayed temporal steps

                    # for now we are assuming same num_rf for every hyperlayer
                    num_rf_prev = params['num_rf']
                    num_wta_layers_prev = len(hyperlayer_layer_start_times[hl - 1])
                    temporal_steps_input_current = self.hyperlayer_rf_temporal_steps[hl]

                    # TODO now input dim
                    hl_params['input_dim'] = num_rf_prev * num_wta_layers_prev * temporal_steps_input_current

                self.hyperlayers.append(WTALayerBrain(params=hl_params))

                # TODO for every hyperlayer need a states history based on what next layer needs as input
                input_history = StatesLimitedHistory(params={'max_delay': max_predict_time,
                                                             'states_dim_list': [self.input_dim]})


    def process_input(self, input_im):
        input_pixels_1 = input_im
        input_arr_1 = input_pixels_1.flatten()[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

        self.last_input_im = input_pixels_1.copy()

        # this assumes that first hyperlayer only uses 1 time step!
        # we may need to change this if needed later
        hl_input = input_exp_1.copy()

        for hl in range(self.num_hl):

            if hl_input is not None:
                hl_output = self._process_hyperlayer(hl_input=hl_input, hl=hl)

            if hl_output is not None:
                assert hl_output.shape[0] > 1, hl_output.shape

                # history update with hl_output

                try:
                    self.hl_output_histories[hl].store_new_states(newest_states_list=[hl_output], extra_data_list=[None])
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
                                                                               delay_short=0).flatten()

                    assert hl_input.shape[0] > 1, hl_input.shape

                    hl_input = hl_input[np.newaxis, :]
            else:
                hl_input = None

        # TODO estimate predictive weight (predictive prob from individual units in hyperlayer K to units in hyperlayer K-1)

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

        layer_start_times = self.layer_start_times_per_hl[hl]

        remainder = hl_input.copy()
        reconstruction = np.zeros((1, input_feature_len))

        hl_output = None

        for layer_n in range(self.num_wta_layers_per_hl[hl]):
            if self.t >= layer_start_times[layer_n]:


                '''
                    on second hyperlayer:

                    eff_frame = np.sum(np.abs(self.weights[hl][layer_n] - remainder), axis=1)
                    ValueError: operands could not be broadcast together with shapes (5,60) (1,660)
                '''

                eff_frame = np.sum(np.abs(self.weights[hl][layer_n] - remainder), axis=1)
                win_rf_index = np.argmin(eff_frame)

                lr = self.lr_base

                if self.t < self.learning_off_time:
                    self.weights[hl][layer_n][win_rf_index, :] = (1.0 - lr) * self.weights[hl][layer_n][win_rf_index, :] + lr * remainder[0, :]

                remainder = remainder - self.weights[hl][layer_n][win_rf_index, :]

                reconstruction = reconstruction + self.weights[hl][layer_n][win_rf_index, :]

                wta_layer_output = np.zeros(self.num_rf_per_wta_layer_per_hl[hl])
                wta_layer_output[win_rf_index] = 1.0

                if hl_output is None:
                    hl_output = wta_layer_output.copy()
                else:
                    hl_output = np.hstack((hl_output, wta_layer_output))
            else:
                if hl_output is None:
                    hl_output = np.zeros(self.num_rf_per_wta_layer_per_hl[hl])
                else:
                    hl_output = np.hstack((hl_output, np.zeros(self.num_rf_per_wta_layer_per_hl[hl])))

        reconstruction[reconstruction > 1] = 1
        reconstruction[reconstruction < 0] = 0

        if hl == 0:
            self.last_remainder_im = remainder.copy()
            self.last_reconstruction_im = reconstruction.copy()

        rec_error = np.sum(np.abs(reconstruction - hl_input))
        # sum_remainder = np.sum(np.abs(remainder))

        self.rec_error[hl][self.t] = rec_error
        self.mean_rec_error[hl][self.t] = np.mean(self.rec_error[hl][max(0, self.t - self.mean_fr):self.t])

        return hl_output

    def get_table_ims(self):

        # TODO need to decide how to visualize non-input hyperlayer

        ims_list = []
        ims_names_list = []

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

            tmp_all_im = None
            for layer_n in range(self.num_wta_layers_per_hl[hl]):
                tmp_im = None

                arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.weights[hl][layer_n], num_bins_per_pixel=self.bins_per_pixel)

                for r_tmp in range(weights.shape[0]):

                    im0 = arr[r_tmp, :].reshape((self.input_im_dim, self.input_im_dim))  # rf_dim
                    im1 = weights[r_tmp, :].reshape((self.input_im_dim, self.input_im_dim))

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

        else:
            # TODO (************************ ABOVE WON'T WORK FOR HL > 0:::!!!!!!!!!!!!!!!!!!!!!!!!
            # TODO how to visualize for hl > 0 ?

            # for layer_n in range(self.num_wta_layers_per_hl[hl]):
            #
            #     w = self.weights[hl][layer_n]
            #

            pass

        return ims_list, ims_names_list

    def _do_plots(self):
        print()
        print('making plot')
        print()

        for hl in range(self.num_hl):
            try:
                self.ax.cla()
                self.ax.plot(self.rec_error[hl][0:self.t], color='r')
                self.ax.plot(self.mean_rec_error[hl][0:self.t], color='b')

                for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]:
                    self.ax.axhline(y=k, color='g')

                for k in self.layer_start_times_per_hl[hl]:
                    self.ax.axvline(x=k, color='g')

                self.ax.axvline(x=self.learning_off_time, color='r')
                self.fig.savefig("rec_error_hyperlayer_" + str(hl) + ".png", dpi=100)
                self.ax.cla()
            except:
                print()
                print('!!!!!!!! Error !!!!!!!!')
                print()
                print('Plot Failed!')
                print()
                print('!!!!!!!! Error !!!!!!!!')
                print()

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



































