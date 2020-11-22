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

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log



class WTADeterminateBrain(object):
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
        self.input_time_steps_per_hl = params['input_num_time_steps_per_hl']  # TODO  np.array([1, 10])
        assert self.input_time_steps_per_hl[0] == 1, 'we do not support more than 1 input time step for first hl; see process_input function'

        self.input_delta_time_steps_per_hl = params['input_delta_time_steps_per_hl']

        # when to start each hyperlayer
        self.start_time_per_hl = params['start_time_per_hl']  # TODO np.array([0, 300000])

        # this enables turning the learning off after the logical time (based on layer start time offset per hl)
        #       at implied last+1 "start" time
        self.learning_off_time_per_hl = params['learning_off_time_per_hl']  # TODO sum of above + some more

        # how often in real seconds to print the plots (at a get_table_ims call)
        self.plot_interval = params['plot_interval_seconds']  # TODO 30

        # enable "generic" weights viz; overall not very useful
        self.enable_generic_weights_viz = params['enable_generic_weights_viz']
        assert self.enable_generic_weights_viz is False, 'generic weights viz not implemented for this one (only WTAMultiLayerBrain)'

        # predictive parameters:
        # how long to predict ahead
        self.predict_time_steps = params['predict_ahead_time']

        # TODO make these parameters

        # for initializing arrays
        max_time = 10000000

        self.use_negative_input = False  # instead of zero, set non-inputs to -1 for all layers

        # ************ derived parameters ************

        self.num_hl = len(self.num_rf_per_hl)

        for hl in range(self.num_hl):
            assert self.learning_off_time_per_hl[hl] < max_time
            assert self.learning_off_time_per_hl[hl] > self.start_time_per_hl[hl]

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

        self.predict_error = []
        self.predict_error_reference = []
        self.mean_predict_error = []
        self.mean_predict_error_reference = []

        self.num_rfs_active = []
        self.mean_num_rfs_active = []

        hl_start_time = 0
        hl_input_state_dim = self.input_im_dim * self.input_im_dim * self.bins_per_pixel

        print()

        for hl in range(self.num_hl):
            print()
            print('Initializing hyperlayer:', hl)
            print()

            # *** history for spatiotemporal RFs, input to next hl ***

            if hl == self.num_hl - 1:
                steps_to_store = max(self.predict_time_steps, 1)  # last hyperlayer; we don't have a use for storing this actually, currently
            else:
                steps_to_store = max(self.predict_time_steps, self.input_delta_time_steps_per_hl[hl + 1] * self.input_time_steps_per_hl[hl + 1]) + self.predict_time_steps

            output_state_dim = self.num_rf_per_hl[hl]

            print('    output history: max_delay', steps_to_store, 'states_dim_list:', [output_state_dim])
            print()
            hl_output_history = StatesLimitedHistory(params={'max_delay': steps_to_store,
                                                             'states_dim_list': [output_state_dim],
                                                             'store_extra_data': False})

            self.hl_output_histories.append(hl_output_history)

            self.rec_error.append(np.zeros(max_time))
            self.mean_rec_error.append(np.zeros(max_time))

            self.num_rfs_active.append(np.zeros(max_time))
            self.mean_num_rfs_active.append(np.zeros(max_time))

            hl_weights = np.zeros((self.num_rf_per_hl[hl], hl_input_state_dim))
            print('    adding feedforward weights for hyperlayer:', hl, ', of shape (num rf per hl, input dim):', hl_weights.shape)

            self.weights.append(hl_weights)

            if hl < self.num_hl - 1:
                # next layer's input state dim
                hl_input_state_dim = output_state_dim * self.input_time_steps_per_hl[hl + 1]

                self.predict_error.append(np.zeros(max_time))
                self.predict_error_reference.append(np.zeros(max_time))
                self.mean_predict_error.append(np.zeros(max_time))
                self.mean_predict_error_reference.append(np.zeros(max_time))

        print()
        print('Initialized!')
        print()
        print('start times per hl:')
        print(self.start_time_per_hl)
        print()
        print('learning off times: ', self.learning_off_time_per_hl)
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

        self.last_activities = []
        for hl in range(self.num_hl):
            self.last_activities.append(np.zeros(self.num_rf_per_hl[hl]))

        # ************ predictive stuff ************
        self.input_im_history = StatesLimitedHistory(params={'max_delay': self.predict_time_steps,
                                                             'states_dim_list': [self.input_im_dim * self.input_im_dim],
                                                             'store_extra_data': False})

        if self.num_hl > 1:
            # all RFs in HL==1 -> all RFs in HL==0

            num_rfs_hl_0 = self.num_rf_per_hl[0]
            num_rfs_hl_1 = self.num_rf_per_hl[1]

            self.pm = np.zeros((num_rfs_hl_1, num_rfs_hl_0))  # shape: (from RFs, to RFs) i.e. (HL==1, HL==0)
            self.pm_lr = self.lr_base

            self.predicted_hl_0_rf_activies_history = StatesLimitedHistory(params={'max_delay': self.predict_time_steps,
                                                                                   'states_dim_list': [num_rfs_hl_0],
                                                                                   'store_extra_data': False})

        self.m_d = []
        self.lr_m_d = self.lr_base
        for hl in range(self.num_hl):
            # measure determinacy
            self.m_d.append(np.zeros(self.num_rf_per_hl[hl]))

        # raster
        self.raster_history = []
        self.raster_history.append(np.zeros((self.input_im_dim * self.input_im_dim * self.bins_per_pixel, max_time), np.uint8))
        for hl in range(self.num_hl):
            self.raster_history.append(np.zeros((self.num_rf_per_hl[0], max_time), np.uint8))


    def process_input(self, input_im):
        input_pixels_1 = input_im
        input_pixels_flat = input_pixels_1.flatten()

        self.input_im_history.store_new_states(newest_states_list=[input_pixels_flat])

        input_arr_1 = input_pixels_flat[np.newaxis, :]
        input_exp_1 = self._bin_pixels_expand_columns(arr=input_arr_1, num_bins_per_pixel=self.bins_per_pixel)

        nnz_raster = np.nonzero(input_exp_1.flatten())[0]
        self.raster_history[0][nnz_raster, self.t] = 1

        self.last_input_im = input_pixels_1.copy()

        # this assumes that first hyperlayer only uses 1 time step!
        # we may need to change this if needed later
        hl_input = input_exp_1.copy()

        for hl in range(self.num_hl):
            if self.t >= self.start_time_per_hl[hl]:
                hl_output = self._process_hyperlayer(hl_input=hl_input, hl=hl)

                assert hl_output.shape[0] > 1, hl_output.shape
                nnz_raster = np.nonzero(hl_output)[0]

                self.raster_history[hl + 1][nnz_raster, self.t] = 1

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
                                                                               delay_long=self.input_delta_time_steps_per_hl[hl + 1] * self.input_time_steps_per_hl[hl + 1] - 1,
                                                                               delay_short=0,
                                                                               oldest_first=False)

                    # hl_input:
                    #   [oldest data     ]
                    #   [...             ]
                    #   [most recent data]

                    time_steps_take = np.arange(self.input_time_steps_per_hl[hl + 1]) * self.input_delta_time_steps_per_hl[hl + 1]
                    hl_input = hl_input[time_steps_take, :].flatten()

                    assert hl_input.shape[0] > 1, hl_input.shape

                    hl_input = hl_input[np.newaxis, :]

                # predictive
                # OLD METHOD: DISABLED
                make_prediction = False
                if make_prediction:
                    if hl == 1 and self.t > 1:

                        hl_0_rf_activities_current = self.hl_output_histories[0].get_state(state_index=0, delay=0)
                        hl_0_rf_activities_past = self.hl_output_histories[0].get_state(state_index=0,
                                                                                        delay=self.predict_time_steps)

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
                            self.pm[rows_exp, cols_exp] = (1.0 - self.pm_lr) * pm_old[rows_exp, cols_exp] + self.pm_lr * 1.0

                            # TODO compute actual next-frame prediction

                            # self.predict_time_steps use here!!!

                            # max is taken over columns; such that for every hl-0 "post", you store the max "pre" prob:
                            # TODO maybe this should actually be a sum, instead; to counter the multiple wta-layers where one post can make *multiple* valid / strong predictions in different wta-layers, that shouldn't be ignored
                            #       i.e. we have order-dependence...

                            # standard:
                            hl_0_probs = np.amax(self.pm[np.nonzero(hl_1_rf_activities_current)[0], :], axis=0)

                            # experiment (worse):
                            # hl_0_probs = np.sum(self.pm[np.nonzero(hl_1_rf_activities_current)[0], :], axis=0)

                            assert hl_0_probs.shape[0] == hl_0_rf_activities_current.shape[0], str(
                                (hl_0_probs.shape, hl_0_rf_activities_current.shape))

                            num_rf_per_iter = int(self.num_rf_per_hl[0] / self.num_iter[0])
                            predictions_grouped = hl_0_probs.reshape((self.num_iter[0], num_rf_per_iter))
                            win_rf_indices = np.arange(self.num_iter[0]) * num_rf_per_iter + np.argmax(predictions_grouped, axis=1)

                            hl_0_prediction = np.zeros(self.num_rf_per_hl[0])
                            hl_0_prediction[win_rf_indices] = 1.0

                            self.predicted_hl_0_rf_activies_history.store_new_states(newest_states_list=[hl_0_prediction])

                            # retrieve old prediction and store errors and reference errors
                            prediction_past = self.predicted_hl_0_rf_activies_history.get_state(state_index=0, delay=self.predict_time_steps)

                            predict_error = np.sum(np.abs(hl_0_rf_activities_current - prediction_past))
                            predict_error_reference = np.sum(np.abs(hl_0_rf_activities_past - prediction_past))

                            self.predict_error[0][self.t] = predict_error
                            self.predict_error_reference[0][self.t] = predict_error_reference
                            self.mean_predict_error[0][self.t] = np.mean(self.predict_error[0][max(0, self.t - self.mean_fr):self.t])
                            self.mean_predict_error_reference[0][self.t] = np.mean(self.predict_error_reference[0][max(0, self.t - self.mean_fr):self.t])

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

        if self.use_negative_input:
            hl_output = -np.ones(self.num_rf_per_hl[hl])
        else:
            hl_output = np.zeros(self.num_rf_per_hl[hl])

        self.last_activities[hl] = np.zeros(self.num_rf_per_hl[hl])

        rfs_valid = np.ones(self.num_rf_per_hl[hl])

        while True:

            # choose which RF best reduces reconstruction error for remainder
            #   ie compute hypothetical remainder given subtracting this RF from current remainder, for all the remaining RFs

            w = self.weights[hl].copy()  # [valid_indices, :]

            # c0 = w <= -0.5
            # c1 = np.logical_and(w > -0.5, w <= 0.0)
            # c2 = np.logical_and(w > 0.0, w <= 0.5)
            # c3 = w > 0.5
            #
            # w[np.nonzero(c0)] = -1.0
            # w[np.nonzero(c1)] = 0.0
            # w[np.nonzero(c2)] = 0.0
            # w[np.nonzero(c3)] = 1.0

            #w[w <= 0.5] = 0.0
            #w[w > 0.5] = 1.0

            rem = remainder
            hypothetical_remainder_sums = np.zeros(self.num_rf_per_hl[hl])

            # is abs(a - b) ==  abs(b - a) always?
            #   yes
            compute_eff(w, rem, hypothetical_remainder_sums)

            valid_indices = np.nonzero(rfs_valid)[0]
            hypothetical_remainder_sums = hypothetical_remainder_sums[valid_indices]
            tmp_argmin = np.argmin(hypothetical_remainder_sums)
            win_rf_index = valid_indices[tmp_argmin]
            win_remainder_sum = hypothetical_remainder_sums[tmp_argmin]

            if win_remainder_sum >= np.sum(np.abs(remainder)):

                # if adding the "best" RF would be worse than current remainder; break and don't add this RF

                #if hl == 1:
                #    print('BREAK win_remainder_sum: ', win_remainder_sum, 'remainder sum:', np.sum(np.abs(remainder)))

                break

            self.last_activities[hl][win_rf_index] += 1
            hl_output[win_rf_index] = 1

            # that RF learns on inputs that it contributed to for remainder
            if self.start_time_per_hl[hl] <= self.t < self.learning_off_time_per_hl[hl]:
                lr = self.lr_base
                self.weights[hl][win_rf_index, :] = (1.0 - lr) * self.weights[hl][win_rf_index, :] + lr * remainder[0, :]

                #per_weight_lr = lr * np.abs(self.weights[hl][win_rf_index, :])
                #per_weight_lr[per_weight_lr < 0.1 * lr] = 0.1 * lr
                #self.weights[hl][win_rf_index, :] = np.multiply((1.0 - per_weight_lr), self.weights[hl][win_rf_index, :]) + np.multiply(per_weight_lr, remainder[0, :])

            # re-compute remainder
            remainder = remainder - w[win_rf_index, :]
            reconstruction = reconstruction + w[win_rf_index, :]

            # if uncommented: can't choose this one again
            # rfs_valid[win_rf_index] = 0

        # all RFs that were not selected learn (slowly?) on remainder after all selections above
        if self.start_time_per_hl[hl] <= self.t < self.learning_off_time_per_hl[hl]:
            lr = 0.1 * self.lr_base
            no_active_rf_indices = np.nonzero(self.last_activities[hl] == 0)

            if hl==1:
                pass
                # print('*', np.amin(remainder), np.amax(remainder), np.count_nonzero(remainder), remainder.shape)

                # 0.0 1.0 66 (1, 720)
                # 0.0 1.0 71 (1, 720)
                # 0.0 1.0 70 (1, 720)
                # 0.0 1.0 72 (1, 720)

            self.weights[hl][no_active_rf_indices, :] = (1.0 - lr) * self.weights[hl][no_active_rf_indices, :] + lr * remainder[0, :]

            #per_weight_lr = lr * np.abs(self.weights[hl][no_active_rf_indices, :])
            #per_weight_lr[per_weight_lr < 0.1 * lr] = 0.1 * lr
            #self.weights[hl][no_active_rf_indices, :] = np.multiply((1.0 - per_weight_lr), self.weights[hl][no_active_rf_indices, :]) + np.multiply(per_weight_lr, remainder[0, :])

        reconstruction[reconstruction > 1] = 1
        reconstruction[reconstruction < 0] = 0

        if hl == 0:
            self.last_remainder_im = remainder.copy()
            self.last_reconstruction_im = reconstruction.copy()

        rec_error = np.sum(np.abs(reconstruction - hl_input))
        # sum_remainder = np.sum(np.abs(remainder))

        self.rec_error[hl][self.t] = rec_error
        self.mean_rec_error[hl][self.t] = np.mean(self.rec_error[hl][max(0, self.t - self.mean_fr):self.t])

        self.num_rfs_active[hl][self.t] = np.count_nonzero(self.last_activities[hl])
        self.mean_num_rfs_active[hl][self.t] = np.mean(self.num_rfs_active[hl][max(0, self.t - self.mean_fr):self.t])

        #if hl == 1:
        #    print(int(np.sum(hl_output)))

        return hl_output

    def get_table_ims(self):
        # print()
        # print('*******************')
        # print()
        # for hl in range(self.num_hl):
        #     print('hl', hl)
        #     print()
        #     print(self.m_d[hl])
        #     print()

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
            tmp_im_all = None
            tmp_im = None

            arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=self.weights[hl],
                                                                   num_bins_per_pixel=self.bins_per_pixel)

            for r_tmp in range(weights.shape[0]):  # per rf

                im0 = arr[r_tmp, :].reshape((self.input_im_dim, self.input_im_dim))  # rf_dim
                im1 = weights[r_tmp, :].reshape((self.input_im_dim, self.input_im_dim))

                # im1 = 0.5 * (im1 + 1)

                active_count = self.last_activities[hl][r_tmp]
                im_active_count = np.zeros((self.input_im_dim, self.input_im_dim))

                include_active_count_im = False
                if include_active_count_im:
                    # font
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # org
                    org = (2, im_active_count.shape[1] - 2)
                    # fontScale
                    fontScale = 0.4
                    # Blue color in BGR
                    color = (40, 40, 40)
                    # Line thickness of 2 px
                    thickness = 1
                    # Using cv2.putText() method
                    im_active_count = cv2.putText(im_active_count, str(int(active_count)), org, font, fontScale, color, thickness, cv2.LINE_AA)

                    # if np.amin(im1) < 0:
                    #     print(np.amin(im1), np.amax(im1))

                    tmp2 = np.hstack((im0, im1, im_active_count))
                else:
                    tmp2 = np.hstack((im0, im1))

                if tmp_im is None:
                    if active_count > 0:
                        tmp2[:, 0] = 0.8
                        tmp2[:, -1] = 0.8
                        tmp2[0, :] = 0.8
                        tmp2[-1, :] = 0.8
                    tmp_im = tmp2.copy()
                else:
                    if active_count > 0:
                        tmp2[:, 0] = 0.8
                        tmp2[:, -1] = 0.8
                        tmp2[0, :] = 0.8
                        tmp2[-1, :] = 0.8
                        tmp3 = 0.2 * np.ones((2, tmp_im.shape[1]))
                    else:
                        tmp3 = 0.2 * np.ones((2, tmp_im.shape[1]))

                    tmp_im = np.vstack((tmp_im, tmp3, tmp2))

                if r_tmp > 0 and (r_tmp + 1) % 10 == 0:
                    if tmp_im_all is None:
                        tmp_im_all = tmp_im.copy()
                    else:
                        spacer = 0.5 * np.ones((tmp_im_all.shape[0], 3))
                        tmp_im_all = np.hstack((tmp_im_all, spacer, tmp_im))

                    tmp_im = None

            tmp_all_im = tmp_im_all

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
            ENABLE_PREDICT_IM = False
            if ENABLE_PREDICT_IM:
                predict_im_show = self._get_predict_im()
                ims_list.append(predict_im_show)
                ims_names_list.append('now, predict, past')

        # TODO need hl==1 image!!!
        elif hl == 1:

            tmp_im_all_v = None
            tmp_im_all_w = None

            tmp_im_v = None
            tmp_im_w = None

            for rf_ind in range(self.num_rf_per_hl[hl]):
                # get sequence image for this RF:
                v_seq_im, w_seq_im = self._get_hl_1_sequence_im(rf_weights=self.weights[hl][rf_ind, :])
                active_count = self.last_activities[hl][rf_ind]

                if tmp_im_v is None:
                    if active_count > 0:
                        v_seq_im[:, 0] = 0.8
                        v_seq_im[:, -1] = 0.8
                        v_seq_im[0, :] = 0.8
                        v_seq_im[-1, :] = 0.8

                        w_seq_im[:, 0] = 0.8
                        w_seq_im[:, -1] = 0.8
                        w_seq_im[0, :] = 0.8
                        w_seq_im[-1, :] = 0.8

                    tmp_im_v = v_seq_im.copy()
                    tmp_im_w = w_seq_im.copy()
                else:

                    if active_count > 0:
                        v_seq_im[:, 0] = 0.8
                        v_seq_im[:, -1] = 0.8
                        v_seq_im[0, :] = 0.8
                        v_seq_im[-1, :] = 0.8

                        w_seq_im[:, 0] = 0.8
                        w_seq_im[:, -1] = 0.8
                        w_seq_im[0, :] = 0.8
                        w_seq_im[-1, :] = 0.8

                        tmp3_v = 0.2 * np.ones((2, tmp_im_v.shape[1]))
                        tmp3_w = 0.2 * np.ones((2, tmp_im_w.shape[1]))
                    else:
                        tmp3_v = 0.2 * np.ones((2, tmp_im_v.shape[1]))
                        tmp3_w = 0.2 * np.ones((2, tmp_im_w.shape[1]))

                    tmp_im_v = np.vstack((tmp_im_v, tmp3_v, v_seq_im))

                    tmp_im_w = np.vstack((tmp_im_w, tmp3_w, w_seq_im))

                if rf_ind > 0 and (rf_ind + 1) % 40 == 0:
                    if tmp_im_all_v is None:
                        tmp_im_all_v = tmp_im_v.copy()
                        tmp_im_all_w = tmp_im_w.copy()
                    else:
                        spacer = 0.5 * np.ones((tmp_im_all_v.shape[0], 6))
                        tmp_im_all_v = np.hstack((tmp_im_all_v, spacer, tmp_im_v))
                        tmp_im_all_w = np.hstack((tmp_im_all_w, spacer, tmp_im_w))

                    tmp_im_v = None
                    tmp_im_w = None

            hl_1_im_v = tmp_im_all_v
            hl_1_im_w = tmp_im_all_w

            # scale this image:
            max_dim = max(hl_1_im_v.shape[0], hl_1_im_v.shape[1])
            imscale = self.im_dim / max_dim  # 0.2: full table, 2.0
            hl_1_im_v = cv2.resize(hl_1_im_v, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

            max_dim = max(hl_1_im_w.shape[0], hl_1_im_w.shape[1])
            imscale = self.im_dim / max_dim  # 0.2: full table, 2.0
            hl_1_im_w = cv2.resize(hl_1_im_w, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

            ims_list.append(hl_1_im_v)
            ims_names_list.append('v_RF_weights_hyperlayer_' + str(hl))

            ims_list.append(hl_1_im_w)
            ims_names_list.append('w_RF_weights_hyperlayer_' + str(hl))

            # reconstruction (of sequence)
            hl_1_rf_activities_current = self.hl_output_histories[1].get_state(state_index=0, delay=0)
            hl_1_rf_indices = np.nonzero(hl_1_rf_activities_current)[0]

            hl_1_rec = self._get_hl_1_reconstruction(hl_1_rf_indices)

            if hl_1_rec is not None:
                ims_list.append(hl_1_rec)
                ims_names_list.append('hl-1-reconstruction')

        # no generic for now

        return ims_list, ims_names_list


    def _get_hl_1_reconstruction(self, hl_1_rf_indices):
        hl = 1

        rf_sums_per_del = [None] * self.input_time_steps_per_hl[hl]

        if len(hl_1_rf_indices) == 0:
            return None

        for rf_ind in hl_1_rf_indices:

            rf_weights = self.weights[hl][rf_ind, :]
            k = 0

            for time_del in range(self.input_time_steps_per_hl[hl]):
                num_weights_step = self.num_rf_per_hl[hl - 1]

                rf_weights_step = rf_weights[k:k + num_weights_step]
                rf_to_add = None

                k2 = 0

                for prev_rf_ind in range(self.num_rf_per_hl[hl - 1]):
                    # if (not self.enable_hack_skip_first_layer) or prev_wta_ind > 0:
                    rf_tmp = self.weights[hl - 1][prev_rf_ind, :]
                    if rf_to_add is None:
                        rf_to_add = rf_weights_step[k2] * rf_tmp
                    else:
                        rf_to_add = rf_to_add + rf_weights_step[k2] * rf_tmp
                    k2 += 1

                k += num_weights_step

                if rf_sums_per_del[time_del] is None:
                    rf_sums_per_del[time_del] = rf_to_add
                else:
                    rf_sums_per_del[time_del] += rf_to_add

        im_t_delay_hstack_v = None
        im_t_delay_hstack_w = None

        actual_sequence = None

        for time_del in range(self.input_time_steps_per_hl[hl]):
            rf_sums_im = rf_sums_per_del[time_del]
            rf_sums_im = rf_sums_im * 1.0 / np.amax(rf_sums_im)

            tmp_im_v, tmp_im_w = self._collapse_binned_columns_to_pixels(arr_exp=rf_sums_im[np.newaxis, :], num_bins_per_pixel=self.bins_per_pixel)
            tmp_im_v = tmp_im_v.reshape((self.input_im_dim, self.input_im_dim))
            tmp_im_w = tmp_im_w.reshape((self.input_im_dim, self.input_im_dim))

            actual_past = self.input_im_history.get_state(state_index=0, delay=time_del).reshape((self.input_im_dim, self.input_im_dim))

            if im_t_delay_hstack_v is None:
                im_t_delay_hstack_v = tmp_im_v.copy()
                im_t_delay_hstack_w = tmp_im_w.copy()
                actual_sequence = actual_past.copy()
            else:
                spacer = 0.5 * np.ones((tmp_im_v.shape[0], 2))
                im_t_delay_hstack_v = np.hstack((im_t_delay_hstack_v, spacer, tmp_im_v))
                im_t_delay_hstack_w = np.hstack((im_t_delay_hstack_w, spacer, tmp_im_w))
                actual_sequence = np.hstack((actual_sequence, spacer, actual_past))

        im_t_delay_rec = np.vstack((im_t_delay_hstack_v, 0.5 * np.ones((2, im_t_delay_hstack_v.shape[1])), im_t_delay_hstack_w, 0.5 * np.ones((2, im_t_delay_hstack_v.shape[1])), actual_sequence))

        return im_t_delay_rec


    def _get_predict_im(self):
        # most straightforward way to predict: pick winner per wta-layer
        # also need: the input image from the past, for comparison, to actual predicted input image

        # assume:
        #   prediction steps: tau
        #   prediction@ t - tau, [should match], actual image @ t
        # show:
        #   prediction image @ t - tau
        #   actual image @ t
        #   actual image @ t - tau

        actual_past = self.input_im_history.get_state(state_index=0, delay=self.predict_time_steps).reshape((self.input_im_dim, self.input_im_dim))
        actual_present = self.input_im_history.get_state(state_index=0, delay=0).reshape((self.input_im_dim, self.input_im_dim))

        prediction_of_hl_0 = self.predicted_hl_0_rf_activies_history.get_state(state_index=0, delay=self.predict_time_steps)  # this is in the past
        # print(prediction_of_hl_0.shape)  # (120,)
        layer_w = self.weights[0]  # (num_rf, feature_len)

        # two ways to get prediction image:
        # (1) choose a "winner" per wta-layer or wta-group, based on prediction strength
        #       add only the "winning" RF to the prediction image
        # (2) sum all RFs by their prediction weight

        prediction_exp_method_1 = np.zeros(self.input_im_dim * self.input_im_dim * self.bins_per_pixel)
        prediction_exp_method_2 = np.zeros(self.input_im_dim * self.input_im_dim * self.bins_per_pixel)

        thing_to_add = None
        max_k_val = -np.inf

        # method 1:
        num_rf_per_iter = int(self.num_rf_per_hl[0] / self.num_iter[0])
        predictions_grouped = prediction_of_hl_0.reshape((self.num_iter[0], num_rf_per_iter))
        win_rf_indices = np.arange(self.num_iter[0]) * num_rf_per_iter + np.argmax(predictions_grouped, axis=1)
        prediction_exp_method_1[:] = np.sum(layer_w[win_rf_indices, :], axis=0)

        # method 2:
        prediction_exp_method_2[:] = np.sum(np.multiply(prediction_of_hl_0[:, np.newaxis], layer_w), axis=0)
        # for rf_i in range(layer_w.shape[0]):
        #    prediction_exp = prediction_exp + thing_to_add
        #
        #     # for rf_i  in range(layer_w.shape[0]):
        #     #     prediction_exp += prediction_past[k] * layer_w[rf_i, :]
        #     #
        #     #     k += 1

        # TODO return image for both methods:

        prediction_exp_method_1[prediction_exp_method_1 < 0] = 0
        prediction_exp_method_1[prediction_exp_method_1 > 1] = 1

        prediction_exp_method_2[prediction_exp_method_2 < 0] = 0
        prediction_exp_method_2[prediction_exp_method_2 > 1] = 1

        predict_im_show = None

        for prediction_exp in [prediction_exp_method_1, prediction_exp_method_2]:
            arr, weights = self._collapse_binned_columns_to_pixels(arr_exp=prediction_exp[np.newaxis, :], num_bins_per_pixel=self.bins_per_pixel)

            im0_v = arr[0, :].reshape((self.input_im_dim, self.input_im_dim))
            im0_w = weights[0, :].reshape((self.input_im_dim, self.input_im_dim))

            # im0_w = (im0_w - np.amin(im0_w)) * 1.0 / (np.amax(im0_w) - np.amin(im0_w))

            predict_im_tmp = np.hstack((actual_present, 0.5 * np.ones((self.input_im_dim, 2)),
                                        im0_v, 0.5 * np.ones((self.input_im_dim, 2)),
                                        im0_w, 0.5 * np.ones((self.input_im_dim, 2)),
                                        actual_past, 0.5 * np.ones((self.input_im_dim, 2))))

            if predict_im_show is None:
                predict_im_show = predict_im_tmp.copy()
            else:
                spacer = 0.2 * np.ones((3, predict_im_show.shape[1]))
                predict_im_show = np.vstack((predict_im_show, spacer, predict_im_tmp))

        return predict_im_show


    def _get_hl_1_sequence_im(self, rf_weights):
        '''

        for an rf with these weights, get hstack sequence of delayed RF sums of hl==0 layer
        i.e. deconstruct temporal RF into delay-step spatial components manually, by weight

        :param rf_weights:
        :return:
        '''

        hl = 1  # this function is only for hl==1

        k = 0

        im_t_delay_hstack_v = None
        im_t_delay_hstack_w = None

        for time_del in range(self.input_time_steps_per_hl[hl]):
            # now we need to weighted-add the hl==0 RFs, weighing with these corresponding RF weights
            # use same hack enable option to leave out rf 0
            num_weights_step = self.num_rf_per_hl[hl - 1]

            rf_weights_step = rf_weights[k:k + num_weights_step]
            rf_to_add = None

            k2 = 0

            for prev_rf_ind in range(self.num_rf_per_hl[hl - 1]):
                # if (not self.enable_hack_skip_first_layer) or prev_wta_ind > 0:
                rf_tmp = self.weights[hl - 1][prev_rf_ind, :]
                if rf_to_add is None:
                    rf_to_add = rf_weights_step[k2] * rf_tmp
                else:
                    rf_to_add = rf_to_add + rf_weights_step[k2] * rf_tmp
                k2 += 1

            k += num_weights_step

            # hstack by time delay
            tmp_im_v, tmp_im_w = self._collapse_binned_columns_to_pixels(arr_exp=rf_to_add[np.newaxis, :], num_bins_per_pixel=self.bins_per_pixel)
            tmp_im_v = tmp_im_v.reshape((self.input_im_dim, self.input_im_dim))
            tmp_im_w = tmp_im_w.reshape((self.input_im_dim, self.input_im_dim))

            # tmp_im_w = 0.5 * (tmp_im_w + 1)

            if im_t_delay_hstack_v is None:
                im_t_delay_hstack_v = tmp_im_v.copy()
                im_t_delay_hstack_w = tmp_im_w.copy()
            else:
                spacer = 0.5 * np.ones((tmp_im_v.shape[0], 2))
                im_t_delay_hstack_v = np.hstack((im_t_delay_hstack_v, spacer, tmp_im_v))
                im_t_delay_hstack_w = np.hstack((im_t_delay_hstack_w, spacer, tmp_im_w))

        return im_t_delay_hstack_v, im_t_delay_hstack_w


    def _do_plots(self):
        print()
        print('making plot')
        print()

        # raster of input
        self.ax.cla()
        # how many to display for input? all of them (1536) is too much for this plot
        num_rf = 480  # self.raster_history[0].shape[0]
        raster_plot = np.transpose(np.multiply(self.raster_history[0][0:num_rf, max(0, self.t - 200):self.t], 1 + np.arange(num_rf)[:, np.newaxis]))
        t = np.arange(raster_plot.shape[0])
        self.ax.plot(t, raster_plot, color='b', marker='.', linestyle='')
        self.fig.savefig("raster_input" + ".png", dpi=100)

        for hl in range(self.num_hl):
            try:
                self.ax.cla()
                self.ax.plot(self.rec_error[hl][0:self.t], color='r')
                self.ax.plot(self.mean_rec_error[hl][0:self.t], color='b')

                # for k in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]:
                #    self.ax.axhline(y=k, color='g')

                # self.ax.axvline(x=self.start_time_per_hl[hl], color='g')
                # self.ax.axvline(x=self.learning_off_time_per_hl[hl], color='r')

                self.fig.savefig("rec_error_hyperlayer_" + str(hl) + ".png", dpi=100)

                self.ax.cla()
                self.ax.plot(self.mean_num_rfs_active[hl][0:self.t])
                self.fig.savefig("num_rfs_active_" + str(hl) + ".png", dpi=100)

                self.ax.cla()
                num_rf = self.raster_history[hl + 1].shape[0]
                raster_plot = np.transpose(np.multiply(self.raster_history[hl + 1][0:num_rf, max(0, self.t - 200):self.t],
                                                       np.arange(num_rf)[:, np.newaxis]))
                t = np.arange(raster_plot.shape[0])
                self.ax.plot(t, raster_plot, color='b', marker='.', linestyle='')
                self.fig.savefig("raster_layer_" + str(hl) + ".png", dpi=100)

                if hl < self.num_hl - 1:
                    self.ax.cla()
                    # self.ax.plot(self.predict_error[hl][0:self.t], color='g')
                    self.ax.plot(self.mean_predict_error[hl][0:self.t], color='r')

                    # self.ax.plot(self.predict_error_reference[hl][0:self.t], color='k')
                    self.ax.plot(self.mean_predict_error_reference[hl][0:self.t], color='b')
                    self.fig.savefig("predict_errors_" + str(hl) + ".png", dpi=100)

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





