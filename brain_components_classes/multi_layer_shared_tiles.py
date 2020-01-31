import time
import cv2
import numpy as np
import random
import pickle

from utils.w_save_load_helper import save_W_prob
from math import sqrt, sin, cos
from cuda_dist_query import CudaTable
from utils.load_sim_history import load_states_history, load_td_info_history
from utils.fps_counter import FPSCounter
from brain_components_classes.states_history import StatesLimitedHistory


class MultiLayerSharedTiles(object):
    def __init__(self, params):
        '''
            collection of SingleLayerSharedTile objects, one per layer
            handles passing of data between them

            example:

            self.predictor_ensemble = SingleLayerTrace(params={
                'input_dim': 96,
                'goal_context_dim': 1,
                'enable_learning': True,
                'tile_entries_per_layer': [8000, 8000, 8000]
                'table_learn_time_per_layer': [16000, 16000, 16000],
                'prediction_learn_time_per_layer': [16000, 16000, 16000],
                'pre_init_goal_contexts': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                'max_history_length': 1000000
            })

        '''

        self.input_dim = params['input_dim']
        self.goal_context_dim = params['goal_context_dim']
        self.learning_enabled = params['enable_learning']
        self.tile_entries_per_layer = params['tile_entries_per_layer']
        self.table_learn_time_per_layer = params['table_learn_time_per_layer']
        self.prediction_learn_time_per_layer = params['prediction_learn_time_per_layer']
        self.tiles_per_layer_NxN = params['tiles_per_layer_NxN']
        self.enable_weight_bias = params['enable_weight_bias']

        self.color_enabled = params['color_enabled']

        pre_init_goal_contexts = params['pre_init_goal_contexts']
        max_history_length = params['max_history_length']

        self.table_ims_scale = float(params['table_ims_scale_pixels'])

        # TODO should we have these arguments??
        #'enable_table_im': True,
        #'combine_table_with_weights': False,
        #'enable_weights_im': True,
        #'enable_selection_im': False,
        #'selection_im_last_k_samples': 1,  # only allows 1 currently
        #'enable_inv_selection_im': True,
        #'inv_selection_im_last_k_samples': 5

        # TODO these are current settings for display:
        self.enable_select_im = True
        self.enable_separate_weights_im = False  # if False, will show combined table + weights with best method
        self.enable_W_im = False
        self.enable_samples_by_row_im = False

        # predictors always try to predict the next step (but there is decaying trace)
        self.predict_time = 1

        assert len(self.tile_entries_per_layer) == len(self.table_learn_time_per_layer) == len(self.prediction_learn_time_per_layer)

        self.n_layers = len(self.tile_entries_per_layer)

        offset_t = 0

        self.layer_table_learn_time_ranges = []
        self.layer_prediction_learn_time_ranges = []
        print()

        self.tables = []
        self.weight_error_sums = []
        self.weight_error_counts = []

        self.W_by_layer = []
        self.W_count_by_layer = []
        self.S_count_by_layer = []
        self.init_I_row_num = []
        self.nz_by_layer_row = []

        self.trace_tau = 0.9
        self.max_trace_time = 100  # length of keeping track of trace for W.
        # so, minimal possible trace value in self.W is = pow(self.trace_tau, self.max_trace_time)

        max_trace_time = self.max_trace_time

        # this is per-tile
        self.I_history = StatesLimitedHistory(params={'max_delay': max_trace_time,
                                                      'states_dim_list': [self.tiles_per_layer_NxN[0] * self.tiles_per_layer_NxN[0]]})

        for k in range(self.n_layers):
            self.init_I_row_num.append(None)

            self.layer_table_learn_time_ranges.append((offset_t + 100,
                                                       offset_t + 100 + self.table_learn_time_per_layer[k]))

            offset_t = offset_t + 100 + self.table_learn_time_per_layer[k]

            self.layer_prediction_learn_time_ranges.append((offset_t + 100,
                                                           offset_t + 100 + self.prediction_learn_time_per_layer[k]))

            offset_t = offset_t + 100 + self.prediction_learn_time_per_layer[k]

            print('layer:', k,
                  'table learn:', self.layer_table_learn_time_ranges[k],
                  'prediction learn:', self.layer_prediction_learn_time_ranges[k])

            if k == 0:
                table_input_dim = self.input_dim
            else:
                assert False, 'upper layers not implemented yet!'
                table_input_dim = self.tile_entries_per_layer[k - 1] * self.num_tiles_per_layer[k - 1]

            num_entries = self.tile_entries_per_layer[k]

            self.tables.append(CudaTable(num_entries=num_entries,
                                         input_dim=int(table_input_dim),
                                         enable_weight_bias=self.enable_weight_bias))

            print('init table with entries:', self.tile_entries_per_layer[k], 'input_dim:', int(table_input_dim))
            print()

            if self.color_enabled:
                assert False, 'Not implemented! need to pass in color_enabled to CudaTable init'
                error_sum = np.zeros((num_entries, int(table_input_dim) / 3), np.float32)
                error_count = np.zeros((num_entries, int(table_input_dim) / 3), np.float32)
            else:
                error_sum = np.zeros((num_entries, int(table_input_dim)), np.float32)
                error_count = np.zeros((num_entries, int(table_input_dim)), np.float32)

            self.weight_error_sums.append(error_sum)
            self.weight_error_counts.append(error_count)

            # init prediction matrix stuff

            self.W_by_layer.append(np.zeros((num_entries, num_entries), np.float))
            self.W_count_by_layer.append(np.zeros((num_entries, num_entries), np.int))
            self.S_count_by_layer.append(np.zeros(num_entries, np.int))

            # nnz lists for optimization

            nnz_counts_list = []
            for row in range(num_entries):
                nnz_counts_list.append([])

            self.nz_by_layer_row.append(nnz_counts_list)

        print()

        assert offset_t < max_history_length

        self.t = 0

        self.last_analysis_time = 0

        self.checkerboard = None

        self.selection_im = None
        self.samples_by_row_im = None
        # TODO init goal context stuff

    def step(self, raycast_image, input_state, input_x_y_theta, goal_context_state_learning, goal_context_state_task, last_motor_command):
        '''

        :param raycast_image: (32x32x3)
        :param input_state:
        :param input_x_y_theta:
        :param goal_context_state_learning:
        :param goal_context_state_task:
        :param last_motor_command:
        :return:
        '''

        for k in range(self.n_layers):
            start_table_learn_t = self.layer_table_learn_time_ranges[k][0]
            stop_table_learn_t = self.layer_table_learn_time_ranges[k][1]

            start_prediction_learn_t = self.layer_prediction_learn_time_ranges[k][0]
            stop_prediction_learn_t = self.layer_prediction_learn_time_ranges[k][1]

            learn_table = self.learning_enabled and (start_table_learn_t < self.t < stop_table_learn_t)
            learn_prediction = self.learning_enabled and (start_prediction_learn_t < self.t < stop_prediction_learn_t)

            # TODO make this its own learning time, but for now, assuming weights learned at same time as predictions
            learn_weights = self.learning_enabled and (start_prediction_learn_t < self.t < stop_prediction_learn_t)

            tiles_NxN = self.tiles_per_layer_NxN[k]

            if k == 0:
                # how to get for each tile the input_state from raycast_image.flatten()?
                # here we need to get multiple input vectors, one for each tile from raycast_image, and process that in parallel
                #   in the cuda table lookup with one parallel lookup (one dot product)

                tile_input_states_mat = []
                grid = raycast_image.shape[0] / tiles_NxN

                for r in range(tiles_NxN):
                    for c in range(tiles_NxN):

                        r0 = int(r * grid)
                        r1 = int((r + 1) * grid)

                        c0 = int(c * grid)
                        c1 = int((c + 1) * grid)

                        if self.color_enabled:
                            tile_input_vect = raycast_image[r0:r1, c0:c1, :].flatten()
                        else:
                            tile_input_vect = raycast_image[r0:r1, c0:c1].flatten()

                        tile_input_states_mat.append(tile_input_vect)

                tile_input_states_mat = np.ascontiguousarray(np.array(tile_input_states_mat, np.float32))

            else:
                # TODO later must be from previous layer
                tile_input_states_mat = None

            I_indices_t, dists = self._lookup_and_learn_table(layer_n=k,
                                                              cuda_table_I=self.tables[k],
                                                              input_states_mat=tile_input_states_mat,
                                                              learn_table=learn_table,
                                                              input_x_y_theta=input_x_y_theta)

            if k == 0:  # first layer
                # print('I_indices_t: ', I_indices_t.shape)  # I_indices_t:  (1024,)
                assert self.tiles_per_layer_NxN[0] * self.tiles_per_layer_NxN[0] == I_indices_t.shape[0]

                self.I_history.store_new_states(newest_states_list=[I_indices_t], extra_data_list=[None])

            if learn_prediction:
                self._learn_predictions()

            if learn_weights:
                self._learn_weights(input_states_mat=tile_input_states_mat, dists=dists)

        motor_out = None

        self.t += 1
        return motor_out

    def _learn_weights(self, input_states_mat, dists):
        '''

        :param input_states_mat: current set of input images, one image per tile
        :return:

        selection im:

        [ tile 1 sample ] -> top k rows in table: [ ][ ][ ][ ][ ]
        [ tile 2 sample ] -> top k rows in table: [ ][ ][ ][ ][ ]
        [ tile 3 sample ] -> top k rows in table: [ ][ ][ ][ ][ ]
        [ tile 4 sample ] -> top k rows in table: [ ][ ][ ][ ][ ]

        '''

        # print('dists.shape', dists.shape)  # dists.shape (256, 6000)  for a total of 256 tiles of input, and 6000 entries. dist for each tile input to each table row
        # this is very slow but needed for making multiple selections of rows, per input tile
        # TODO this is very slow, needs to be sped up or done on GPU, etc
        sorted_dists_indices = np.argsort(dists, axis=1)  # (256, 6000)

        weights = self.tables[0].weight_masks

        tile_r_c = int(sqrt(weights.shape[1]))

        # leave buffer of 1 pixel between images, leave room for input images

        select_im_top_k = 5  # how many of the top selected rows to display for each input tile
        num_to_disp = 32  # input_states_mat.shape[0]  # this is a lot!
        disp_list = list(np.random.permutation(input_states_mat.shape[0])[0:num_to_disp])
        curr_disp_ind = 0

        selection_im = np.zeros((tile_r_c * num_to_disp + num_to_disp, tile_r_c * (1 + select_im_top_k) + 2 * select_im_top_k + 4, 3), np.float32)

        # this is the currently selected row index of the table, one row index per tile
        current_I_index_arr, _ = self.I_history.get_state(state_index=0, delay=0)

        tile_im_false_color = np.zeros((tile_r_c, tile_r_c, 3), np.float32)

        # for each input_state in input_states_list

        r_offset = 0

        for k in range(input_states_mat.shape[0]):
            #   currently: weight is (1.0 - average_pixel_error)
            #       lower average error per pixel (row - input) --> higher weight
            #       higher average error per pixel (row - input) --> lower weight
            #   new, if learning a sequence:
            #       if error low for previous selected row -> don't allow higher weight (?)
            #       or perhaps: error for row is max(error for that pixel for previous selected rows, and current error) or something like this...
            #           (error is per_pixel_dist)
            #           or min or something like that
            # minimum distance of previous selected rows. use max(1.0-prev_min_dist, computed dist) for each row, as dist to compute new weights

            tile_input_vect = input_states_mat[k, :]
            dist_bias = np.ones_like(tile_input_vect)

            for k2 in range(select_im_top_k):
                table_row_index = int(sorted_dists_indices[k, k2])

                table_row, _, _ = self.tables[0].get_matrix_row(row_index=table_row_index)

                per_pixel_dist_row = np.abs(tile_input_vect - table_row)

                per_pixel_dist_weight_learn = np.maximum(per_pixel_dist_row, 1.0 - dist_bias)

                dist_bias = np.minimum(per_pixel_dist_row, dist_bias)

                self.weight_error_sums[0][table_row_index, :] = self.weight_error_sums[0][table_row_index, :] + per_pixel_dist_weight_learn
                self.weight_error_counts[0][table_row_index, :] = self.weight_error_counts[0][table_row_index, :] + 1

                self.tables[0].set_row_weights(row_index=table_row_index,
                                               weights=1.0 - np.divide(self.weight_error_sums[0][table_row_index, :], self.weight_error_counts[0][table_row_index, :]),
                                               row_values=table_row,
                                               fast_set_need_commit=True)  # needs row values to set wsquared*table

            if self.enable_select_im and k in disp_list:

                r0 = curr_disp_ind * tile_r_c + r_offset
                r1 = (curr_disp_ind + 1) * tile_r_c + r_offset
                c00 = 0
                c01 = tile_r_c

                # print(tile_input_vect.shape, tile_r_c)
                tmp0 = tile_input_vect.reshape((tile_r_c, tile_r_c))
                # print(selection_im.shape, r0, r1, c00, c01, tmp0.shape)
                selection_im[r0:r1, c00:c01, 0] = tmp0[:, :]
                selection_im[r0:r1, c00:c01, 1] = tmp0[:, :]
                selection_im[r0:r1, c00:c01, 2] = tmp0[:, :]

                for k2 in range(select_im_top_k):

                    # TODO generalize c indices below, given k2
                    c10 = int((k2 + 1) * (tile_r_c + 2) + 4)
                    c11 = int(c10 + tile_r_c)

                    # c10 = tile_r_c + 2
                    # c11 = tile_r_c + 2 + tile_r_c
                    # c20 = tile_r_c + 2 + tile_r_c + 2
                    # c21 = tile_r_c + 2 + tile_r_c + 2 + tile_r_c

                    table_row_index = int(sorted_dists_indices[k, k2])

                    table_row, _, _ = self.tables[0].get_matrix_row(row_index=table_row_index)

                    tile_row_im = table_row.reshape((tile_r_c, tile_r_c))

                    tile_weights = weights[table_row_index, :].reshape((tile_r_c, tile_r_c))

                    # selection_im[r0:r1, c10:c11] = tile_row_im[:, :]  # tile_weights[:, :]

                    tmp_r_n, tmp_c_n = np.nonzero(tile_weights < 0.9/(tile_r_c*tile_r_c))  # 0.5
                    # tmp_r_p, tmp_c_p = np.nonzero(tile_weights > 0.5)  # 0.5

                    # FALSE COLOR
                    tile_im_false_color[:, :, 0] = tile_row_im[:, :]
                    tile_im_false_color[:, :, 1] = tile_row_im[:, :]
                    tile_im_false_color[:, :, 2] = tile_row_im[:, :]

                    tile_im_false_color[tmp_r_n, tmp_c_n, 1] = 0.0
                    tile_im_false_color[tmp_r_n, tmp_c_n, 0] = tile_row_im[tmp_r_n, tmp_c_n]
                    tile_im_false_color[tmp_r_n, tmp_c_n, 2] = 0.0

                    selection_im[r0:r1, c10:c11, :] = tile_im_false_color[:, :, :]

                r_offset += 1

                curr_disp_ind += 1

        self.selection_im = selection_im
        self.tables[0].commit_weights_changes()
        # if self.enable_samples_by_row_im:
        #     for k in range(num_to_disp):  # this could also be a random subset instead of first N
        #         table_row, _, _ = self.tables[0].get_matrix_row(row_index=k)

    # @profile
    def _learn_predictions(self):

        # *** self.W learning ***

        # current_I_index_arr: 1024-length (N-tiles X N-tiles) of indices (int)
        current_I_index_arr, _ = self.I_history.get_state(state_index=0, delay=0)

        to_indices = current_I_index_arr.astype(np.int)

        prediction_type = 0  # 0, 1

        if prediction_type == 0:
            prev_I_index_arr, _ = self.I_history.get_state(state_index=0, delay=1)
            from_indices = prev_I_index_arr.astype(np.int)

            # use to_indices, from_indices
            for tile_index in range(to_indices.shape[0]):
                to_index = to_indices[tile_index]
                from_index = from_indices[tile_index]

                if self.W_count_by_layer[0][to_index, from_index] == 0:
                    self.nz_by_layer_row[0][from_index].append(to_index)

                self.W_count_by_layer[0][to_index, from_index] += 1
                self.S_count_by_layer[0][from_index] += 1

                # THIS IS STILL VERY SLOW- NEEDS TO BE STORED AND ITERATED INSTEAD OF COMPUTED EACH TIME
                # tmp_ind = np.nonzero(self.W_count_by_layer[0][:, from_index])[0]
                tmp_ind = self.nz_by_layer_row[0][from_index]

                self.W_by_layer[0][tmp_ind, from_index] = self.W_count_by_layer[0][tmp_ind, from_index] / self.S_count_by_layer[0][from_index]

            # super slow, instead we put above. Take this out when confirmed:
            # W_prob = self.W_count_by_layer[0] * 1.0 / self.S_count_by_layer[0]
            # use W_by_layer to represent the result (W_prob)
            # self.W_by_layer[0] = W_prob

            do_analysis = (time.time() - self.last_analysis_time > 30)
            if do_analysis:
                print()
                print('Prediction Analysis:')
                print()

                # clustering
                # top K per row, above threshold, target number of labels, etc.

                save_W_prob(filename='tmp.txt', nz=self.nz_by_layer_row[0], W=self.W_by_layer[0])

                # debug
                # nz = self.nz_by_layer_row[0]
                # for k in range(len(nz)):
                #     print('    ', k, len(nz[k]))  # , self.W_by_layer[0][nz[k], k])

                print()
                self.last_analysis_time = time.time()

        elif prediction_type == 1:
            # to do later speed up this function
            from_indices = self.I_history.get_state_sequence(state_index=0,
                                                             delay_long=self.max_trace_time,
                                                             delay_short=1).astype(np.int)

            # minimal possible trace value in self.W is = pow(self.trace_tau, self.max_trace_time)
            trace_value_arr = np.power(self.trace_tau, np.arange(self.max_trace_time - 1, -1, -1))

            # for 8x8 == 64 tiles, and max trace time of 100:
            #   to_indices: (64,) int64
            #   from_indices: (100, 64) int64
            #   trace_value_arr: (100,) float64

            # TODO speed up with cython when needed
            for tile_index in range(to_indices.shape[0]):
                to_index = to_indices[tile_index]
                from_index_arr = from_indices[:, tile_index]
                self.W_by_layer[0][to_index, from_index_arr] = np.maximum(trace_value_arr, self.W_by_layer[0][to_index, from_index_arr])
        else:
            assert False, 'Error! Invalid prediction_type: ' + str(prediction_type)

    # @profile
    def _lookup_and_learn_table(self, layer_n, cuda_table_I, input_states_mat, learn_table, input_x_y_theta):
        '''

        why is lookup and learn one function?
            if they were separate, we would have to do same lookup twice, or pass info in an awkward way

        :param input_state:
        :param learn_table:
        :return: I (sparse vector, one-hot)
        '''

        dists, argmin_dists = cuda_table_I.query_multiple_rows(query_inputs=input_states_mat)

        I_indices = argmin_dists  # needed? #np.argmin(dists)

        if learn_table:
            self._learn_table(layer_n=layer_n,
                              cuda_table_I=cuda_table_I,
                              dists=dists,
                              argmin_dists=argmin_dists,
                              input_states_mat=input_states_mat,
                              input_x_y_theta=input_x_y_theta)

        return I_indices, dists  # learning might have invalidated this; doesn't matter for now because for now we are not using lookup if learning

    def _learn_table(self, layer_n, cuda_table_I, dists, argmin_dists, input_states_mat, input_x_y_theta):
        '''

        :param cuda_table_I:
                    # (8000 x 192)
        :param dists:
                    distances from query vectors (input_states_mat) to each of current table rows
                    dists_arr = np.array(dists)
                    # (16, 8000)
        :param input_states_mat:
                    query vectors to insert into the table
                    # (16, 192)

        :param input_x_y_theta:
        :return:
        '''

        # to do later on replacement or learning: should zero out W from that row for all predictions
        #   why don't we do this now?
        #   because, for now, we learn table, then we leave it alone when learning predictions later

        row_replaced = False
        dists_arr = dists

        # print()
        # print(dists.shape)  # (16, 8000) or now (1024, 8000)  ---  note: 16=4x4, 1024=32x32
        # (4096, 8000)

        # print(input_states_mat.shape)  # (16, 192) or now (1024, 12)  --- note: 12=2x2x3 (per tile)
        # print()

        # after improvement to get 4x4 pixel tiles at higher resolution image:s
        # dists_arr.shape   (1024, 8000)
        # input_states_mat.shape    (1024, 48)

        # This was extremely inefficient:
        # sorted_dist_indices = np.argsort(dists_arr)  # consuming 64% of processing of this function
        ## new min ind: for each candidate row, what is "min dist" index (0->8k) in current table?
        # new_min_inds = sorted_dist_indices[:, 0].flatten()  # length 16
        ##  [ 2  0 25  9  2  0 25  9  2  0 25  9 14 12 29 13]

        # better way:
        new_min_inds = argmin_dists  # np.argmin(dists_arr, axis=1)

        new_min_dists = dists_arr[range(self.tiles_per_layer_NxN[layer_n] * self.tiles_per_layer_NxN[layer_n]), new_min_inds]  # length 16

        if self.init_I_row_num[layer_n] is None:
            self.init_I_row_num[layer_n] = 0

        init_already_done = True
        for k in range(len(new_min_dists)):
            if self.init_I_row_num[layer_n] < cuda_table_I.get_num_rows():
                # print('init row:', self.init_I_row_num[layer_n])
                init_already_done = False

                input_state_tmp = input_states_mat[k, :].flatten().astype(np.float32)
                # inefficient: another query to get updated dists
                dists_tmp = cuda_table_I.query(query_input=input_state_tmp)
                dists_tmp[self.init_I_row_num[layer_n]] = np.inf

                cuda_table_I.set_matrix_row(row_index=self.init_I_row_num[layer_n],
                                            row_input=input_state_tmp,
                                            row_to_table_dists=dists_tmp,
                                            fast_init=True)

                row_replaced = True
                self.init_I_row_num[layer_n] = self.init_I_row_num[layer_n] + 1

        if init_already_done:
            if not cuda_table_I.post_init_done:

                print()
                print('Table init done! doing post-init...')
                print()

                cuda_table_I.post_init()


            table_min_dist, table_min_dist_r, table_min_dist_c = cuda_table_I.get_min_dist()

            #if table_min_dist_r < 4000:
            #    print('**********************', table_min_dist, table_min_dist_r, table_min_dist_c)

            # for now, just replace one row max on every time step. whatever the max dist is across all.

            tmp = np.argmax(new_min_dists)
            row_replace_candidate = input_states_mat[tmp, :]
            new_min_dist = new_min_dists[tmp]
            dists_tmp = dists_arr[tmp, :].flatten()

            if new_min_dist > table_min_dist:
                # minimum distance of new row to current rows is greater than current minimum row-row distance
                # so: replace one row of current minimum, with new row

                # get one of the row indices of current minimum dist pair
                r_r_ind = table_min_dist_r  # could be table_min_dist_c

                dists_tmp[r_r_ind] = np.inf

                # replace the current min dist row, with the new row
                cuda_table_I.set_matrix_row(row_index=r_r_ind,
                                            row_input=row_replace_candidate,
                                            row_to_table_dists=dists_tmp)

                #if input_x_y_theta is not None:
                #    self.entries_x_y_theta_input[r_r_ind, :] = input_x_y_theta[:]

                row_replaced = True

        return row_replaced

    def disable_learning(self):
        self.learning_enabled = False

    def get_current_plan(self):
        return [], None

    def get_table_ims(self):
        '''

        :return:
        '''

        cuda_table = self.tables[0]
        table = np.transpose(cuda_table.get_table_from_gpu())
        weights = self.tables[0].weight_masks

        # 31 for 1000
        # 63 for 4000
        N = 31  # display NxN tiles of 8k entries
        entries = N*N

        rows = table[0:entries, :]
        # print(rows.shape, rows.dtype, np.amin(rows), np.amax(rows))
        # (16, 192) float32 0.0 0.923078

        tile_n = 0

        if self.color_enabled:
            tile_r_c = int(sqrt(rows.shape[1] / 3))
            table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1, 3)) + 0.5
            tile_weights_color = np.zeros((tile_r_c, tile_r_c, 3), np.float32)
        else:
            tile_r_c = int(sqrt(rows.shape[1]))
            table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

            if self.enable_separate_weights_im:
                weights_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1)) + 0.5

            # false color:
            table_im_false_color = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1, 3)) + 0.5
            tile_im_false_color = np.zeros((tile_r_c, tile_r_c, 3), np.float32)

        r_offset = 0
        for disp_r in range(N):

            c_offset = 0
            for disp_c in range(N):
                r0 = disp_r * tile_r_c + r_offset
                r1 = (disp_r + 1) * tile_r_c + r_offset
                c0 = disp_c * tile_r_c + c_offset
                c1 = (disp_c + 1) * tile_r_c + c_offset

                if self.color_enabled:
                    tile_im = table[tile_n, :].reshape((tile_r_c, tile_r_c, 3))
                    tile_weights = weights[tile_n, :].reshape((tile_r_c, tile_r_c))
                    tile_weights_color[:, :, 0] = tile_weights[:, :]
                    tile_weights_color[:, :, 1] = tile_weights[:, :]
                    tile_weights_color[:, :, 2] = tile_weights[:, :]

                    table_im[r0:r1, c0:c1, :] = np.multiply(tile_weights_color, tile_im)
                    # table_im[r0:r1, c0:c1, :] = tile_im
                else:
                    # SEPARATE TABLE IM, WEIGHTS IM
                    tile_im = table[tile_n, :].reshape((tile_r_c, tile_r_c))
                    tile_weights = weights[tile_n, :].reshape((tile_r_c, tile_r_c))

                    if self.enable_separate_weights_im:
                        table_im[r0:r1, c0:c1] = tile_im[:, :]
                        weights_im[r0:r1, c0:c1] = tile_weights[:, :]
                    else:
                        # not very clear, just grayscale multiply:
                        # table_im[r0:r1, c0:c1] = np.multiply(np.multiply(tile_weights, tile_weights), tile_im)

                        #
                        tmp_r_n, tmp_c_n = np.nonzero(tile_weights < 0.75/(tile_r_c*tile_r_c))  # 0.5
                        # tmp_r_p, tmp_c_p = np.nonzero(tile_weights > 0.5)  # 0.5

                        # FALSE COLOR
                        tile_im_false_color[:, :, 0] = tile_im[:, :]
                        tile_im_false_color[:, :, 1] = tile_im[:, :]
                        tile_im_false_color[:, :, 2] = tile_im[:, :]

                        #tile_im_false_color[tmp_r_p, tmp_c_p, 0] = tile_im[tmp_r_p, tmp_c_p]
                        #tile_im_false_color[tmp_r_n, tmp_c_n, 1] = tile_im[tmp_r_n, tmp_c_n]
                        #tile_im_false_color[:, :, 2] = 0.0

                        # what we want
                        # W=1: [1] = tile_im, [0, 2] = 0
                        # W=0: [0] = tile_im, [1, 2] = 0

                        # tile_im_false_color[:, :, 0] = 0
                        # tile_im_false_color[:, :, 1] = np.multiply(np.multiply(tile_weights, tile_weights), tile_im)
                        # tile_im_false_color[:, :, 2] = np.multiply(np.multiply(1.0 - tile_weights, 1.0 - tile_weights), tile_im)

                        tile_im_false_color[tmp_r_n, tmp_c_n, 1] = 0.0
                        tile_im_false_color[tmp_r_n, tmp_c_n, 0] = tile_im[tmp_r_n, tmp_c_n]
                        tile_im_false_color[tmp_r_n, tmp_c_n, 2] = 0.0

                        table_im_false_color[r0:r1, c0:c1, :] = tile_im_false_color[:, :, :]

                tile_n += 1

                c_offset += 1

            r_offset += 1

        # false color
        table_im = table_im_false_color

        max_dim = max(table_im.shape[0], table_im.shape[1])
        imscale = self.table_ims_scale / max_dim  # 0.2: full table, 2.0
        table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        if self.enable_separate_weights_im:
            max_dim = max(weights_im.shape[0], weights_im.shape[1])
            imscale = self.table_ims_scale / max_dim  # 0.2: full table, 2.0
            weights_im = cv2.resize(weights_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)
            # weights_im = None
        else:
            weights_im = None

        if self.selection_im is not None and self.enable_select_im:
            select_im = self.selection_im
            max_dim = max(select_im.shape[0], select_im.shape[1])
            imscale = self.table_ims_scale / max_dim  # 0.2: full table, 2.0
            select_im = cv2.resize(select_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)
            other_im = select_im
        else:
            select_im = None

        if self.enable_W_im:
            W_im = (self.W_by_layer[0] * 255.0).astype(np.uint8)
            max_dim = max(W_im.shape[0], W_im.shape[1])

            imscale = self.table_ims_scale / max_dim  # 0.2: full table, 2.0
            # imscale = 5.0
            W_im = cv2.resize(W_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)
            other_im = W_im
        else:
            W_im = None

        # TODO make flags!
        # table_im = None
        # weights_im = None

        return [table_im, W_im, weights_im, select_im], ['tile', 'W', 'weights', 'select']


def build_checkerboard(w, h):
    re = np.r_[w * [0, 1]]  # even-numbered rows
    ro = np.r_[w * [1, 0]]  # odd-numbered rows
    return np.row_stack(h * (re, ro))


class SingleLayerSharedTile(object):
    def __init__(self):
        '''
            stores CudaTable for tile
            handles reuse of same tile across one layer for learning and action
        '''



# **************** OLD ****************

class TiledTrace(object):
    '''

    should this be a collection of SingleLayerTrace objects?
    or is there an advantage to folding functionality into one class i.e. parallelization?

    for simplicity, what if we leave it for now as a collection of SingleLayerTrace objects
    we need to tile in each layer, and then have multiple layers

    the problem is: this will get very slow

    how many tiles of what size?

    current: 8000 entries for a (32 * 3) = 96-dim input for a 1-D color image
    so approximately: entries = 100 * dim

    let's say we operate on a 60x40 monochrome image
    60 * 40 = 2400-dim input

    how to tile for 100-dim input predictors?
    6 tiles by 4 tiles = 24 tiles total in first layer

    24 tiles of (10x10) monochrome pixels: 100-dim inputs

    --- but for testing in virtual world: need color inputs!


    '''

    def __init__(self):
        '''

        '''

