import time
import cv2
import numpy as np
import random
import pickle

from cuda_dist_query import CudaTable
from brain_components_classes.mknn_cuda_table import MKNNCudaTable
from brain_components_classes.multi_sparse_binary_knn import MultiSparseBinaryKNN

from utils.w_save_load_helper import save_W_prob
from math import sqrt, sin, cos
from utils.load_sim_history import load_states_history, load_td_info_history
from utils.fps_counter import FPSCounter

from utils.one_time_messages import OneTimeMessages


class MultiLayerDynamicTiles(object):
    def __init__(self, params):
        '''

        :param params:
        {

    # **************************** TABLE ****************************

            'max_history_length': MAX_HISTORY_LENGTH
            'ims_scale_pixels': 1200
            'image_dim_NxN_pixels': IM_DIM: 128
            'tile_dim_NxN_pixels_per_layer': TILE_DIM:       [3, 6, 12, 24, 48, 96, 128]
            'tiles_offset_N_pixels_per_layer': TILES_OFFSET: [2, 5, 11, 23, 47, 95, None]
            'rows_per_tile_per_layer': ROWS_PER_TILE: [1600, 1600, 1600, 1600, 1600, 1600, 1600]
            'table_learn_time_multiple': TABLE_LEARN_TIME_MULTIPLE: 4

    # **************************** PREDICTION ****************************

            'learning_off_time': TRAIN_STEPS: ROWS_PER_TILE * TABLE_LEARN_TIME_MULTIPLE + PREDICTION_LEARN_TIME
            'lateral_predictor_radius_N_pixels_per_layer': PREDICT_RADIUS: [0, 0, 0, 0, 0, 0, 0]
            'lateral_predictor_delay_list_ff_per_layer': INPUT_DELAY_LIST: [0, 0, 0, 0, 0, 0, 0]
            'predict_steps_ahead_per_layer': PREDICT_STEPS: [1, 1, 1, 1, 1, 1, 1]
        }

        As a starting point, we will just have tables and look at representations per layer (no prediction yet)

        '''

        self.otm = OneTimeMessages()

        self.n_layers = len(params['tile_dim_NxN_pixels_per_layer'])
        assert len(params['tile_dim_NxN_pixels_per_layer']) == len(params['tiles_offset_N_pixels_per_layer'])
        assert len(params['tile_dim_NxN_pixels_per_layer']) == len(params['rows_per_tile_per_layer'])

        self.max_history_length = params['max_history_length']
        self.ims_scale_pixels = float(params['ims_scale_pixels'])  # visualizer

        self.image_dim_NxN_pixels = params['image_dim_NxN_pixels']

        self.tile_dim_NxN_pixels_per_layer = np.array(params['tile_dim_NxN_pixels_per_layer'])
        self.tiles_offset_N_pixels_per_layer = np.array(params['tiles_offset_N_pixels_per_layer'])
        self.rows_per_tile_per_layer = np.array(params['rows_per_tile_per_layer'])

        self.table_learn_time_multiple = params['table_learn_time_multiple']

        # initialize cuda table, or mknn-cuda-table, per layer

        self.tables = []

        self.num_tiles_by_layer = []
        self.input_dim_by_layer = []

        self.table_learn_time_start_by_layer = []
        self.table_learn_time_end_by_layer = []

        self.pre_tile_indices_per_tile_by_layer = []

        self.num_tiles_NxN_by_layer = []

        self.center_tile_r = []  # used internally by _compute_layer_inputs
        self.center_tile_c = []  # used internally by _compute_layer_inputs

        self.init_I_row_num = np.zeros(self.n_layers, np.int)

        time_tmp = 0
        for layer_n in range(self.n_layers):

            num_entries = self.rows_per_tile_per_layer[layer_n]

            # input_dim:
            #   size of input for this layer's table; i.e. size of vector stored in the table
            # input_tile_r_indices_per_output_tile, input_tile_c_indices_per_output_tile:
            #   for each tile in this layer, what are input tile indices that make up its vector
            #   for input 0, this is image pixel indices; i.e. im_r_indices, im_c_indices in dynamic_tiles.py

            input_dim, num_tiles, layer_inputs, num_tiles_NxN = self._compute_layer_inputs(layer_n=layer_n)

            self.num_tiles_NxN_by_layer.append(num_tiles_NxN)

            if layer_n == 0:
                input_tile_r_indices, input_tile_c_indices = layer_inputs

                self.im_r_indices = input_tile_r_indices  # shape: (num_tiles_layer_n, input_dim)
                self.im_c_indices = input_tile_c_indices  # shape: (num_tiles_layer_n, input_dim)

                assert self.im_r_indices.shape[0] == num_tiles
                assert self.im_r_indices.shape[1] == input_dim
                assert self.im_c_indices.shape[1] == input_dim

                self.pre_tile_indices_per_tile_by_layer.append(None)

            else:

                assert layer_inputs.shape[0] == num_tiles
                assert layer_inputs.shape[1] == input_dim

                self.pre_tile_indices_per_tile_by_layer.append(layer_inputs)

            if layer_n == 0:
                self.tables.append(CudaTable(num_entries=num_entries,
                                             input_dim=input_dim,
                                             disable_row_row_dist=False,
                                             enable_weight_bias=False))
            else:
                pass

                # p = {
                #     'use_cuda': False,  # not implemented!
                #     'use_cython': True,
                #     'sparse_io_dim': self.rows_per_tile_per_layer[layer_n - 1],  # rows per tile of cuda tables
                #     'num_sparse_inputs': input_dim,
                #     'num_sparse_outputs': 1,  # not actually used, we care only about row (not any associated output)
                #     'num_rows': num_entries,
                #     'num_knn': 1,
                #     'random_init': False  # for debugging
                # }
                # self.tables.append(MultiSparseBinaryKNN(params=p))

                # TODO problems with above:
                #   MultiSparseBinaryKNN:
                #       does not implement query_multiple_rows
                #       does not implement distance helper / seq-nn
                #       assumes an output (prediction) is also trained

                self.tables.append(MKNNCudaTable(num_tiles=num_tiles,  # num tables
                                                 num_entries=num_entries,  # num entries per table
                                                 input_dim=input_dim,  # number of sub-tiles composing a tile; typically 4
                                                 sparse_io_dim=self.rows_per_tile_per_layer[layer_n - 1]  # max value per element of input; rows per tile of prev layer cuda tables
                                                 ))

                # TODO problems with above:
                #   MKNNCUdaTable needs to have new cuda-based logic to handle sparse binary input;
                #   cannot just be a wrapper
                #   Also: optional: need to implement a way for each tile to have independent tables instead
                #       CudaTable even for layer 0 does not support this!

            self.num_tiles_by_layer.append(num_tiles)
            self.input_dim_by_layer.append(input_dim)

            table_learn_start = time_tmp
            table_learn_end = time_tmp + num_entries * self.table_learn_time_multiple

            self.table_learn_time_start_by_layer.append(table_learn_start)
            self.table_learn_time_end_by_layer.append(table_learn_end)

            time_tmp = table_learn_end

        self.num_row_swaps = 0
        self.last_row_swap_disp = 0
        self.row_swap_disp_time = 1

        self.t = 0

    def _compute_layer_inputs(self, layer_n):
        '''

        # compute tiling: which indices from lower layers go as inputs to the tables

        'image_dim_NxN_pixels': IM_DIM: 128
        'tile_dim_NxN_pixels_per_layer': TILE_DIM:       [2, 4, 8, 16, 32, 64, 128]
        'tiles_offset_N_pixels_per_layer': TILES_OFFSET: [1, 2, 4, 8,  16, 32, None]

        Assume: offset always half of dim, dim always factor of two

        :param layer_n:
        :return:
        '''


        # what happens if we assume:
        #   tile dim is factors of two
        #   image dim is factor of two
        #   tile offset is always half of tile dim

        tile_dim_NxN_pixels = self.tile_dim_NxN_pixels_per_layer[layer_n]
        tiles_offset_N_pixels = self.tiles_offset_N_pixels_per_layer[layer_n]

        assert tile_dim_NxN_pixels == tiles_offset_N_pixels, 'Overlap not supported! offset must be same as dim' + str(tiles_offset_N_pixels, tile_dim_NxN_pixels)

        # TODO we are here; re-writing for no overlap of tiles

        num_tiles_NxN = int(self.image_dim_NxN_pixels / tiles_offset_N_pixels)

        # adjust num_tiles_NxN so no incomplete tiles at end
        for tile_r_ in range(num_tiles_NxN):
            r_start = tile_r_ * tiles_offset_N_pixels
            r_end = r_start + tile_dim_NxN_pixels - 1
            if r_end > self.image_dim_NxN_pixels - 1:
                num_tiles_NxN -= 1
                assert False, 'Should never happen with no overlap'
                print(         'decreasing num_tiles_NxN:',
                               'layer_n', layer_n,
                               'num_tiles_NxN', num_tiles_NxN,
                               'tile_dim_NxN_pixels', tile_dim_NxN_pixels,
                               'tiles_offset_N_pixels', tiles_offset_N_pixels,
                               'self.image_dim_NxN_pixels', self.image_dim_NxN_pixels,
                               'r_end', r_end,
                               'self.image_dim_NxN_pixels - 1', self.image_dim_NxN_pixels - 1)

        num_tiles = num_tiles_NxN * num_tiles_NxN

        self.center_tile_r.append(np.zeros(num_tiles, np.float32))
        self.center_tile_c.append(np.zeros(num_tiles, np.float32))

        if layer_n == 0:
            num_pixels_per_tile = tile_dim_NxN_pixels * tile_dim_NxN_pixels
            im_r_indices = np.zeros((num_tiles, num_pixels_per_tile), np.int)
            im_c_indices = np.zeros((num_tiles, num_pixels_per_tile), np.int)
            layer_inputs = None  # unused
            input_dim = num_pixels_per_tile
        else:
            im_r_indices = None  # unused
            im_c_indices = None  # unused

            # input_dim = int((tile_dim_NxN_pixels / 2) * (tile_dim_NxN_pixels / 2))  # assumption: tiles are always twice as large as lower layer
            # input_dim = tile_dim_NxN_pixels * tile_dim_NxN_pixels  # TODO correct? because of overlap of 1 works for layer 1
            # layer_inputs = np.zeros((num_tiles, input_dim), np.int)

            # input_dim is hard to determine here
            input_dim = None
            layer_inputs = None

        for tile_r_ind in range(num_tiles_NxN):
            for tile_c_ind in range(num_tiles_NxN):
                tile_index = tile_r_ind * num_tiles_NxN + tile_c_ind
                # if tile_index % 100 == 0:
                # print('layer', layer_n, 'tile_index', tile_index, 'of', num_tiles_NxN*num_tiles_NxN)
                r_start = tile_r_ind * tiles_offset_N_pixels
                r_end = r_start + tile_dim_NxN_pixels

                c_start = tile_c_ind * tiles_offset_N_pixels
                c_end = c_start + tile_dim_NxN_pixels

                self.center_tile_r[layer_n][tile_index] = (r_start + r_end) / 2.0
                self.center_tile_c[layer_n][tile_index] = (c_start + c_end) / 2.0

                # manual meshgrid:
                r_indices = []
                c_indices = []
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        r_indices.append(r)
                        c_indices.append(c)

                r_indices = np.array(r_indices).astype(np.int)
                c_indices = np.array(c_indices).astype(np.int)

                if layer_n == 0:
                    im_r_indices[tile_index, :] = r_indices[:]
                    im_c_indices[tile_index, :] = c_indices[:]
                    layer_inputs = im_r_indices, im_c_indices
                else:
                    tile_dim_prev = self.tile_dim_NxN_pixels_per_layer[layer_n - 1]
                    input_dim = int((tile_dim_NxN_pixels / tile_dim_prev) * (tile_dim_NxN_pixels / tile_dim_prev))

                    if layer_inputs is None:
                        layer_inputs = np.zeros((num_tiles, input_dim), np.int)

                    # todo set layer_inputs

                    prev_center_rs = self.center_tile_r[layer_n - 1]
                    prev_center_cs = self.center_tile_c[layer_n - 1]

                    cond_1 = np.logical_and(prev_center_rs > r_start, prev_center_rs < r_end)
                    cond_2 = np.logical_and(prev_center_cs > c_start, prev_center_cs < c_end)

                    prev_indices_within_bounds = np.nonzero(np.logical_and(cond_1, cond_2))[0]

                    assert prev_indices_within_bounds.shape[0] == input_dim, 'input dim doesnt match: '  + str((input_dim, prev_indices_within_bounds.shape[0] )) + ' for layer ' + str(layer_n)
                    layer_inputs[tile_index, :] = prev_indices_within_bounds[:]

        # layer_n == 0: layer_inputs = (im_r_indices, im_c_indices)
        # else:         layer_inputs is one array
        return input_dim, num_tiles, layer_inputs, num_tiles_NxN


    def _verify_tiling(self, layer_n):
        '''

        display tiling and show input tiles from previous layer

        :param layer_n:
        :return:
        '''

        pass

    def step(self, raycast_image, input_state, input_x_y_theta, goal_context_state_learning, goal_context_state_task, last_motor_command):
        '''

        :param raycast_image:
        :param input_state:
        :param input_x_y_theta:
        :param goal_context_state_learning:
        :param goal_context_state_task:
        :param last_motor_command:
        :return:
        '''

        assert input_state is None
        assert input_x_y_theta is None
        assert goal_context_state_learning is None
        assert goal_context_state_task is None
        assert last_motor_command is None

        return self._step_actual(input_image=raycast_image)

    # @profile
    def _step_actual(self, input_image):
        '''

        :param input_image:
        :return:
        '''

        self.otm.print_once(msg='MultiLayerDynamicTiles\n'
                            '\n'
                            '   input_image.shape: ' + str(input_image.shape) + ''
                            '   input_image.dtype: ' + str(input_image.dtype) + ''
                            '   input_image min, max: ' + str((np.amin(input_image), np.amax(input_image))),
                            msg_hash='MultiLayerDynamicTiles-init')

        # finish this first, simplify computation, assumning:
        #   tile dim is factors of two
        #   image dim is factor of two
        #   tile offset is always half of tile dim

        prev_layer_activity = input_image
        for layer_n in range(self.n_layers):
            if self.t >= self.table_learn_time_start_by_layer[layer_n]:  # run this layer only if it is learning, or already learned
                # (1) get tile inputs from previous layer activity, or from pixels
                tile_input_states_mat = self._get_tile_inputs_for_layer(layer_n=layer_n, prev_layer_activity=prev_layer_activity)
                # (2) get

                assert tile_input_states_mat.shape[0] == self.num_tiles_by_layer[layer_n]
                assert tile_input_states_mat.shape[1] == self.input_dim_by_layer[layer_n]

                self.otm.print_once('    layer ' + str(layer_n) + ', tile_input_states_mat: ' + str(tile_input_states_mat.shape) + ' ' + str(tile_input_states_mat.dtype))

                dists, argmin_dists = self.tables[layer_n].query_multiple_rows(query_inputs=tile_input_states_mat)
                prev_layer_activity = argmin_dists.copy()

                assert argmin_dists.shape[0] == self.num_tiles_by_layer[layer_n]

                self.otm.print_once(('    layer ' + str(layer_n) + ', dists.shape: ', dists.shape))
                self.otm.print_once(('    layer ' + str(layer_n) + ', dists.dtype: ', dists.dtype))
                # this is slow; dists is large:
                # self.otm.print_once(('    layer ' + str(layer_n) + ', dists min, max: ', np.amin(dists), np.amax(dists)), msg_hash='layer ' + str(layer_n) + ', dists min, max')
                self.otm.print_once(('    layer ' + str(layer_n) + ', argmin_dists.shape: ', argmin_dists.shape))
                self.otm.print_once(('    layer ' + str(layer_n) + ', argmin_dists.dtype: ', argmin_dists.dtype))
                self.otm.print_once(('    layer ' + str(layer_n) + ', argmin_dists min, max: ', np.amin(argmin_dists), np.amax(argmin_dists)), msg_hash='layer ' + str(layer_n) + ', argmin_dists min, max')

                # table learn time condition
                # layers learn sequentially

                if self.table_learn_time_start_by_layer[layer_n] <= self.t < self.table_learn_time_end_by_layer[layer_n]:
                    self.otm.print_once('dynamic_tiles:: starting learning of table, layer: ' + str(layer_n))
                    if layer_n == 0:
                        # TODO incorporate this into CudaTable class as part of changes we need to make to it:
                        self._learn_table_layer_0(layer_n=layer_n,
                                                  cuda_table=self.tables[layer_n],
                                                  dists=dists,
                                                  argmin_dists=argmin_dists,
                                                  tile_input_states_mat=tile_input_states_mat,
                                                  input_x_y_theta=None)
                    else:
                        self.tables[layer_n].seq_nn_learn_last_query()
                else:
                    self.otm.print_once('dynamic_tiles:: finished learning of table, layer: ' + str(layer_n))

        self.t += 1

    def _get_tile_inputs_for_layer(self, layer_n, prev_layer_activity):
        if layer_n == 0:

            input_image = prev_layer_activity
            # im_r_indices: (num_tiles, input_dim), im_c_indices: (num_tiles, input_dim)
            #   so basically these are "pre_pixel_indices_per_tile", separated into r and c
            tile_input_states_mat = input_image[self.im_r_indices, self.im_c_indices]

        else:
            # prev_layer_activity: (i.e. argmin_dists of previous layer):
            #     length: number of tiles in previous layer

            # self.pre_tile_indices_by_tile: (num_tiles, input_dim)

            tile_input_states_mat = prev_layer_activity[self.pre_tile_indices_per_tile_by_layer[layer_n]]


        return tile_input_states_mat

    # @profile
    def _learn_table_layer_0(self, layer_n, cuda_table, dists, argmin_dists, tile_input_states_mat, input_x_y_theta):
        """

        :param cuda_table:
        :param dists:
        :param argmin_dists:
        :param tile_input_states_mat:
        :param input_x_y_theta:
        :return:
        """

        assert layer_n == 0, 'we only use this for layer 0'

        if not cuda_table.post_init_done:
            for k in range(tile_input_states_mat.shape[0]):

                if self.init_I_row_num[layer_n] < cuda_table.get_num_rows():

                    # this should be determined in a sophisticated way after init is done
                    # for now we assume we are only learning during init:
                    replacement_candidate_row_index = self.init_I_row_num[layer_n]

                    # get candidate row + weights we are adding to table
                    candidate_row = tile_input_states_mat[k, :]

                    dists_tmp = cuda_table.query(query_input=candidate_row)
                    dists_tmp[replacement_candidate_row_index] = np.inf

                    cuda_table.set_matrix_row(row_index=replacement_candidate_row_index,
                                              row_input=candidate_row,
                                              row_weights=None,
                                              row_to_table_dists=dists_tmp,
                                              fast_init=True)

                    self.init_I_row_num[layer_n] += 1
                else:
                    if not cuda_table.post_init_done:
                        print()
                        print('Table init done! doing post-init...')
                        print()

                    cuda_table.post_init()

        if cuda_table.post_init_done:

            table_min_dist, table_min_dist_r, table_min_dist_c = cuda_table.get_min_dist()

            # print(table_min_dist_r, table_min_dist_c)

            # from the 256 new inputs, choose the one with largest [minimum distance to table rows]
            min_dist_per_input_tile = dists[range(self.num_tiles_by_layer[layer_n]), argmin_dists]

            #print('. ', min_dist_per_input_tile)

            input_tile_index_with_max_min_table_dist = np.argmax(min_dist_per_input_tile)
            tmp = input_tile_index_with_max_min_table_dist

            candidate_row = tile_input_states_mat[tmp, :]
            new_min_dist = min_dist_per_input_tile[tmp]
            dists_tmp = dists[tmp, :].flatten()

            if new_min_dist > table_min_dist:
                self.num_row_swaps += 1
                # print('replacing ', new_min_dist, table_min_dist)
                dists_tmp[table_min_dist_r] = np.inf

                # replace the current min dist row, with the new row
                cuda_table.set_matrix_row(row_index=table_min_dist_r,
                                          row_input=candidate_row,
                                          row_to_table_dists=dists_tmp)
                # print('... ! replacing')
            else:
                # print('NOT replacing ', new_min_dist, table_min_dist)
                pass

            if time.time() - self.last_row_swap_disp > self.row_swap_disp_time:
                print('    num row swaps:', self.num_row_swaps)
                self.num_row_swaps  = 0
                self.last_row_swap_disp = time.time()


    def get_table_ims(self):
        '''

        :return:
        '''

        ims_list = []
        ims_names_list = []

        return ims_list, ims_names_list


























