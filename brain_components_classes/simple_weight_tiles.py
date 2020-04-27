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


class SimpleWeightTiles(object):
    def __init__(self, params):
        '''

        'color_enabled':                    False
        'enable_weight_bias':               True
        'input_dim':                        IM_PIXELS / (TILES_LAYER_0 * TILES_LAYER_0)
        'goal_context_dim':                 None
        'enable_learning':                  True
        'tile_entries_per_layer':           [TABLE_ENTRIES]
        'table_learn_time_per_layer':       [TABLE_LEARN_TIME]
        'prediction_learn_time_per_layer':  [PREDICTION_LEARN_TIME]
        'tiles_per_layer_NxN':              [TILES_LAYER_0]
        'pre_init_goal_contexts':           None
        'max_history_length':               MAX_HISTORY_LENGTH
        'table_ims_scale_pixels':           1600 or 2000

        :param params:


        Assumptions:
            always: color_enabled=False, enable_weight_bias=True, goal_context_dim=None, enable_learning=True
                (can ignore these flags)
            assume one layer only
            not doing prediction for now

        '''

        # *** assumptions ***

        assert params['color_enabled'] is False
        assert params['enable_weight_bias'] is True

        assert params['goal_context_dim'] is None
        assert params['enable_learning'] is True

        assert len(params['tile_entries_per_layer']) == 1
        assert len(params['table_learn_time_per_layer']) == 1
        assert len(params['prediction_learn_time_per_layer']) == 1
        assert len(params['tiles_per_layer_NxN']) == 1

        assert params['pre_init_goal_contexts'] is None

        # *** parameters ***

        self.input_dim = params['input_dim']

        self.num_entries = params['tile_entries_per_layer'][0]
        self.table_learn_time = params['table_learn_time_per_layer'][0]
        self.prediction_learn_time = params['prediction_learn_time_per_layer'][0]
        self.tiles_NxN = params['tiles_per_layer_NxN'][0]

        self.max_history_length = params['max_history_length']
        self.table_ims_scale = float(params['table_ims_scale_pixels'])

        print()
        print('Initializing NewSharedTiles...')
        print()
        print('    ' + 'input_dim: ', self.input_dim)
        print('    ' + 'entries: ', self.num_entries)
        print('    ' + 'table_learn_time: ', self.table_learn_time)
        print('    ' + 'prediction_learn_time: ', self.prediction_learn_time)
        print('    ' + 'tiles_NxN: ', self.tiles_NxN)
        print('    ' + 'max_history_length: ', self.max_history_length)
        print('    ' + 'table_ims_scale: ', self.table_ims_scale)
        print()

        # *** derived values and constants ***

        self.trace_tau = 0.9
        self.max_trace_time = 100  # length of keeping track of trace for W.
        # so, minimal possible trace value in self.W is = pow(self.trace_tau, self.max_trace_time)

        # this is per-tile
        self.I_history = StatesLimitedHistory(params={'max_delay': self.max_trace_time,
                                                      'states_dim_list': [self.tiles_NxN * self.tiles_NxN]})
        self.init_I_row_num = 0  # where we are in initializing the rows

        t_start_offset = 1  # 100 was what we used before, not sure why
        self.table_learn_time_start = t_start_offset
        self.table_learn_time_end = t_start_offset + self.table_learn_time

        self.prediction_learn_time_start = self.table_learn_time_end + 100
        self.prediction_learn_time_end = self.prediction_learn_time_start + self.prediction_learn_time

        self.table = CudaTable(num_entries=self.num_entries,
                               input_dim=self.input_dim,
                               disable_row_row_dist=False,  # TODO depending on learning rule, may need to re-enable!
                               enable_weight_bias=False)  # TODO set correctly; for now disable, only learn weight, don't apply

        print()
        print('Done Initializing NewSharedTiles.')
        print()

        # *** working variables ***

        self.im_r_indices = None
        self.im_c_indices = None

        self.t = 0

        self.stage_index = None  # what stage of learning are we in

        # *** debug / print variables ***

        self.printed_init_step = False
        self.last_select_im_data = None

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

        assert input_state is None
        assert input_x_y_theta is None
        assert goal_context_state_learning is None
        assert goal_context_state_task is None
        assert last_motor_command is None

        # ensure that dim of an entry == number of pixels in a tile == input image divided into tiles_NxN tiles
        dim_per_tile = int(raycast_image.shape[0] / self.tiles_NxN)
        assert dim_per_tile * dim_per_tile == self.input_dim

        if not self.printed_init_step:
            print()
            print('NewSharedTiles.step: ', self.t)
            print()
            print('    raycast_image.shape: ', raycast_image.shape)
            print('    raycast_image.dtype: ', raycast_image.dtype)
            print('    raycast_image min, max: ', np.amin(raycast_image), np.amax(raycast_image))

        tile_input_states_mat = self._get_tile_inputs_for_image(image=raycast_image)
        assert tile_input_states_mat.shape[0] == self.tiles_NxN * self.tiles_NxN
        assert tile_input_states_mat.shape[1] == self.input_dim

        if not self.printed_init_step:
            print('    tile_input_states_mat.shape: ', tile_input_states_mat.shape)
            print('    tile_input_states_mat.dtype: ', tile_input_states_mat.dtype)
            print('    tile_input_states_mat min, max: ', np.amin(tile_input_states_mat), np.amax(tile_input_states_mat))

        dists, argmin_dists = self.table.query_multiple_rows(query_inputs=tile_input_states_mat)
        I_indices = argmin_dists

        if not self.printed_init_step:
            print('    dists.shape: ', dists.shape)
            print('    dists.dtype: ', dists.dtype)
            print('    dists min, max: ', np.amin(dists), np.amax(dists))
            print('    argmin_dists.shape: ', argmin_dists.shape)
            print('    argmin_dists.dtype: ', argmin_dists.dtype)
            print('    argmin_dists min, max: ', np.amin(argmin_dists), np.amax(argmin_dists))

        if self.table_learn_time_start < self.t < self.table_learn_time_end:
            if self.stage_index is None:
                print()
                print('Starting: _learn_table')
                print()
                self.stage_index = 0

            self._learn_table(cuda_table=self.table,
                              dists=dists,
                              argmin_dists=argmin_dists,
                              tile_input_states_mat=tile_input_states_mat,
                              input_x_y_theta=input_x_y_theta)

        if self.prediction_learn_time_start < self.t < self.prediction_learn_time_end:
            # for now, we learn weights during prediction time
            # hypothesis: predictions + weights learn together

            if self.stage_index == 0:
                print()
                print('Starting: _learn_weights')
                print()
                self.stage_index = 1

            self._learn_weights(cuda_table=self.table,
                                dists=dists,
                                argmin_dists=argmin_dists,
                                tile_input_states_mat=tile_input_states_mat,
                                input_x_y_theta=input_x_y_theta)

        if not self.printed_init_step:
            print()

        if self.t > 3:
            self.printed_init_step = True

        self.t += 1
        return None

    def _learn_table(self, cuda_table, dists, argmin_dists, tile_input_states_mat, input_x_y_theta):
        """

        :param cuda_table:
        :param dists:
        :param argmin_dists:
        :param tile_input_states_mat:
        :param input_x_y_theta:
        :return:
        """

        for k in range(tile_input_states_mat.shape[0]):
            if self.init_I_row_num < cuda_table.get_num_rows():

                # this should be determined in a sophisticated way after init is done
                # for now we assume we are only learning during init:
                replacement_candidate_row_index = self.init_I_row_num

                # get candidate row + weights we are adding to table
                candidate_row = tile_input_states_mat[k, :]

                dists_tmp = cuda_table.query(query_input=candidate_row)
                dists_tmp[replacement_candidate_row_index] = np.inf

                cuda_table.set_matrix_row(row_index=replacement_candidate_row_index,
                                          row_input=candidate_row,
                                          row_weights=None,
                                          row_to_table_dists=dists_tmp,
                                          fast_init=True)

                self.init_I_row_num += 1
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
            min_dist_per_input_tile = dists[range(self.tiles_NxN * self.tiles_NxN), argmin_dists]

            #print('. ', min_dist_per_input_tile)

            input_tile_index_with_max_min_table_dist = np.argmax(min_dist_per_input_tile)
            tmp = input_tile_index_with_max_min_table_dist

            candidate_row = tile_input_states_mat[tmp, :]
            new_min_dist = min_dist_per_input_tile[tmp]
            dists_tmp = dists[tmp, :].flatten()

            if new_min_dist > table_min_dist:
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

    def _learn_weights(self, cuda_table, dists, argmin_dists, tile_input_states_mat, input_x_y_theta):
        '''

        :param cuda_table:
        :param dists:
        :param argmin_dists:
        :param tile_input_states_mat:
        :param input_x_y_theta:
        :return:
        '''

        self.last_select_im_data = []

        for k in range(tile_input_states_mat.shape[0]):

            # only update top 1 match weights

            tile_input_vect = tile_input_states_mat[k, :]
            row_index = argmin_dists[k]

            table_row, _, _ = cuda_table.get_matrix_row(row_index=row_index)
            current_weights = cuda_table.get_row_weights(row_index=row_index)

            self.last_select_im_data.append((tile_input_vect, table_row, current_weights))

            per_pixel_dist_row = np.abs(tile_input_vect - table_row)

            learning_rate = 0.1
            new_weights = learning_rate * (1.0 - per_pixel_dist_row) + (1.0 - learning_rate) * current_weights

            cuda_table.set_row_weights(row_index=row_index, weights=new_weights, row_values=tile_input_vect)

    def _get_tile_inputs_for_image(self, image):
        '''

        :param image:
        :return:
        '''

        if self.im_r_indices is None:
            self._compute_im_indices_for_tile_inputs_mat(full_input_im=image)

        tile_inputs_mat = image[self.im_r_indices, self.im_c_indices]

        return tile_inputs_mat

    def _compute_im_indices_for_tile_inputs_mat(self, full_input_im):
        '''

        computes index matrices, once (slowly) to be able to get tile input states matrix from a single index operation from the input image

        :return:
        '''

        tiles_NxN = self.tiles_NxN
        a = full_input_im

        tile_size_NxN = int(a.shape[0] / tiles_NxN)

        im_r_indices = np.zeros((int(a.shape[0] * a.shape[1] / (tile_size_NxN * tile_size_NxN)), tile_size_NxN * tile_size_NxN), np.int)
        im_c_indices = np.zeros((int(a.shape[0] * a.shape[1] / (tile_size_NxN * tile_size_NxN)), tile_size_NxN * tile_size_NxN), np.int)

        for tile_row in range(tiles_NxN):
            for tile_col in range(tiles_NxN):
                tile_index = tile_row * tiles_NxN + tile_col

                r_start = tile_row * tile_size_NxN
                r_end = (tile_row + 1) * tile_size_NxN

                c_start = tile_col * tile_size_NxN
                c_end = (tile_col + 1) * tile_size_NxN

                r_indices = []
                c_indices = []
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        r_indices.append(r)
                        c_indices.append(c)

                im_r_indices[tile_index, :] = np.array(r_indices)
                im_c_indices[tile_index, :] = np.array(c_indices)

        self.im_r_indices = im_r_indices
        self.im_c_indices = im_c_indices

    def get_table_ims(self):
        '''

        :return:
        '''

        # 31 for 1000
        # 63 for 4000
        # N = int(min(31, sqrt(self.num_entries) - 1))  # display NxN tiles of 8k entries
        N = int(sqrt(self.num_entries))  # display NxN tiles of 8k entries

        entries = N*N

        cuda_table = self.table

        rows = cuda_table.table_i
        weights_orig = cuda_table.weight_masks

        # for display, normalize back to max 1 for each weight
        weights = np.multiply(weights_orig, 1.0 / np.amax(weights_orig, axis=1)[:, np.newaxis])

        tile_r_c = int(sqrt(rows.shape[1]))

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

                table_im[r0:r1, c0:c1] = tile_im #np.multiply(tile_weights, tile_im)

                weights_im[r0:r1, c0:c1] = tile_weights

                tile_n += 1
                c_offset += 1

            r_offset += 1

        # selection image

        if self.last_select_im_data is not None:
            num_select = 30

            select_im = np.zeros((num_select * tile_r_c + num_select * 1, 3 * tile_r_c + 3 * 1)) + 0.5

            select_list = np.random.permutation(len(self.last_select_im_data))

            for r in range(num_select):
                thing = self.last_select_im_data[select_list[r]]

                # self.last_select_im_data.append((tile_input_vect, table_row, current_weights))
                tile_input_vect, table_row, current_weights = thing
                r0 = r * tile_r_c + r
                r1 = (r + 1) * tile_r_c + r

                current_weights = current_weights * 1.0 / np.amax(current_weights)

                select_im[r0:r1, 0:tile_r_c] = tile_input_vect.reshape((tile_r_c, tile_r_c))
                select_im[r0:r1, tile_r_c+1:2*tile_r_c + 1] = table_row.reshape((tile_r_c, tile_r_c))
                select_im[r0:r1, 2*tile_r_c+2:3*tile_r_c + 2] = current_weights.reshape((tile_r_c, tile_r_c))
        else:
            select_im = None

        # scale all

        max_dim = max(table_im.shape[0], table_im.shape[1])
        imscale = self.table_ims_scale / max_dim  # 0.2: full table, 2.0
        table_im = cv2.resize(table_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        max_dim = max(weights_im.shape[0], weights_im.shape[1])
        imscale = self.table_ims_scale / max_dim  # 0.2: full table, 2.0
        weights_im = cv2.resize(weights_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        ims_list = [table_im, weights_im]
        ims_names_list = ['tiles', 'weights']

        if select_im is not None:
            max_dim = max(select_im.shape[0], select_im.shape[1])
            imscale = self.table_ims_scale / max_dim  # 0.2: full table, 2.0
            select_im = cv2.resize(select_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

            ims_list.append(select_im)
            ims_names_list.append('select')

        return ims_list, ims_names_list
















































