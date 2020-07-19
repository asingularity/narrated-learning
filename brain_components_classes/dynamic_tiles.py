import time
import cv2
import numpy as np
import random
import pickle

from cuda_dist_query import CudaTable
from utils.w_save_load_helper import save_W_prob
from math import sqrt, sin, cos
from utils.load_sim_history import load_states_history, load_td_info_history
from utils.fps_counter import FPSCounter
from brain_components_classes.states_history import StatesLimitedHistory
from brain_components_classes.sparse_binary_knn import SparseBinaryKNN
from brain_components_classes.multi_sparse_binary_knn import MultiSparseBinaryKNN


class OneTimeMessages(object):
    def __init__(self):
        self.d = []

    def print_once(self, msg):
        if msg not in self.d:
            print(msg)
            self.d.append(msg)


class DynamicTiles(object):
    def __init__(self, params):
        '''

        # assume grayscale
        # assume binning

        :param params:
        '''

        self.otm = OneTimeMessages()

        self.learning_off_time = params['learning_off_time']

        self.use_old_knn = False
        self.use_cython_if_new_knn = True

        self.max_history_length = params['max_history_length']
        self.ims_scale_pixels = float(params['ims_scale_pixels'])  # visualizer

        self.image_dim_NxN_pixels = params['image_dim_NxN_pixels']
        self.tile_dim_NxN_pixels = params['tile_dim_NxN_pixels']
        self.tiles_offset_N_pixels = params['tiles_offset_N_pixels']  # density

        self.rows_per_tile = params['rows_per_tile']

        assert 1 <= self.tiles_offset_N_pixels <= self.tile_dim_NxN_pixels
        # max density: only 1 pixel offset between spatially neighboring tiles; max overlap between tiles without being identical
        # min density: exactly size of tile offset; no overlap between tiles

        # set, no longer from params dict, but from other params!
        self.table_learn_time = 2 * self.rows_per_tile  # params['table_learn_time']  # seq-nn-table learn time

        # these go to sparse binary knn
        # TODO these should be demo level params
        # this was 2000... at 200, during training of weights there seem to be misses where no stored knn row has the winning row output index of the tile
        self.binary_knn_rows = 2000
        self.binary_knn_learn_every_k = 1
        # prediction_learn_time is no longer set here, but it will be: self.binary_knn_rows * self.binary_knn_learn_every_k

        self.prediction_radius_N_pixels = params['prediction_radius_N_pixels']  # farthest that a tile should predict another tile, spatially
        self.prediction_tau_list = params['prediction_tau_list']

        # self.num_bins_per_pixel = 10
        # self.tile_input_dim = self.tile_dim_NxN_pixels * self.tile_dim_NxN_pixels * self.num_bins_per_pixel
        # total input dim of one tile: num_pixels * num_bins

        self.tile_input_dim = self.tile_dim_NxN_pixels * self.tile_dim_NxN_pixels

        self._compute_tiling()
        self._verify_tiling()

        self.table = CudaTable(num_entries=self.rows_per_tile,
                               input_dim=self.tile_input_dim,
                               disable_row_row_dist=False,
                               enable_weight_bias=False)

        self.predict_steps_ahead = 1  # if changing this, would need to update how we display predicted image vs. current image etc. for debugging

        self.t = 0
        self.printed_init_step = False
        self.init_I_row_num = 0

        self.num_row_swaps = 0
        self.last_row_swap_disp = time.time()
        self.row_swap_disp_time = 1  # every K seconds

        self.predict_im = None
        self.current_im = None
        self.last_current_im = None
        self.last_predict_im = None  # ! assumes predict_tau = 1 !

    def _compute_tiling(self):
        '''

        should initialize:
            self.valid_post_tiles
            self.predictor_tile_indices
            self.knn

            self.im_r_indices
            self.im_c_indices

            self.tiles_r_start
            self.tiles_r_end
            self.tiles_c_start
            self.tiles_c_end

        :return:
        '''

        num_tiles_NxN = int(self.image_dim_NxN_pixels / self.tiles_offset_N_pixels)

        # adjust so no incomplete tiles at end
        for tile_r_ in range(num_tiles_NxN):
            r_start = tile_r_ * self.tiles_offset_N_pixels
            r_end = r_start + self.tile_dim_NxN_pixels

            if r_end > self.image_dim_NxN_pixels - 1:
                num_tiles_NxN -= 1

        num_tiles = num_tiles_NxN * num_tiles_NxN
        num_pixels_per_tile = self.tile_dim_NxN_pixels * self.tile_dim_NxN_pixels

        # ************
        # (0) declare arrays
        # ************

        im_r_indices = np.zeros((num_tiles, num_pixels_per_tile), np.int)
        im_c_indices = np.zeros((num_tiles, num_pixels_per_tile), np.int)

        tiles_r_start = np.zeros(num_tiles, np.int)
        tiles_r_end = np.zeros(num_tiles, np.int)
        tiles_c_start = np.zeros(num_tiles, np.int)
        tiles_c_end = np.zeros(num_tiles, np.int)

        # ************
        # (1) initialize relative neighborhood indices
        # ************

        ref_r = (num_tiles_NxN / 2)
        ref_c = (num_tiles_NxN / 2)

        ref_r_start = ref_r * self.tiles_offset_N_pixels
        ref_r_end = ref_r_start + self.tile_dim_NxN_pixels

        ref_c_start = ref_c * self.tiles_offset_N_pixels
        ref_c_end = ref_c_start + self.tile_dim_NxN_pixels

        ref_r_mid = (ref_r_start + ref_r_end) * 0.5
        ref_c_mid = (ref_c_start + ref_c_end) * 0.5

        print()
        print('computing prediction map...')
        print()
        rel_r_list = []
        rel_c_list = []

        for tile_r in range(num_tiles_NxN):
            for tile_c in range(num_tiles_NxN):
                r_start = tile_r * self.tiles_offset_N_pixels
                r_end = r_start + self.tile_dim_NxN_pixels

                c_start = tile_c * self.tiles_offset_N_pixels
                c_end = c_start + self.tile_dim_NxN_pixels

                r0 = (r_start + r_end) * 0.5
                c0 = (c_start + c_end) * 0.5

                dst = sqrt(pow(r0 - ref_r_mid, 2) + pow(c0 - ref_c_mid, 2))

                rel_r = int(ref_r - tile_r)
                rel_c = int(ref_c - tile_c)

                if dst <= self.prediction_radius_N_pixels:
                    print('    ', rel_r, rel_c)
                    rel_r_list.append(rel_r)
                    rel_c_list.append(rel_c)

        rel_r = np.array(rel_r_list)
        rel_c = np.array(rel_c_list)

        print()

        # ************
        # (2) compute pre-post prediction map
        # ************
        valid_post_tile_list = []
        num_predictor_tiles = rel_r.shape[0]
        predictor_tile_indices = np.zeros((num_tiles, num_predictor_tiles), np.int)

        for tile_r_ind in range(num_tiles_NxN):
            for tile_c_ind in range(num_tiles_NxN):
                tile_index = tile_r_ind * num_tiles_NxN + tile_c_ind

                # ************
                # (3) initialize image coordinates for tile: im_r_indices, im_c_indices
                # ************

                r_start = tile_r_ind * self.tiles_offset_N_pixels
                r_end = r_start + self.tile_dim_NxN_pixels

                c_start = tile_c_ind * self.tiles_offset_N_pixels
                c_end = c_start + self.tile_dim_NxN_pixels

                tiles_r_start[tile_index] = r_start
                tiles_r_end[tile_index] = r_end
                tiles_c_start[tile_index] = c_start
                tiles_c_end[tile_index] = c_end

                r_indices = []
                c_indices = []
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        r_indices.append(r)
                        c_indices.append(c)

                im_r_indices[tile_index, :] = np.array(r_indices)
                im_c_indices[tile_index, :] = np.array(c_indices)

                # ************
                # (4) initialize neighborhood for predictive map
                # ************

                valid_post = True

                # each pre tile
                for k in range(rel_r.shape[0]):
                    rel_r_ind = rel_r[k]
                    rel_c_ind = rel_c[k]

                    neighbor_r_ind = tile_r_ind - rel_r_ind
                    neighbor_c_ind = tile_c_ind - rel_c_ind

                    if 0 <= neighbor_r_ind <= num_tiles_NxN - 1 and 0 <= neighbor_c_ind <= num_tiles_NxN - 1:
                        neighbor_tile_ind = neighbor_r_ind * num_tiles_NxN + neighbor_c_ind
                        predictor_tile_indices[tile_index, k] = neighbor_tile_ind
                    else:
                        # this tile should not be a post tile for prediction (since it has an incomplete set of "pre" tiles)
                        # self.predict_tile_indices[tile_index, k] = num_tiles  # index for "non-existing neigbor for this geometric position" i.e. beyond edge of image

                        valid_post = False

                if valid_post:
                    valid_post_tile_list.append(tile_index)

        # ************
        # (5) initialize knn
        # ************

        if self.use_old_knn:  # old
            knn_list = []
            knn_params = None
            for tile_ind in range(num_tiles):
                # not all used
                if tile_ind in valid_post_tile_list:
                    knn_params = {
                        'sparse_io_dim': self.rows_per_tile,
                        'num_sparse_inputs': num_predictor_tiles * len(self.prediction_tau_list),
                        'num_sparse_outputs': 1,
                        'num_rows': self.binary_knn_rows,
                        'learn_row_every_k': self.binary_knn_learn_every_k
                    }
                    knn = SparseBinaryKNN(params=knn_params)
                else:
                    # save memory
                    knn = None

                knn_list.append(knn)

            print()
            print('initialized ', len(knn_list), ' knns.')
            print('sample knn params:')
            print(knn_params)
            print()

            self.knn = knn_list

        else:
            p = {
                'use_cuda': False,  # not implemented!
                'use_cython': self.use_cython_if_new_knn,
                'sparse_io_dim': self.rows_per_tile,  # rows per tile of cuda tables
                'num_sparse_inputs': num_predictor_tiles * len(self.prediction_tau_list),
                'num_sparse_outputs': 1,
                'num_rows': self.binary_knn_rows,
                'num_knn': len(valid_post_tile_list),
                'random_init': False  # for debugging
            }
            self.mknn = MultiSparseBinaryKNN(params=p)

            print()
            print('Initialized MultiSparseBinaryKNN')
            print()
            print(p)
            print()

        # ************
        # (6) class variables set
        # ************

        self.valid_post_tiles = np.array(valid_post_tile_list, np.int)
        self.predictor_tile_indices = predictor_tile_indices

        self.im_r_indices = im_r_indices
        self.im_c_indices = im_c_indices

        self.tiles_r_start = tiles_r_start  # for verify tiles
        self.tiles_r_end = tiles_r_end  # for verify tiles
        self.tiles_c_start = tiles_c_start  # for verify tiles
        self.tiles_c_end = tiles_c_end  # for verify tiles

        self.num_tiles = num_tiles  # for verify tiles

        self.win_row_history = StatesLimitedHistory(params={'max_delay': np.amax(np.array(self.prediction_tau_list)),
                                                            'states_dim_list': [self.num_tiles]})

    def _verify_tiling(self):
        f_scale = float(self.ims_scale_pixels) / float(self.image_dim_NxN_pixels)
        im_to_show = np.zeros((int(self.image_dim_NxN_pixels * f_scale), int(self.image_dim_NxN_pixels * f_scale)), np.float)

        tile_indices = list(np.arange(self.num_tiles))

        for ind in tile_indices:
            r0 = self.tiles_r_start[ind]
            r1 = self.tiles_r_end[ind]
            c0 = self.tiles_c_start[ind]
            c1 = self.tiles_c_end[ind]

            color_to_use = 0.2  #  + random.random() * 0.2
            jt = 0  # random.randint(-4, 4)
            cv2.rectangle(img=im_to_show, pt1=(int(c0 * f_scale) + jt, int(r0 * f_scale) + jt), pt2=(int(c1 * f_scale) + jt, int(r1 * f_scale) + jt), color=color_to_use, thickness=1)

        ind = int(self.num_tiles / 2)  # choose an index to display; for simplicity, somewhere near the middle

        r0 = self.tiles_r_start[ind]
        r1 = self.tiles_r_end[ind]
        c0 = self.tiles_c_start[ind]
        c1 = self.tiles_c_end[ind]

        color_to_use = 0.8

        cv2.rectangle(img=im_to_show, pt1=(int(c0 * f_scale), int(r0 * f_scale)), pt2=(int(c1 * f_scale), int(r1 * f_scale)), color=color_to_use, thickness=2)

        cv2.circle(img=im_to_show, center=(int((c0+c1) * 0.5 * f_scale), int((r0+ r1) * 0.5 * f_scale)), radius=int(self.prediction_radius_N_pixels * f_scale), color=0.5, thickness=2)

        list_predictors = list(self.predictor_tile_indices[ind, :].flatten())

        for ind2 in tile_indices:
            if ind2 in list_predictors:
                r = (self.tiles_r_start[ind2] + self.tiles_r_end[ind2]) * 0.5 * f_scale
                c = (self.tiles_c_start[ind2] + self.tiles_c_end[ind2]) * 0.5 * f_scale

                cv2.circle(img=im_to_show, center=(int(c), int(r)), radius=2, color=0.5, thickness=2)

        cv2.imshow('tiling', im_to_show)
        cv2.waitKey(1)

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

    #@profile
    def _step_actual(self, input_image):
        '''

        :param input_image:
        :return:
        '''

        # ensure that dim of an entry == number of pixels in a tile == input image divided into tiles_NxN tiles

        # assert input_image.shape[0] * input_image.shape[1] == self.tile_input_dim

        if not self.printed_init_step:
            print()
            print('DynamicTiles.step: ', self.t)
            print()
            print('    input_image.shape: ', input_image.shape)
            print('    input_image.dtype: ', input_image.dtype)
            print('    input_image min, max: ', np.amin(input_image), np.amax(input_image))

        tile_input_states_mat = self._get_tile_inputs_for_image(image=input_image)
        assert tile_input_states_mat.shape[0] == self.num_tiles
        assert tile_input_states_mat.shape[1] == self.tile_input_dim

        if not self.printed_init_step:
            print('    tile_input_states_mat.shape: ', tile_input_states_mat.shape)
            print('    tile_input_states_mat.dtype: ', tile_input_states_mat.dtype)
            print('    tile_input_states_mat min, max: ', np.amin(tile_input_states_mat), np.amax(tile_input_states_mat))

        dists, argmin_dists = self.table.query_multiple_rows(query_inputs=tile_input_states_mat)
        I_indices = argmin_dists

        self.win_row_history.process_new_states(newest_states_list=[argmin_dists], extra_data_list=[None])

        if not self.printed_init_step:
            print('    dists.shape: ', dists.shape)
            print('    dists.dtype: ', dists.dtype)
            print('    dists min, max: ', np.amin(dists), np.amax(dists))
            print('    argmin_dists.shape: ', argmin_dists.shape)
            print('    argmin_dists.dtype: ', argmin_dists.dtype)
            print('    argmin_dists min, max: ', np.amin(argmin_dists), np.amax(argmin_dists))

        if self.t < self.table_learn_time:
            self.otm.print_once('dynamic_tiles:: starting learning of table')
            self._learn_table(cuda_table=self.table,
                              dists=dists,
                              argmin_dists=argmin_dists,
                              tile_input_states_mat=tile_input_states_mat,
                              input_x_y_theta=None)
        else:
            self.otm.print_once('dynamic_tiles:: finished learning of table')

        if self.t > self.table_learn_time:
            # internally, this will first learn knn rows, and then learn weights
            if self.t < self.learning_off_time:
                self.otm.print_once('dynamic_tiles:: started learning of predictions')
                self._learn_prediction(cuda_table=self.table,
                                       dists=dists,
                                       argmin_dists=argmin_dists,
                                       tile_input_states_mat=tile_input_states_mat)
            else:
                self.otm.print_once('dynamic_tiles:: finished learning of predictions')

            # start making predictions even while still learning prediction matrices
            predict_im = self._make_prediction(cuda_table=self.table,
                                               dists=dists,
                                               argmin_dists=argmin_dists,
                                               tile_input_states_mat=tile_input_states_mat)

            # This logic below assumes we are predicting 1 step ahead! If it was more, we would need to change this to a states history!

            if self.predict_im is not None:
                self.last_current_im = self.current_im.copy()
                self.last_predict_im = self.predict_im.copy()

            self.current_im = input_image.copy()
            self.predict_im = predict_im.copy()

        if not self.printed_init_step:
            print()

        if self.t > 3:
            self.printed_init_step = True

        self.t += 1
        return None

    def _get_tile_inputs_for_image(self, image):
        '''

        :param image:
        :return:
        '''

        tile_inputs_mat = image[self.im_r_indices, self.im_c_indices]

        return tile_inputs_mat

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
            min_dist_per_input_tile = dists[range(self.num_tiles), argmin_dists]

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

    #@profile
    def _learn_prediction(self, cuda_table, dists, argmin_dists, tile_input_states_mat):
        '''

        :param cuda_table:
        :param dists:
        :param argmin_dists:
        :param tile_input_states_mat:
        :return:

        def _expand_win_row(win_row_index):
            tmp = np.zeros(self.rows_per_tile)
            tmp[win_row_index] = 1.0
            return tmp

        def _expand_win_rows(win_row_indices):
            # expand [3, 5, 9] to long binary array [0, 0, 0, 1, 0, 0, 0, ...]
            return tmp

        '''

        knn_input_win_rows_list = []
        knn_output_win_tile_list = []
        tmp5 = {}

        for tile_n in list(self.valid_post_tiles):

            win_row_post_tile_now = self.win_row_history.get_state(state_index=0, delay=0)[0].astype(np.int)[tile_n]
            pre_tile_indices = self.predictor_tile_indices[tile_n, :]  # [post, pre]: (num_tiles, num_predictor_tiles)
            knn_input_all_tau = np.array([])

            for tau in self.prediction_tau_list:
                # get inputs from its from_tiles

                if tau not in tmp5:
                    tmp4 = self.win_row_history.get_state(state_index=0, delay=tau)[0]
                    tmp4 = tmp4.astype(np.int)
                    tmp5[tau] = tmp4
                else:
                    tmp4 = tmp5[tau]

                win_rows_pre_tiles_past = tmp4[pre_tile_indices]

                # old line:
                # win_rows_pre_tiles_past = self.win_row_history.get_state(state_index=0, delay=tau)[0].astype(np.int)[pre_tile_indices]

                knn_input_all_tau = np.concatenate((knn_input_all_tau, win_rows_pre_tiles_past))

            knn_input_win_rows_list.append(knn_input_all_tau)
            knn_output_win_tile_list.append(win_row_post_tile_now)

        self.mknn.train(knn_input_win_rows_2d_arr=np.array(knn_input_win_rows_list, np.int32), knn_output_win_row_arr=np.array(knn_output_win_tile_list, np.int32))

    def _make_prediction(self, cuda_table, dists, argmin_dists, tile_input_states_mat):

        knn_input_win_rows_list = []

        tmp5 = {}

        for tile_n in list(self.valid_post_tiles):
            pre_tile_indices = self.predictor_tile_indices[tile_n, :]

            knn_input_all_tau = np.array([])

            for tau in self.prediction_tau_list:
                predict_steps_ahead = self.predict_steps_ahead
                assert tau >= predict_steps_ahead

                if tau not in tmp5:
                    tmp4 = self.win_row_history.get_state(state_index=0, delay=tau - predict_steps_ahead)[0]
                    tmp4 = tmp4.astype(np.int)
                    tmp5[tau] = tmp4
                else:
                    tmp4 = tmp5[tau]

                win_rows_pre_tiles_past = tmp4[pre_tile_indices]

                # old line:
                # win_rows_pre_tiles_past = self.win_row_history.get_state(state_index=0, delay=tau - predict_steps_ahead)[0].astype(np.int)[pre_tile_indices]

                knn_input_all_tau = np.concatenate((knn_input_all_tau, win_rows_pre_tiles_past))

            knn_input_win_rows_list.append(knn_input_all_tau)

        knn_input_win_rows_2d_arr = np.array(knn_input_win_rows_list, np.int32)

        predicted_row_per_valid_post_tile = self.mknn.predict(knn_input_win_rows_2d_arr=knn_input_win_rows_2d_arr)

        # second layer
        # print(predicted_row_per_valid_post_tile.shape)  # (3025,)

        # *************** make prediction image ***************

        sum_im = np.zeros((self.image_dim_NxN_pixels, self.image_dim_NxN_pixels))
        num_im = np.zeros((self.image_dim_NxN_pixels, self.image_dim_NxN_pixels))

        k = 0
        for tile_n in list(self.valid_post_tiles):
            r0 = self.tiles_r_start[tile_n]
            r1 = self.tiles_r_end[tile_n]
            c0 = self.tiles_c_start[tile_n]
            c1 = self.tiles_c_end[tile_n]

            tile_row_im = cuda_table.get_matrix_row(row_index=predicted_row_per_valid_post_tile[k])[0].reshape((self.tile_dim_NxN_pixels, self.tile_dim_NxN_pixels))

            # this is an invalid hack! can't assume anything about image values!
            # max_im[r0:r1, c0:c1] = np.maximum(max_im[r0:r1, c0:c1], tile_row_im)

            sum_im[r0:r1, c0:c1] = sum_im[r0:r1, c0:c1] + tile_row_im  # [0] because still returns I, O, C
            num_im[r0:r1, c0:c1] = num_im[r0:r1, c0:c1] + 1

            k += 1

        self.prediction_made_indices = np.nonzero(num_im > 0)  # for computing prediction error in get_table_ims, only in proper areas

        mean_im = np.divide(sum_im, num_im + 1e-12)
        predict_im = mean_im

        return predict_im

    def get_table_ims(self):
        '''

        :return:
        '''

        # 31 for 1000
        # 63 for 4000
        # N = int(min(31, sqrt(self.num_entries) - 1))  # display NxN tiles of 8k entries
        N = int(sqrt(self.rows_per_tile))  # display NxN tiles of 8k entries

        cuda_table = self.table
        rows = cuda_table.table_i

        #print('***', np.amin(rows), np.amax(rows))
        #print('---', np.amin(cuda_table.table_i), np.amax(cuda_table.table_i))

        tile_r_c = int(sqrt(rows.shape[1]))

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

        ims_list = [table_im]
        ims_names_list = ['tiles']

        if select_im is not None:
            max_dim = max(select_im.shape[0], select_im.shape[1])
            imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
            select_im = cv2.resize(select_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

            ims_list.append(select_im)
            ims_names_list.append('select')

        if self.last_predict_im is not None:

            # this assumes we are predicting 1 step ahead! if was more, we would need to change to rolling buffer i.e. states history for this debug display!
            assert self.predict_steps_ahead == 1

            #max_dim = max(self.predict_im.shape[0], self.predict_im.shape[1])
            #imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
            #p_im = cv2.resize(self.predict_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)


            # append in order: current (t), prediction (t'+1), future (t+1)

            current_im = self.last_current_im  # "current" input (from past for display)
            predict_im = self.last_predict_im  # "current" prediction based on that input (from past for display)
            future_im = self.current_im  # "future" input (from present for display)

            # this is comparing: [current - prediction] vs. [future - prediction]
            # error_to_current = np.fabs(np.sum(current_im[self.prediction_made_indices] - predict_im[self.prediction_made_indices]))
            # error_to_future = np.fabs(np.sum(future_im[self.prediction_made_indices] - predict_im[self.prediction_made_indices]))

            # this is comparing: [future - current] vs. [future - prediction], to see if it does better than "use current image as prediction of future"
            error_to_current = np.fabs(np.sum(future_im[self.prediction_made_indices] - current_im[self.prediction_made_indices]))
            error_to_predict = np.fabs(np.sum(future_im[self.prediction_made_indices] - predict_im[self.prediction_made_indices]))

            good = error_to_predict < error_to_current
            if good:
                good = 'Y'
            else:
                good = ' '

            print(good, 'err to input:', error_to_current, error_to_predict, ': err to predict')

            spacer = 0.0 * np.ones((current_im.shape[0], 5))

            c_tmp = np.zeros_like(current_im)
            c_tmp[self.prediction_made_indices] = current_im[self.prediction_made_indices]
            f_tmp = np.zeros_like(future_im)
            f_tmp[self.prediction_made_indices] = future_im[self.prediction_made_indices]

            p_im = np.hstack((c_tmp, spacer, predict_im, spacer, f_tmp))

            # print(current_im.shape, predict_im.shape, future_im.shape)

            ims_list.append(p_im)
            ims_names_list.append('current, predict, future')

        #output_prop_im = self.mknn.output_prop_arr.copy()
        #max_dim = max(output_prop_im.shape[0], output_prop_im.shape[1])
        #imscale = self.ims_scale_pixels / max_dim  # 0.2: full table, 2.0
        #output_prop_im = cv2.resize(output_prop_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)
        #ims_list.append(output_prop_im)
        #ims_names_list.append('output prop over knn rows')

        return ims_list, ims_names_list



















