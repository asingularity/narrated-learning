
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


class DynamicTiles(object):
    def __init__(self, params):
        '''

        # assume grayscale
        # assume binning

        :param params:
        '''

        self.max_history_length = params['max_history_length']
        self.ims_scale_pixels = float(params['ims_scale_pixels'])  # visualizer

        self.image_dim_NxN_pixels = params['image_dim_NxN_pixels']
        self.tile_dim_NxN_pixels = params['tile_dim_NxN_pixels']
        self.tiles_offset_N_pixels = params['tiles_offset_N_pixels']  # density

        self.rows_per_tile = params['rows_per_tile']

        assert 1 <= self.tiles_offset_N_pixels <= self.tile_dim_NxN_pixels
        # max density: only 1 pixel offset between spatially neighboring tiles; max overlap between tiles without being identical
        # min density: exactly size of tile offset; no overlap between tiles

        self.table_learn_time = params['table_learn_time']  # seq-nn-table learn time
        self.prediction_learn_time = params['prediction_learn_time']
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

        self.t = 0
        self.printed_init_step = False
        self.init_I_row_num = 0

        self.num_row_swaps = 0
        self.last_row_swap_disp = time.time()
        self.row_swap_disp_time = 1  # every K seconds

    def _compute_tiling(self):
        '''

        computes index matrices, once (slowly) to be able to get tile input states matrix from a single index operation from the input image

        also compute prediction radii

        :return:
        '''

        num_tiles_NxN = int(self.image_dim_NxN_pixels / self.tiles_offset_N_pixels)

        # adjust so no incomplete tiles at end
        for tile_r in range(num_tiles_NxN):
            r_start = tile_r * self.tiles_offset_N_pixels
            r_end = r_start + self.tile_dim_NxN_pixels

            if r_end > self.image_dim_NxN_pixels - 1:
                num_tiles_NxN -= 1

        # if self.image_dim_NxN_pixels % self.tile_dim_NxN_pixels > 0:
        #    num_tiles_NxN -= 1

        num_tiles = num_tiles_NxN * num_tiles_NxN
        num_pixels_per_tile = self.tile_dim_NxN_pixels * self.tile_dim_NxN_pixels

        # assert

        im_r_indices = np.zeros((num_tiles, num_pixels_per_tile), np.int)
        im_c_indices = np.zeros((num_tiles, num_pixels_per_tile), np.int)

        tiles_r_start = np.zeros(num_tiles, np.int)
        tiles_r_end = np.zeros(num_tiles, np.int)
        tiles_c_start = np.zeros(num_tiles, np.int)
        tiles_c_end = np.zeros(num_tiles, np.int)

        tiles_indices_within_radius = np.zeros((num_tiles, num_tiles), np.int)  # binary matrix: [from, to], is there a predictive projection?

        for tile_r in range(num_tiles_NxN):
            for tile_c in range(num_tiles_NxN):
                tile_index = tile_r * num_tiles_NxN + tile_c

                r_start = tile_r * self.tiles_offset_N_pixels
                r_end = r_start + self.tile_dim_NxN_pixels

                c_start = tile_c * self.tiles_offset_N_pixels
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

                # predictive connections
                for tile_r_to in range(num_tiles_NxN):
                    for tile_c_to in range(num_tiles_NxN):

                        r0 = (r_start + r_end) * 0.5
                        c0 = (c_start + c_end) * 0.5

                        r_start_1 = tile_r_to * self.tiles_offset_N_pixels
                        r_end_1 = r_start_1 + self.tile_dim_NxN_pixels

                        c_start_1 = tile_c_to * self.tiles_offset_N_pixels
                        c_end_1 = c_start_1 + self.tile_dim_NxN_pixels

                        r1 = (r_start_1 + r_end_1) * 0.5
                        c1 = (c_start_1 + c_end_1) * 0.5

                        dst = sqrt(pow(r0 - r1, 2) + pow(c0 - c1 , 2))

                        if dst <= self.prediction_radius_N_pixels:
                            tile_index_to = tile_r_to * num_tiles_NxN + tile_c_to
                            tiles_indices_within_radius[tile_index, tile_index_to] = 1

        self.num_tiles_NxN = num_tiles_NxN
        self.num_tiles = num_tiles
        self.num_pixels_per_tile = num_pixels_per_tile

        self.tiles_indices_within_radius = tiles_indices_within_radius

        self.tiles_r_start = tiles_r_start
        self.tiles_r_end = tiles_r_end
        self.tiles_c_start = tiles_c_start
        self.tiles_c_end = tiles_c_end

        self.im_r_indices = im_r_indices
        self.im_c_indices = im_c_indices


        print()
        print('*** _compute_tiling: ***')
        print('    ', 'tiles_offset_N_pixels:', self.tiles_offset_N_pixels)
        print('    ', 'tile_dim_NxN_pixels:', self.tile_dim_NxN_pixels)
        print('    ', 'num_tiles_NxN:', self.num_tiles_NxN)
        print('    ', 'num_pixels_per_tile:', self.num_pixels_per_tile)
        print('    ', 'num_tiles:', self.num_tiles)
        print()

        # there is a more efficient way to store
        # prediction weights storage should be optimized, as it is sparse, as computed above; within a radius
        # otherwise this matrix is huge
        # self.predict_weights = np.zeros((num_tiles * self.rows_per_tile, num_tiles * self.rows_per_tile), np.float)

        # tile_indices_within_radius: [num_tiles, num_tiles]: [from, to]

        self.win_row_history = StatesLimitedHistory(params={'max_delay': np.amax(np.array(self.prediction_tau_list)),
                                                            'states_dim_list': [self.num_tiles]})


        return

        num_tile_projections_per_tile = np.sum(tiles_indices_within_radius, axis=1).astype(np.int)  # tiles, not rows
        max_num_tile_projections = np.amax(num_tile_projections_per_tile) + 1  # tiles, not rows
        # why +1 above? we are designating a throwaway tile projection 0 for all to-indices that don't exist for a tile (i.e. from-tiles that are on the edges)

        self.max_num_tile_projections = max_num_tile_projections

        for tile_from in range(num_tiles):
            nt = 0
            for tile_to in range(num_tiles):
                # this is not gonna work; we've lost the geometric information here
                if tiles_indices_within_radius[tile_from, tile_to] > 0.5:
                    projection_indices[tile_from, nt] = tile_to
                    projection_weights[tile_from, nt] = 0.0
                    nt += 1

        # for tau in self.prediction_tau_list:

        projection_indices = np.zeros((num_tiles, max_num_projections), np.int)  # from, to (or zero/nan) - per tile

        if False:
            # NOT RIGHT below -> predictions are from all rows of from-tile to all rows of to-tiles
            projection_weights = np.zeros((num_tiles * self.rows_per_tile, max_num_projections * self.rows_per_tile), np.float)  # from, to (or zero/nan) - per tile - row

            for tile_from in range(num_tiles):
                nt = 0
                for tile_to in range(num_tiles):
                    if tiles_indices_within_radius[tile_from, tile_to] > 0.5:
                        projection_indices[tile_from, nt] = tile_to
                        projection_weights[tile_from, nt] = 0.0
                        nt += 1

                assert nt == num_projections_per_tile[tile_from]  # consistency check

            # self.max_num_projections = max_num_projections  # N, int
            self.projection_indices = projection_indices  # (num_tiles, N), int
            # projection weights tables: one per tau value
            self.projection_weights = {}  # dict of: tau -> (num_tiles, N), float
            for tau in self.prediction_tau_list:
                self.projection_weights[tau] = projection_weights.copy()

            self.num_projections_per_tile = num_projections_per_tile  # (num_tiles), int

    def _verify_tiling(self):
        '''

        display an image of tiles,
        outline two tiles to show overlap,
        and show a circle for prediction radius (with tile centers highlighted for those within circle)

        maybe do this a few times for random pairs of tiles

        :return:
        '''

        f_scale = float(self.ims_scale_pixels) / float(self.image_dim_NxN_pixels)

        num_pairs_to_show = 1  # self.num_tiles
        tile_index_1 = 47

        for p_tmp in range(num_pairs_to_show):
            im_to_show = np.zeros((int(self.image_dim_NxN_pixels * f_scale), int(self.image_dim_NxN_pixels * f_scale)), np.float)

            tile_indices = list(np.arange(self.num_tiles))

            for ind in tile_indices:
                r0 = self.tiles_r_start[ind]
                r1 = self.tiles_r_end[ind]
                c0 = self.tiles_c_start[ind]
                c1 = self.tiles_c_end[ind]

                color_to_use = 0.1  #  + random.random() * 0.2
                jt = 0  # random.randint(-4, 4)
                cv2.rectangle(img=im_to_show, pt1=(int(c0 * f_scale) + jt, int(r0 * f_scale) + jt), pt2=(int(c1 * f_scale) + jt, int(r1 * f_scale) + jt), color=color_to_use, thickness=1)

            # tile_index_1 = random.randint(0, self.num_tiles - 1)
            # tile_index_2 = random.randint(0, self.num_tiles - 1)

            for ind in [tile_index_1]: #, tile_index_2]:
                r0 = self.tiles_r_start[ind]
                r1 = self.tiles_r_end[ind]
                c0 = self.tiles_c_start[ind]
                c1 = self.tiles_c_end[ind]

                color_to_use = 0.8

                cv2.rectangle(img=im_to_show, pt1=(int(c0 * f_scale), int(r0 * f_scale)), pt2=(int(c1 * f_scale), int(r1 * f_scale)), color=color_to_use, thickness=2)

                cv2.circle(img=im_to_show, center=(int((c0+c1) * 0.5 * f_scale), int((r0+ r1) * 0.5 * f_scale)), radius=int(self.prediction_radius_N_pixels * f_scale), color=0.5, thickness=2)

                for ind2 in tile_indices:
                    if self.tiles_indices_within_radius[ind, ind2] > 0:
                        r = (self.tiles_r_start[ind2] + self.tiles_r_end[ind2]) * 0.5 * f_scale
                        c = (self.tiles_c_start[ind2] + self.tiles_c_end[ind2]) * 0.5 * f_scale

                        cv2.circle(img=im_to_show, center=(int(c), int(r)), radius=3, color=0.5, thickness=2)

            #t0 = time.time()
            #while time.time() - t0 < 0.2 * 1:
            cv2.imshow('tiling', im_to_show)
            cv2.waitKey(1)

            tile_index_1 += 1

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
            self._learn_table(cuda_table=self.table,
                              dists=dists,
                              argmin_dists=argmin_dists,
                              tile_input_states_mat=tile_input_states_mat,
                              input_x_y_theta=None)

        elif self.table_learn_time < self.t < self.prediction_learn_time:
            self._learn_prediction(cuda_table=self.table,
                                   dists=dists,
                                   argmin_dists=argmin_dists,
                                   tile_input_states_mat=tile_input_states_mat)
        else:
            pass

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

    def _learn_prediction(self, cuda_table, dists, argmin_dists, tile_input_states_mat):
        '''

        num_tiles: 196
        rows_per_tile: 1000

        :param cuda_table:
        :param dists: (num_tiles, rows_per_tile)
        :param argmin_dists: (num_tiles)
        :param tile_input_states_mat: (num_tiles, tile_input_dim)
        :return:
        '''

        # increment / apply learning to all active weights (past -> (tau) -> present)

        # tiles_indices_within_radius  # binary matrix, [num_tiles, num_tiles]: (from, to)

        # weight matrix is something like:
        #   a = np.random.random((rows_per_tile, max_num_projections, rows_per_tile))
        #   (1000, 21, 1000)
        #       or
        #   a = np.random.random((rows_per_tile, max_num_projections * rows_per_tile))
        #   (1000, 21 * 1000)

        return

        learn_rate_0 = 0.01
        keep_rate_0 = 1.0 - learn_rate_0

        learn_rate_1 = 0.02
        keep_rate_1 = 1.0 - learn_rate_1

        for tau in self.prediction_tau_list:

            win_row_per_tile_now = argmin_dists  # (num_tiles)
            win_row_per_tile_past = self.win_row_history.get_state(state_index=0, delay=tau)  # (num_tiles)  # todo use states history
            W = self.predict_weights[tau]  # rows_per_tile, max_num_projections * rows_per_tile

            # parallelize this loop in cython
            for from_tile in range(1, self.num_tiles):  # throwaway tile 0

                # to tiles is list of tiles this one projects to
                # each index is a specific geometric location
                # 0 index: we define to be: no tile exists for it on the grid
                #   meaning: max num projections is actually 22 (N+1), given the throwaway 0

                to_tiles = self.predict_tile_indices[from_tile, :]

                W = learn_rate_0 * 0.0 + keep_rate_0 * W
                # all rows unlearn

                # winning rows learn double
                #   indexing: [0 + (win), 1000 + (win), 2000 + (win), ...]
                tmp_to = self.rows_per_tile * np.arange(self.max_num_tile_projections) + win_row_per_tile_now[to_tiles]
                W[win_row_per_tile_past[from_tile], tmp_to] = learn_rate_1 * 1.0 + keep_rate_1 * W[win_row_per_tile_past[from_tile], tmp_to]
                # assumption for this to work with throwaway 0:
                # win_row_per_tile_past[0, :], win_row_per_tile_now[0] = 0 always, so W[0, :] and W[:, 0] is throwaway
                # set max_num_projections to N+1;


        return

        # self.max_num_projections  # int
        # self.projection_indices   # (num_tiles, max_num_projections), int
        # self.projection_weights   # { tau : (num_tiles, max_num_projections), float }
        # self.num_projections_per_tile   # (num_tiles), int

        print(self.max_num_projections)

        learn_rate = 0.001
        keep_rate = 1.0 - learn_rate

        for tau in self.prediction_tau_list:
            # adjust projection weights
            W = self.projection_weights[tau]

            # tile_active_or_not_past: [0, 0, 0, 0, 1, 1, 0, 0, 0, ...]
            # tile_indices_active_past: [4, 5, 19, 22, ...]
            #   (tau ago)
            # tile_active_or_not_now: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, ...]
            # tile_indices_active_now: [3, 7, 34, ...]
            #   (current time step)

            # tmp_indices_active_now =

            # W[tile_indices_active_past, tmp_indices_active_now] = learn_rate * 1.0 + keep_rate * W[tile_indices_active_past, tmp_indices_active_now]


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

        return ims_list, ims_names_list
