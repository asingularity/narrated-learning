import time
import cv2
import numpy as np
import random
import pickle

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

        pre_init_goal_contexts = params['pre_init_goal_contexts']
        max_history_length = params['max_history_length']

        # predictors always try to predict the next step (but there is decaying trace)
        self.predict_time = 1

        assert len(self.tile_entries_per_layer) == len(self.table_learn_time_per_layer) == len(self.prediction_learn_time_per_layer)

        self.n_layers = len(self.tile_entries_per_layer)

        offset_t = 0

        self.layer_table_learn_time_ranges = []
        self.layer_prediction_learn_time_ranges = []
        print()

        self.tables = []
        self.init_I_row_num = []

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

            self.tables.append(CudaTable(num_entries=self.tile_entries_per_layer[k],
                                         input_dim=int(table_input_dim)))
            print('init table with entries:', self.tile_entries_per_layer[k], 'input_dim:', int(table_input_dim))
            print()

            # TODO init prediction matrix stuff

        print()

        assert offset_t < max_history_length

        self.t = 0

        # TODO init goal context stuff

    # @profile
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
            learn_table = self.learning_enabled and (start_table_learn_t < self.t < stop_table_learn_t)
            tiles_NxN = self.tiles_per_layer_NxN[k]

            if k == 0:
                # how to get for each tile the input_state from raycast_image.flatten()?
                # here we need to get multiple input vectors, one for each tile from raycast_image, and process that in parallel
                #   in the cuda table lookup with one parallel lookup (one dot product)

                tile_input_states_mat = []
                grid = raycast_image.shape[0] / tiles_NxN

                for r in range(tiles_NxN):
                    for c in range(tiles_NxN):

                        r0 = r * grid
                        r1 = (r + 1) * grid

                        c0 = c * grid
                        c1 = (c + 1) * grid

                        tile_input_vect = raycast_image[r0:r1, c0:c1, :].flatten()

                        tile_input_states_mat.append(tile_input_vect)

                tile_input_states_mat = np.ascontiguousarray(np.array(tile_input_states_mat, np.float32))

            else:
                # TODO must be from previous layer
                tile_input_states_mat = None

            I_index_t = self._lookup_and_learn_table(layer_n=k,
                                                     cuda_table_I=self.tables[k],
                                                     input_states_mat=tile_input_states_mat,
                                                     learn_table=learn_table,
                                                     input_x_y_theta=input_x_y_theta)

        motor_out = None

        self.t += 1
        return motor_out

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

        I_index = None  # needed? #np.argmin(dists)

        if learn_table:
            self._learn_table(layer_n=layer_n,
                              cuda_table_I=cuda_table_I,
                              dists=dists,
                              argmin_dists=argmin_dists,
                              input_states_mat=input_states_mat,
                              input_x_y_theta=input_x_y_theta)

        return I_index  # learning might have invalidated this; doesn't matter for now because for now we are not using lookup if learning

    # @profile
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
                cuda_table_I.post_init()

            table_min_dist, table_min_dist_r, table_min_dist_c = cuda_table_I.get_min_dist()

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

        N = 64  # display NxN tiles of 8k entries
        entries = N*N

        rows = table[0:entries, :]
        # print(rows.shape, rows.dtype, np.amin(rows), np.amax(rows))
        # (16, 192) float32 0.0 0.923078

        tile_r_c = int(sqrt(rows.shape[1]/3))
        table_im = np.zeros((N * tile_r_c + N * 1, N * tile_r_c + N * 1, 3)) + 0.5
        tile_n = 0

        r_offset = 0
        for disp_r in range(N):

            c_offset = 0
            for disp_c in range(N):
                r0 = disp_r * tile_r_c + r_offset
                r1 = (disp_r + 1) * tile_r_c + r_offset
                c0 = disp_c * tile_r_c + c_offset
                c1 = (disp_c + 1) * tile_r_c + c_offset

                tile_im = table[tile_n, :].reshape((tile_r_c, tile_r_c, 3))

                table_im[r0:r1, c0:c1, :] = tile_im

                tile_n += 1

                c_offset += 1

            r_offset += 1

        # print(np.amin(table_im), np.amax(table_im))
        imscale = 4
        table_im = cv2.resize(table_im, dsize=(0,0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        return [table_im]


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

