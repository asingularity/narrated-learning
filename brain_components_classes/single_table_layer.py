import time
import cv2
import numpy as np
from math import sin, cos
from cuda_dist_query import CudaTable
from utils.load_sim_history import load_states_history, load_td_info_history
from utils.fps_counter import FPSCounter
from brain_components_classes.states_history import StatesLimitedHistory

NL_SIM_DIR = '/srv/projects/NL-sim/'


def _scale_dists(dists):
    '''
        such that linear for:
        min distance pair -> output confidence = 1.0
        max distance pair -> output confidence = 0.0
        hard nonlinearity otherwise (maxed out to 0 or 1)
    '''

    dists_copy = dists.copy()
    temp = np.argsort(dists_copy)

    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(dists_copy))
    dists_copy = ranks

    dists_copy = (dists_copy * 1.0 / len(dists_copy)).astype(dists.dtype)
    return dists_copy


class SingleTableLayer(object):
    def __init__(self, params):
        print('Initializing single layer...')

        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.context_dim = params['context_dim']

        self.num_entries = params['num_entries']
        self.predict_time = params['predict_time']

        self.include_layers = params['include_layers']

        self.cuda_table = CudaTable(num_entries=self.num_entries,
                                    input_dim=self.input_dim,
                                    output_dim=self.output_dim,
                                    context_dim=self.context_dim,
                                    include_layers=self.include_layers)

        self.valid_table = np.ones((self.num_entries, self.input_dim + self.output_dim + self.context_dim), np.int)
        self.row_last_replaced_by_refresh = np.zeros(self.num_entries, np.int)

        self.input_history = StatesLimitedHistory(params={'max_delay': self.predict_time,
                                                          'states_dim_list': [self.input_dim]})

        self.context_history = StatesLimitedHistory(params={'max_delay': self.predict_time,
                                                            'states_dim_list': [self.context_dim]})

        self.entries_x_y_theta_input = np.zeros((self.num_entries, 3), np.float)
        self.entries_x_y_theta_output = np.zeros((self.num_entries, 3), np.float)

        # used for init
        self.init_row_num = None

    def _scale_dists(self, dists):
        '''
            such that linear for:
            min distance pair -> output confidence = 1.0
            max distance pair -> output confidence = 0.0
            hard nonlinearity otherwise (maxed out to 0 or 1)
        '''

        return _scale_dists(dists=dists)

    def step(self, input_state, context_state, input_x_y_theta, learn, debug_print=False):

        assert input_state.shape[0] == self.input_dim
        if self.context_dim == 0:
            assert context_state is None
        else:
            assert context_state.shape[0] == self.context_dim

        dists = self.cuda_table.query(query_input=input_state,
                                      query_output=None,
                                      query_context=context_state)

        ind = np.argmin(dists)
        if debug_print:
            print('min dist ind IC:', ind)
        min_d, _, _ = self.cuda_table.get_min_dist()  # TODO base on input + context
        max_d, _, _ = self.cuda_table.get_max_dist()  # TODO base on input + context

        scaled_ic_dists = self._scale_dists(dists)

        self.input_history.process_new_states(newest_states_list=[input_state], extra_data_list=[input_x_y_theta])
        self.context_history.process_new_states(newest_states_list=[context_state], extra_data_list=[input_x_y_theta])  # TODO is it correct here to use input_x_y_theta?

        # define input, output, context for learning

        train_input, train_input_x_y_theta = self.input_history.get_state(state_index=0, delay=self.predict_time)
        train_output, train_output_x_y_theta = self.input_history.get_state(state_index=0, delay=0)
        train_context, _ = self.context_history.get_state(state_index=0, delay=self.predict_time)

        assert sum(train_output - input_state) == 0, 'state mismatch! ' + str(train_output) + ', ' + str(input_state)

        if learn:
            rep_row_ind = self._learn(input_state=train_input,
                                      output_state=train_output,
                                      context_state=train_context,
                                      input_x_y_theta=train_input_x_y_theta,
                                      output_x_y_theta=train_output_x_y_theta,
                                      debug_print=debug_print)
        else:
            rep_row_ind = None

        replaced_row_index = rep_row_ind
        return scaled_ic_dists, replaced_row_index

    def get_invalidate_stats(self):
        prop_table_valid = np.sum(self.valid_table) * 1.0 / (self.valid_table.shape[0] * self.valid_table.shape[1])

        prop_rows_refreshed = np.sum(self.row_last_replaced_by_refresh) * 1.0 / (self.row_last_replaced_by_refresh.shape[0])

        return prop_table_valid, prop_rows_refreshed

    def invalidate_index_context(self, column_index):
        '''
        invalidate this column_index for all rows of table in the context of the row
        will need to store for whole table- refresh will fix rows individually

        :param column_index:
        :return:
        '''

        self.valid_table[:, self.input_dim + self.output_dim + column_index] = 0

    def invalidate_index_input(self, column_index):
        '''
        invalidate this column_index for all rows of table in the input part of the row
        will need to store for whole table- refresh will fix rows individually

        :param column_index:
        :return:
        '''

        self.valid_table[:, column_index] = 0

    def _refresh_invalidated_indices(self, row_index, input_state, output_state, context_state, row_to_table_dists, debug_print=False):
        '''

        :param row_index:
        :param input_state:
        :param output_state:
        :param context_state:
        :return:
        '''

        #print('new_min_ind for learning: ', row_index)
        return False

        dists = row_to_table_dists

        did_replace = False
        if np.sum(self.valid_table[row_index, :]) < self.input_dim + self.output_dim + self.context_dim:
            dists[row_index] = np.inf
            self.cuda_table.set_matrix_row(row_index=row_index,
                                           row_input=input_state,
                                           row_output=output_state,
                                           row_context=context_state,
                                           row_to_table_dists=dists)

            if debug_print:
                print('refreshing row: ', row_index)
            self.valid_table[row_index, :] = 1
            self.row_last_replaced_by_refresh[row_index] = 1
            did_replace = True
        else:
            if debug_print:
                print('NOT refreshing row: ', row_index)

        return did_replace

    def _learn(self, input_state, output_state, context_state, input_x_y_theta=None, output_x_y_theta=None, debug_print=False):
        assert input_state.shape[0] == self.input_dim
        assert output_state.shape[0] == self.output_dim
        if self.context_dim == 0:
            assert context_state is None
        else:
            assert context_state.shape[0] == self.context_dim

        rep_row_ind = None

        # (1) get distance of new row to all current rows in table

        dists = self.cuda_table.query(query_input=input_state,
                                      query_output=output_state,
                                      query_context=context_state)

        sorted_dist_indices = np.argsort(dists)
        new_min_ind = sorted_dist_indices[0]

        if debug_print:
            print('min dist ind IOC:', new_min_ind)

        new_min_dist = dists[new_min_ind]
        ind2 = sorted_dist_indices[1]
        #print(new_min_ind)

        if self.init_row_num is None:
            self.init_row_num = 0

        if self.init_row_num < self.cuda_table.get_num_rows():
            # necessary so dist matrix helper is not so slow at start
            dists[self.init_row_num] = np.inf
            self.cuda_table.set_matrix_row(row_index=self.init_row_num,
                                           row_input=input_state,
                                           row_output=output_state,
                                           row_context=context_state,
                                           row_to_table_dists=dists,
                                           fast_init=True)
            self.valid_table[self.init_row_num, :] = 1
            self.row_last_replaced_by_refresh[self.init_row_num] = 0
            rep_row_ind = self.init_row_num

            # This only really makes sense for the first layer
            if output_x_y_theta is not None:
                self.entries_x_y_theta_output[self.init_row_num, :] = output_x_y_theta[:]
            if input_x_y_theta is not None:
                self.entries_x_y_theta_input[self.init_row_num, :] = input_x_y_theta[:]

            self.init_row_num += 1
        else:
            if not self.cuda_table.post_init_done:
                self.cuda_table.post_init()

            did_replace = self._refresh_invalidated_indices(row_index=new_min_ind,
                                                            input_state=input_state,
                                                            output_state=output_state,
                                                            context_state=context_state,
                                                            row_to_table_dists=dists,
                                                            debug_print=debug_print)

            # TODO set or not set rep_row_ind based on did_replace?

            if not did_replace:
                table_min_dist, table_min_dist_r, table_min_dist_c = self.cuda_table.get_min_dist()

                skip_min_dist_condition = True

                if new_min_dist > table_min_dist or skip_min_dist_condition:
                    # minimum distance of new row to current rows is greater than current minimum row-row distance
                    # so: replace one row of current minimum, with new row

                    # get one of the row indices of current minimum dist pair
                    r_r_ind = table_min_dist_r  # could be table_min_dist_c

                    if debug_print:
                        print('YES: replacing row: ', (table_min_dist, table_min_dist_r, table_min_dist_c), np.min(dists))
                        # if skip_min_dist_conditio is True: I+O min index is always I+O+C min index, which is one of rows being replaced

                    dists[r_r_ind] = np.inf

                    # replace the current min dist row, with the new row
                    self.cuda_table.set_matrix_row(row_index=r_r_ind,
                                                   row_input=input_state,
                                                   row_output=output_state,
                                                   row_context=context_state,
                                                   row_to_table_dists=dists)
                    self.valid_table[r_r_ind, :] = 1
                    self.row_last_replaced_by_refresh[r_r_ind] = 0
                    rep_row_ind = r_r_ind

                    if output_x_y_theta is not None:
                        self.entries_x_y_theta_output[r_r_ind, :] = output_x_y_theta[:]
                    if input_x_y_theta is not None:
                        self.entries_x_y_theta_input[r_r_ind, :] = input_x_y_theta[:]
                else:
                    if debug_print:
                        print('NOT: replacing row: ', (table_min_dist, table_min_dist_r, table_min_dist_c), np.min(dists))


        return rep_row_ind

    def get_table_im(self, layer_index=0):
        cuda_table = self.cuda_table

        # layer 0: display as color images below
        # layer 1...N-1: display as grayscale [0, 1] values?

        input_dim = cuda_table.input_dim
        output_dim = cuda_table.output_dim
        context_dim = cuda_table.context_dim

        table = np.transpose(cuda_table.get_table_from_gpu())

        if layer_index == 0:
            entries = 40 * 2 * 3
        else:
            entries = table.shape[0]

        im_input = table[0:entries, 0:input_dim]
        im_prediction = table[0:entries, input_dim:input_dim + output_dim]
        im_context = table[0:entries, input_dim + output_dim::]

        if layer_index == 0:
            A = im_input
            B = im_prediction
            C = np.empty((A.shape[0] + B.shape[0], A.shape[1]))
            C[::2, :] = A
            C[1::2, :] = B

            im = np.reshape(C, (C.shape[0], C.shape[1] / 3, 3))

            im = cv2.resize(im, dsize=(0,0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        else:
            # A = im_input
            # B = im_prediction
            # C = im_context
            # D = np.hstack((A, B, C))
            D = table

            imscale = 4.0  # full table
            # imscale = 5.0
            im = cv2.resize(D, dsize=(0,0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        return im

    def get_prediction_im(self, current_x_y_theta, scaled_i_c_dists, env_size):
        '''

        shows a 2D position & angle plot with:
            (1) current position & angle
            (2) stored positions & angles for all rows, with intensity or color reflecting rank of distances to current input

        :param current_x_y_theta:
        :return:
        '''

        show_current = True  # show closest current input rows
        show_prediction = True  # show predictions of closest current input rows

        num_to_show = 5

        im2 = np.zeros((30, 30, 3), np.float)
        scale_factor = 20

        im2 = cv2.resize(src=im2, dsize=(0, 0), fx=scale_factor, fy=scale_factor,
                         interpolation=cv2.INTER_NEAREST)

        sorted_indices = np.argsort(scaled_i_c_dists)

        if show_current:
            for k in range(num_to_show):  # scaled_i_c_dists.shape[0]

                ind = sorted_indices[k]

                p_x, p_y, p_theta = self.entries_x_y_theta_input[ind]

                cv2.circle(img=im2,
                           center=(int(p_x * scale_factor), int(p_y * scale_factor)),
                           radius=5,
                           color=(1.0 * (num_to_show - k) / num_to_show, 1.0 * (num_to_show - k) / num_to_show, 0.0 * (num_to_show - k) / num_to_show),
                           thickness=3)
                g2_x = p_x + 1. * cos(p_theta)
                g2_y = p_y + 1. * sin(p_theta)
                cv2.line(im2,
                         pt1=(int(p_x * scale_factor), int(p_y * scale_factor)),
                         pt2=(int(g2_x * scale_factor), int(g2_y * scale_factor)),
                         color=(1.0, 1.0, 0),
                         thickness=2)

        if show_prediction:
            for k in range(num_to_show):  # scaled_i_c_dists.shape[0]

                ind = sorted_indices[k]

                p_x, p_y, p_theta = self.entries_x_y_theta_output[ind]

                cv2.circle(img=im2,
                           center=(int(p_x * scale_factor), int(p_y * scale_factor)),
                           radius=5,
                           color=(0.0 * (num_to_show - k) / num_to_show, 1.0 * (num_to_show - k) / num_to_show,
                                  1.0 * (num_to_show - k) / num_to_show),
                           thickness=3)
                g2_x = p_x + 1. * cos(p_theta)
                g2_y = p_y + 1. * sin(p_theta)
                cv2.line(im2,
                         pt1=(int(p_x * scale_factor), int(p_y * scale_factor)),
                         pt2=(int(g2_x * scale_factor), int(g2_y * scale_factor)),
                         color=(0.0, 1.0, 1.0),
                         thickness=2)

        cur_x, cur_y, cur_theta = current_x_y_theta
        if cur_x is not None:
            cv2.circle(img=im2,
                       center=(int(cur_x * scale_factor), int(cur_y * scale_factor)),
                       radius=8,
                       color=(1.0, 0.0, 1.0),
                       thickness=3)
            g2_x = cur_x + 2. * cos(cur_theta)
            g2_y = cur_y + 2. * sin(cur_theta)
            cv2.line(im2,
                     pt1=(int(cur_x * scale_factor), int(cur_y * scale_factor)),
                     pt2=(int(g2_x * scale_factor), int(g2_y * scale_factor)),
                     color=(1.0, 0.0, 1.0),
                     thickness=2)

        return im2


def test_run_single_table_layer():
    '''
    run on a subset of our data, without context, and make sure learning a good representative set with dist/angle correlation as expected
    :return:
    '''

    sim_load_name = '48DIMx1M_states_positions_saved_2018-01-31T13:09:15.676826'
    plots_save_folder = NL_SIM_DIR + sim_load_name + '/'
    states_history = load_states_history(plots_save_folder)
    td_info_history = load_td_info_history(plots_save_folder)

    dim = states_history.shape[1]
    max_history_length = states_history.shape[0]

    table = SingleTableLayer(params={
        'input_dim': dim,
        'output_dim': dim,
        'context_dim': 0,
        'num_entries': 4000,
        'predict_time': 8,
        'include_layers': ['io_only', 'i_only']  # no context, io: lookup for learning, i: lookup for testing
    })

    fps = FPSCounter(params={'display_every_k_seconds': 5})

    # show image
    last_imshow_time = time.time()
    imshow_every_k_seconds = 2.0

    for t in range(max_history_length):

        input_state = states_history[t, :]
        x_y_theta = td_info_history[t, :]

        scaled_i_c_dists, _ = table.step(input_state=input_state,
                                         context_state=None,
                                         input_x_y_theta=x_y_theta,
                                         learn=True)

        if time.time() > last_imshow_time + imshow_every_k_seconds:
            im = table.get_table_im()
            if im is not None:
                cv2.imshow('table_im', im)

            im2 = table.get_prediction_im(current_x_y_theta=x_y_theta,
                                          scaled_i_c_dists=scaled_i_c_dists,
                                          env_size=30)
            if im2 is not None:
                cv2.imshow('env_im', im2)

            last_imshow_time = time.time()

        cv2.waitKey(1)
        fps.update()


