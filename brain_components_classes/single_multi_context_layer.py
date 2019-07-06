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


class SingleMultiContextLayer(object):
    def __init__(self, params):
        print('Initializing single layer...')

        self.input_dim = params['input_dim']
        self.output_dim = params['output_dim']
        self.context_dim = params['context_dim']

        self.num_IO_entries = params['num_IO_entries']
        self.num_C_entries = params['num_C_entries']
        self.predict_time = params['predict_time']

        self.include_context = params['include_context']  # bool

        assert (self.include_context and self.context_dim > 0) or (not self.include_context and self.context_dim == 0)
        #include_layers = ['io_only', 'i_only']  # last layer: io: learning, i: running
        #include_layers = ['ioc', 'ic_only']  # ioc: learning, ic: running

        # I+O Cuda Table

        self.cuda_table_IO = CudaTable(num_entries=self.num_IO_entries,
                                       input_dim=self.input_dim,
                                       output_dim=self.output_dim,
                                       context_dim=0,
                                       include_layers=['io_only', 'i_only'])

        self.cuda_table_C = None
        self.IO_to_C_W = None

        if self.include_context:
            # C table
            # *important but confusing*: this is storing context but internal to CudaTable, calling it "input"!
            self.cuda_table_C = CudaTable(num_entries=self.num_C_entries,
                                          input_dim=self.context_dim,
                                          output_dim=0,
                                          context_dim=0,
                                          include_layers=['i_only'])

            self.IO_to_C_W = np.zeros((self.num_C_entries, self.num_IO_entries), np.int)

        self.input_history = StatesLimitedHistory(params={'max_delay': self.predict_time,
                                                          'states_dim_list': [self.input_dim]})

        self.context_history = StatesLimitedHistory(params={'max_delay': self.predict_time,
                                                            'states_dim_list': [self.context_dim]})

        self.entries_x_y_theta_input = np.zeros((self.num_IO_entries, 3), np.float)
        self.entries_x_y_theta_output = np.zeros((self.num_IO_entries, 3), np.float)

        # used for init
        self.init_IO_row_num = None
        self.init_C_row_num = None

    def _scale_dists(self, dists):
        '''
            such that linear for:
            min distance pair -> output confidence = 1.0
            max distance pair -> output confidence = 0.0
            hard nonlinearity otherwise (maxed out to 0 or 1)
        '''

        return _scale_dists(dists=dists)

    def step(self, input_state, context_state, input_x_y_theta, learn_IO, learn_C):
        '''

        step once in real-time

        :param input_state:
        :param context_state:
        :param input_x_y_theta:
        :param learn_IO: True/False: learning of IO currently enabled
        :param learn_C: True/False: learning of C currently enabled
        :return:
        '''

        assert input_state.shape[0] == self.input_dim
        if self.context_dim == 0:
            assert context_state is None
        else:
            if learn_C or context_state is not None:
                assert context_state.shape[0] == self.context_dim

        # compute dists
        #   if context not defined, then based on I alone
        #   if context defined, then based on I+C (TBD)
        #   so - same line- context_state is None or valid here:

        # TODO how to incorporate a valid context_state here??
        dists = self.cuda_table_IO.query(query_input=input_state,
                                         query_output=None,
                                         query_context=None)  # TODO incorporate context but from context table!

        scaled_ic_dists = self._scale_dists(dists)

        self.input_history.store_new_states(newest_states_list=[input_state], extra_data_list=[input_x_y_theta])
        self.context_history.store_new_states(newest_states_list=[context_state], extra_data_list=[input_x_y_theta])

        # define input, output, context for learning

        train_input, train_input_x_y_theta = self.input_history.get_state(state_index=0, delay=self.predict_time)
        train_output, train_output_x_y_theta = self.input_history.get_state(state_index=0, delay=0)
        train_context, _ = self.context_history.get_state(state_index=0, delay=self.predict_time)

        assert abs(sum(train_output - input_state)) < 1e-5, 'state mismatch! ' + str(train_output) + ', ' + str(input_state) + ', sum: ' + str(sum(train_output - input_state))

        if learn_IO:
            IO_row_replaced = self._learn_IO(input_state=train_input,
                                             output_state=train_output,
                                             input_x_y_theta=train_input_x_y_theta,
                                             output_x_y_theta=train_output_x_y_theta)
        else:
            IO_row_replaced = False

        if learn_C and self.include_context:
            C_row_replaced = self._learn_C(input_state=train_input,
                                           output_state=train_output,
                                           context_state=context_state)
        else:
            C_row_replaced = False

        return scaled_ic_dists, IO_row_replaced, C_row_replaced

    def _learn_IO(self, input_state, output_state, input_x_y_theta, output_x_y_theta):
        row_replaced = False

        assert input_state.shape[0] == self.input_dim
        assert output_state.shape[0] == self.output_dim

        dists = self.cuda_table_IO.query(query_input=input_state,
                                         query_output=output_state,
                                         query_context=None)

        sorted_dist_indices = np.argsort(dists)
        new_min_ind = sorted_dist_indices[0]
        new_min_dist = dists[new_min_ind]

        if self.init_IO_row_num is None:
            self.init_IO_row_num = 0

        if self.init_IO_row_num < self.cuda_table_IO.get_num_rows():
            # necessary so dist matrix helper is not so slow at start
            dists[self.init_IO_row_num] = np.inf
            self.cuda_table_IO.set_matrix_row(row_index=self.init_IO_row_num,
                                           row_input=input_state,
                                           row_output=output_state,
                                           row_context=None,
                                           row_to_table_dists=dists,
                                           fast_init=True)

            # This only really makes sense for the first layer
            if output_x_y_theta is not None:
                self.entries_x_y_theta_output[self.init_IO_row_num, :] = output_x_y_theta[:]
            if input_x_y_theta is not None:
                self.entries_x_y_theta_input[self.init_IO_row_num, :] = input_x_y_theta[:]

            row_replaced = True
            self.init_IO_row_num += 1
        else:
            if not self.cuda_table_IO.post_init_done:
                self.cuda_table_IO.post_init()

            table_min_dist, table_min_dist_r, table_min_dist_c = self.cuda_table_IO.get_min_dist()

            if new_min_dist > table_min_dist:
                # minimum distance of new row to current rows is greater than current minimum row-row distance
                # so: replace one row of current minimum, with new row

                # get one of the row indices of current minimum dist pair
                r_r_ind = table_min_dist_r  # could be table_min_dist_c

                dists[r_r_ind] = np.inf

                # replace the current min dist row, with the new row
                self.cuda_table_IO.set_matrix_row(row_index=r_r_ind,
                                               row_input=input_state,
                                               row_output=output_state,
                                               row_context=None,
                                               row_to_table_dists=dists)

                if output_x_y_theta is not None:
                    self.entries_x_y_theta_output[r_r_ind, :] = output_x_y_theta[:]
                if input_x_y_theta is not None:
                    self.entries_x_y_theta_input[r_r_ind, :] = input_x_y_theta[:]

                row_replaced = True

        return row_replaced

    def _learn_C(self, input_state, output_state, context_state):
        row_replaced = False

        assert input_state.shape[0] == self.input_dim
        assert output_state.shape[0] == self.output_dim

        if self.context_dim == 0:
            assert context_state is None
            return row_replaced
        else:
            assert context_state is not None  # TODO might need to wait self.predict_time steps!
            assert context_state.shape[0] == self.context_dim

        # two ways to do it. learn cuda table, and W matrix independently. or at same time.
        # for now, same time. so, W will be reset occasionally.
        # W should learn every step, even if C not reset.
        #   reference:
        #
        #   self.IO_to_C_W = np.zeros((self.num_C_entries, self.num_IO_entries), np.int)


        # *****
        # (1) learn context table: cuda_table_C
        # *****

        dists = self.cuda_table_C.query(query_input=context_state, query_output=None, query_context=None)

        sorted_dist_indices = np.argsort(dists)
        new_min_ind = sorted_dist_indices[0]
        new_min_dist = dists[new_min_ind]

        min_row_C = new_min_ind

        if self.init_C_row_num is None:
            self.init_C_row_num = 0

        if self.init_C_row_num < self.cuda_table_C.get_num_rows():
            # necessary so dist matrix helper is not so slow at start
            dists[self.init_C_row_num] = np.inf
            self.cuda_table_C.set_matrix_row(row_index=self.init_C_row_num,
                                             row_input=context_state,
                                             row_output=None,
                                             row_context=None,
                                             row_to_table_dists=dists,
                                             fast_init=True)
            row_replaced = True
            min_row_C = self.init_C_row_num

            self.init_C_row_num += 1
        else:
            if not self.cuda_table_C.post_init_done:
                self.cuda_table_C.post_init()

            table_min_dist, table_min_dist_r, table_min_dist_c = self.cuda_table_C.get_min_dist()

            if new_min_dist > table_min_dist:
                # minimum distance of new row to current rows is greater than current minimum row-row distance
                # so: replace one row of current minimum, with new row

                # get one of the row indices of current minimum dist pair
                r_r_ind = table_min_dist_r  # could be table_min_dist_c

                dists[r_r_ind] = np.inf

                # replace the current min dist row, with the new row
                self.cuda_table_C.set_matrix_row(row_index=r_r_ind,
                                                 row_input=context_state,
                                                 row_output=None,
                                                 row_context=None,
                                                 row_to_table_dists=dists)

                min_row_C = r_r_ind
                row_replaced = True

                # zero out its connections to IO
                self.IO_to_C_W[r_r_ind, :] = 0

        # *****
        # (2) learn IO to Context association table: IO_to_C_W
        # *****

        # here, deal with self.IO_to_C_W

        # min_row_C: closest C table row (index to associate to)

        # look up closest IO row to training IO
        dists = self.cuda_table_IO.query(query_input=input_state,
                                         query_output=output_state,
                                         query_context=None)

        sorted_dist_indices = np.argsort(dists)
        min_row_IO = sorted_dist_indices[0]

        self.IO_to_C_W[min_row_C, min_row_IO] = 1

        return row_replaced

    def _get_table_im(self, cuda_table, layer_index=0):

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

            imscale = 2.0  # 0.2: full table
            # imscale = 5.0
            im = cv2.resize(D, dsize=(0,0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

        return im

    def get_IO_im(self, layer_index=0):
        return self._get_table_im(cuda_table=self.cuda_table_IO, layer_index=layer_index)

    def get_C_im(self, force_layer_0=False):

        if self.cuda_table_C is None:
            return None

        layer_index = 1
        if force_layer_0:
            layer_index = 0

        return self._get_table_im(cuda_table=self.cuda_table_C, layer_index=layer_index)

    def get_W_im(self):
        if self.IO_to_C_W is None:
            return None

        return self.IO_to_C_W.astype(np.uint8) * 255

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


# TODO test adding multiple contexts!
def test_run_single_multi_context_layer():
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

    # for purposes of this test, we are using future input as context, so, equal to input dim
    context_dim_for_testing = dim

    num_IO_entries = 400
    num_C_entries = 600

    table = SingleMultiContextLayer(params={
        'input_dim': dim,
        'output_dim': dim,
        'context_dim': context_dim_for_testing,
        'num_IO_entries': num_IO_entries,
        'num_C_entries': num_C_entries,
        'predict_time': 8,
        'include_context': True  # no context, io: lookup for learning, i: lookup for testing
    })

    fps = FPSCounter(params={'display_every_k_seconds': 5})

    # show image
    last_imshow_time = time.time()
    imshow_every_k_seconds = 2.0

    testing_dt_for_context = 30  # future value to I to use for context, for testing purposes for this script

    for t in range(max_history_length - testing_dt_for_context):

        input_state = states_history[t, :]
        x_y_theta = td_info_history[t, :]

        # for purposes of this test, we are using a future state of input as the context
        context_state_for_testing = states_history[t + testing_dt_for_context, :]

        scaled_i_c_dists, IO_replaced, C_replaced = table.step(input_state=input_state,
                                                               context_state=context_state_for_testing,
                                                               input_x_y_theta=x_y_theta,
                                                               learn_IO=True,
                                                               learn_C=t > num_IO_entries)

        if time.time() > last_imshow_time + imshow_every_k_seconds:
            im = table.get_IO_im(layer_index=0)
            if im is not None:
                cv2.imshow('IO_im', im)

            im3 = table.get_C_im(force_layer_0=False)  # force display as if it was layer 0, to see RGB
            if im3 is not None:
                cv2.imshow('C_im', im3)

            im4 = table.get_W_im()

            print(im4.shape, im4.dtype, np.amin(im4), np.amax(im4), np.count_nonzero(im4) * 1.0 / (im4.shape[0] * im4.shape[1]))

            if im4 is not None:
                cv2.imshow('W_im', im4)

            im2 = table.get_prediction_im(current_x_y_theta=x_y_theta,
                                          scaled_i_c_dists=scaled_i_c_dists,
                                          env_size=30)
            if im2 is not None:
                cv2.imshow('env_im', im2)

            last_imshow_time = time.time()

        cv2.waitKey(1)
        fps.update()


