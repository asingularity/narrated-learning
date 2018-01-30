
import time
import cv2
import pickle
import numpy as np
np.set_printoptions(suppress=True)
from PVM.PVM_framework import MLP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_states_history():
    data_file = '/home/intec/NL-sim/24DIMx4M_states_saved_2017-12-30T16:59:11.615393/states_history_0.pkl'
    #data_file = '/home/intec/NL-sim/48DIMx4M_states_saved_2017-12-31T11:02:04.272824/states_history_0.pkl'

    print 'loading states history...'
    f = open(data_file, 'r')
    states_history = pickle.load(f)
    f.close()
    print states_history.shape  # (4000001, 48)
    return states_history


def _do_display(input_state, dim, scale_camera_factor):
    input_im_color = np.zeros((1, dim / 3, 3))
    input_im_color[0, :, :] = input_state.reshape((dim / 3, 3))

    resized_input_im = cv2.resize(src=input_im_color, dsize=(0, 0), fx=scale_camera_factor,
                                  fy=scale_camera_factor, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('actual', resized_input_im)
    cv2.waitKey(1)


def _compute_sparse_inputs(input_state, n_bins):

    io_bounds = (0.0, 1.0)

    input_sparse = np.zeros(input_state.shape[0] * n_bins) + io_bounds[0]
    input_indices = (input_state * n_bins).astype(np.int)
    input_indices[input_indices == n_bins] = n_bins - 1
    offset = np.arange(0, input_indices.shape[0] * n_bins, n_bins)
    input_indices = input_indices + offset
    input_sparse[input_indices] = io_bounds[1]

    return input_sparse.astype(np.float)


class PredictorEnsemble(object):
    def __init__(self, params):
        num_units = params['num_units']
        num_branches = params['num_branches']
        num_branch_inputs = params['num_branch_inputs']
        num_ff_inputs = params['num_ff_inputs']
        learning_rate = params['learning_rate']

        print 'initializing ensemble...'

        self.learning_rate = learning_rate
        self.num_units = num_units
        self.num_branches = num_branches
        self.num_branch_inputs = num_branch_inputs

        self.ff_weights = np.random.random((num_units, num_ff_inputs))
        self.branch_weights = np.random.random((num_units, num_branches, num_branch_inputs))

        self.last_unit_predictions = np.zeros(num_units)
        self.last_branch_activations = None
        self.last_branch_input = None

    def step(self, input_sparse):

        predict_threshold = 0.05
        ff_threshold = 0.05

        # (1) compute feedforward activations

        curr_ff_activations = np.dot(self.ff_weights, input_sparse) * 1.0 / input_sparse.shape[0]
        #print '***', np.amin(curr_ff_activations), np.amax(curr_ff_activations), np.mean(curr_ff_activations)

        # (2) compute branch activations
        branch_input = np.concatenate((input_sparse, self.last_unit_predictions))
        curr_branch_activations = np.dot(self.branch_weights, branch_input) * 1.0 / self.num_branch_inputs

        # (3) compute unit predictions
        curr_unit_predictions = np.amax(curr_branch_activations, axis=1)

        # (4) learn
        if self.last_branch_activations is not None:
            units_false_positive = np.nonzero(np.logical_and(self.last_unit_predictions > 0.5, curr_ff_activations < ff_threshold))[0]
            units_false_negative = np.nonzero(np.logical_and(self.last_unit_predictions < 0.5, curr_ff_activations > ff_threshold))[0]

            num_true_positives = np.count_nonzero(np.logical_and(self.last_unit_predictions > 0.5, curr_ff_activations > ff_threshold))
            num_true_negatives = np.count_nonzero(np.logical_and(self.last_unit_predictions < 0.5, curr_ff_activations < ff_threshold))

            #print 'true positives:', num_true_positives, 'true negatives:', num_true_negatives, 'false positives:', len(units_false_positive), 'false negatives:', len(units_false_negative)
            print 'Success:', (num_true_negatives + num_true_positives) * 1.0 / (num_true_negatives + num_true_positives + len(units_false_negative) + len(units_false_positive))
            #print (num_true_negatives + num_true_positives + len(units_false_negative) + len(units_false_positive))

            max_branches = np.argmax(self.last_branch_activations[units_false_negative, :], axis=1)
            dw = self.learning_rate * self.last_branch_input

            self.branch_weights[units_false_negative, max_branches, :] = self.branch_weights[units_false_negative, max_branches, :] + dw

            max_branches = np.argmax(self.last_branch_activations[units_false_positive, :], axis=1)
            dw = -self.learning_rate * self.last_branch_input
            self.branch_weights[units_false_positive, max_branches, :] = self.branch_weights[units_false_positive, max_branches, :] + dw

            # TODO enable?
            self.branch_weights[self.branch_weights < 0.0] = 0.0
            #self.branch_weights[self.branch_weights > 1.0] = 1.0

            #print '*', np.count_nonzero(curr_ff_activations < ff_threshold), np.count_nonzero(curr_ff_activations > ff_threshold)
            #print '**', np.count_nonzero(self.last_unit_predictions < 0.5), np.count_nonzero(self.last_unit_predictions > 0.5)
            #print '***', len(units_false_positive), len(units_false_negative)

        # (5) store last values needed later

        self.last_unit_predictions[:] = (curr_unit_predictions > predict_threshold).astype(np.float)
        self.last_branch_activations = curr_branch_activations.copy()
        self.last_branch_input = branch_input.copy()

        # TODO use weights "as if" they were binary- coincidence detection (say if weight > some threshold == 1 else 0). but train as if continuous [0, 1]
        #curr_branch_activations = np.dot(self.branch_weights, np.concatenate((input_sparse, self.last_unit_activations))) # * 1.0 / (input_sparse.shape[0] + self.last_unit_activations.shape[0])

        #print '***'
        #print np.sum(curr_ff_activations), input_sparse.shape[0], np.sum(curr_ff_activations) * 1.0 / input_sparse.shape[0]

        #print np.amin(curr_ff_activations), np.amax(curr_ff_activations), np.mean(curr_ff_activations)
        #print np.amin(curr_branch_activations), np.amax(curr_branch_activations), np.mean(curr_branch_activations)

        # (3) apply hebbian learning on (last branch activations, current feedforward sum):
        #if self.last_branch_activations is not None:
        #    pass
            #units_active = np.nonzero(curr_ff_activations)[0]
            #print units_active

        # TODO ***************************
        # TODO not curr_ff_activations but curr predicted ff activations here!!!
        # TODO ***************************
        #self.last_unit_activations[:] = curr_ff_activations[:]

        #self.last_branch_activations = curr_branch_activations.copy()


def run_experiment():
    plots_save_folder = '/home/intec/NL-tmp/'

    states_history = _load_states_history()

    num_units = 1000
    num_branches = 16

    dim = states_history.shape[1]
    n_bins = 10
    num_ff_inputs = dim * n_bins

    max_history_length = states_history.shape[0]
    scale_camera_factor = 32
    do_display = True

    k_to_train = np.arange(3, max_history_length)

    ensemble = PredictorEnsemble(params={'num_units': num_units,
                                     'num_branches': num_branches,
                                     'num_branch_inputs': num_ff_inputs + num_units,  # context = last ff + lateral feedback
                                     'num_ff_inputs': num_ff_inputs,
                                     'learning_rate': 0.1})

    fps_frames = 0
    last_fps_time = time.time()

    for k in k_to_train.tolist():
        fps_frames += 1

        if time.time() > last_fps_time + 5:
            print 'FPS: ', fps_frames / (time.time() - last_fps_time)
            fps_frames = 0
            last_fps_time = time.time()

        input_state = states_history[k, :]

        if do_display:
            _do_display(input_state, dim, scale_camera_factor)

        input_sparse = _compute_sparse_inputs(input_state, n_bins=n_bins)

        ensemble.step(input_sparse=input_sparse)


if __name__ == '__main__':
    run_experiment()
















