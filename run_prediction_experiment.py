
import time
import cv2
import pickle
import numpy as np
np.set_printoptions(suppress=True)
from PVM.PVM_framework import MLP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _get_mlp_OLD(num_inputs, num_hidden, num_outputs, learning_rate):
    state = {}
    state['layers'] = [
        {'activation': np.zeros((num_inputs + 1,)), 'error': np.zeros((num_inputs + 1,)), 'delta': np.zeros((num_inputs + 1,))},
        {'activation': np.zeros((num_hidden + 1,)), 'error': np.zeros((num_hidden + 1,)), 'delta': np.zeros((num_hidden + 1,))},
        {'activation': np.zeros((num_outputs + 1,)), 'error': np.zeros((num_outputs + 1,)), 'delta': np.zeros((num_outputs + 1,))},
    ]
    state['weights'] = [
        MLP.initialize_weights(np.zeros((num_inputs + 1, num_hidden)), False),
        MLP.initialize_weights(np.zeros((num_hidden + 1, num_outputs)), False)
    ]
    state['beta'] = np.array([1.0])
    state['learning_rate'] = np.array([learning_rate])
    state['momentum'] = np.array([0.5])
    state['mse'] = np.array([0.0])

    return MLP.MLP(state)


class EnsembleNet(object):
    def __init__(self, params):

        uniform_init = False
        if uniform_init:
            self.bases = np.ones((params['num_branches'], params['num_inputs']))
            self.stored_prediction = np.ones(params['num_outputs'])
        else:
            self.bases = np.random.random((params['num_branches'], params['num_inputs']))
            self.stored_prediction = np.random.random(params['num_outputs'])
        self.learning_rate = params['learning_rate']

        self.trained_num = 0
        self.last_trained = time.time()
        # print self.bases.shape, self.stored_prediction.shape
        # (10, 48) (24,)
        self.params = params

    def train(self, net_input, net_output):
        self.trained_num += 1
        self.last_trained = time.time()

        self.stored_prediction = self.learning_rate * net_output + (1.0 - self.learning_rate) * self.stored_prediction
        df = self.bases - net_input
        df_sum = np.sum(np.fabs(df), axis=1)
        best_branch = np.argmin(df_sum)
        self.bases[best_branch] = self.learning_rate * net_input + (1.0 - self.learning_rate) * self.bases[best_branch]

    def evaluate(self, net_input, randomize=False):

        if self.trained_num == 0 or (randomize and (time.time() - self.last_trained > 0.1)):
            params = self.params
            self.bases = np.random.random((params['num_branches'], params['num_inputs']))
            self.stored_prediction = np.random.random(params['num_outputs'])

        return self.stored_prediction


def _get_mlp(num_inputs, num_hidden, num_outputs, learning_rate):

    params = {'num_inputs': num_inputs,
              'num_outputs': num_outputs,
              'learning_rate': learning_rate,
              'num_branches': 100}  # 50

    return EnsembleNet(params)


def _load_states_history():
    #data_file = '/home/intec/NL-sim/24DIMx4M_states_saved_2017-12-30T16:59:11.615393/states_history_0.pkl'
    data_file = '/home/intec/NL-sim/48DIMx4M_states_saved_2017-12-31T11:02:04.272824/states_history_0.pkl'

    print 'loading states history...'
    f = open(data_file, 'r')
    states_history = pickle.load(f)
    f.close()
    print states_history.shape
    return states_history


def run_experiment():
    plots_save_folder = '/home/intec/NL-tmp/'

    states_history = _load_states_history()

    dim = states_history.shape[1]
    max_history_length = states_history.shape[0]
    scale_camera_factor = 32
    num_mlp = 200  # 100
    do_display = False
    learning_rate = 0.01  # 0.01  # 0.01 * 0.5
    error_average_steps = 1000
    do_training = True

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    print 'initializing mlp list...'
    mlp_list = []
    for k in range(num_mlp):
        mlp_list.append(_get_mlp(num_inputs=dim * 2, num_hidden=dim, num_outputs=dim, learning_rate=learning_rate))

    nets_trained_hist = np.zeros(num_mlp)

    single_mlp = _get_mlp(num_inputs=dim * 2, num_hidden=dim, num_outputs=dim, learning_rate=learning_rate)

    error_histories = np.zeros((num_mlp, max_history_length))
    mean_error_histories = np.zeros((num_mlp, max_history_length))
    error_steps = np.zeros(num_mlp)

    mean_error_history_best = np.zeros(max_history_length)
    best_step = 0

    single_error_history = np.zeros(max_history_length)
    single_mean_error_history = np.zeros(max_history_length)
    single_error_step = 0

    last_time = time.time()

    for k in range(3, max_history_length):
        #if k > 10000:
        #    do_training = False

        if do_display:
            autoenc_images = np.zeros((3, dim))
            autoenc_images[0, :] = states_history[k - 2, :]
            autoenc_images[1, :] = states_history[k - 1, :]
            autoenc_images[2, :] = states_history[k - 0, :]

            autoenc_images_color = np.zeros((autoenc_images.shape[0], autoenc_images.shape[1] / 3, 3))

            for r in range(autoenc_images.shape[0]):
                tmp = autoenc_images[r, :].reshape(autoenc_images.shape[1] / 3, 3)
                autoenc_images_color[r, :, 0] = tmp[:, 0]
                autoenc_images_color[r, :, 1] = tmp[:, 1]
                autoenc_images_color[r, :, 2] = tmp[:, 2]

            resized_autoenc = cv2.resize(src=autoenc_images_color, dsize=(0, 0), fx=scale_camera_factor,
                                         fy=scale_camera_factor, interpolation=cv2.INTER_NEAREST)
            cv2.imshow('autoenc', resized_autoenc)
            cv2.waitKey(1)

        net_input = np.concatenate((states_history[k - 2, :], states_history[k - 0, :]))
        net_output = states_history[k - 1, :]

        # *** multi ***

        net_i = 0
        output_errors = np.zeros(num_mlp)
        for net in mlp_list:
            output_eval = net.evaluate(net_input.copy(), randomize=False) #(k < 20000))

            error = output_eval - net_output
            error = np.mean(np.fabs(error))

            output_errors[net_i] = error
            net_i += 1

        net_to_train = np.argmin(output_errors)
        error_histories[net_to_train, error_steps[net_to_train]] = output_errors[net_to_train]

        mean_index_0 = max(0, error_steps[net_to_train] - error_average_steps)
        mean_index_1 = error_steps[net_to_train]

        curr_net_mean = np.mean(error_histories[net_to_train, mean_index_0:mean_index_1])
        mean_error_histories[net_to_train, error_steps[net_to_train]] = curr_net_mean

        error_steps[net_to_train] += 1

        mean_error_history_best[best_step] = curr_net_mean
        best_step += 1

        if do_training:
            mlp_list[net_to_train].train(net_input.copy(), net_output.copy())
        nets_trained_hist[net_to_train] += 1

        # *** single ***

        single_error = single_mlp.evaluate(net_input.copy()) - net_output
        single_error = np.mean(np.fabs(single_error))

        single_error_history[single_error_step] = single_error

        mean_index_0 = max(0, single_error_step - error_average_steps)
        mean_index_1 = single_error_step

        single_mean_error_history[single_error_step] = np.mean(single_error_history[mean_index_0:mean_index_1])
        single_error_step += 1

        if do_training:
            single_mlp.train(net_input.copy(), net_output.copy())

        if k % 5000 == 0:
            print time.time() - last_time, k
            last_time = time.time()

            sorted_net_indices = np.argsort(nets_trained_hist)[::-1]

            # TODO mlp Errors over time, averaged
            # TODO use monolithic_mlp too
            # TODO plot histogram of mlp use vs. index (sort by use / num times trained)

            ax.cla()
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            ax.bar(np.arange(num_mlp), nets_trained_hist[sorted_net_indices])
            fig.savefig(plots_save_folder + '/' + 'mlp_use_hist_step_' + '_' + str(k) + '.png', dpi=100)

            for tmp in range(min(num_mlp, 10)):
                net_index = sorted_net_indices[tmp]
                ax.cla()
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                thing_to_plot = mean_error_histories[net_index, 0:error_steps[net_index]]
                ax.plot(thing_to_plot, 'b-')
                fig.savefig(plots_save_folder + '/' + 'error_history_net_' + str(tmp) + '.png', dpi=100)

            ax.cla()
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            thing_to_plot = single_mean_error_history[0:single_error_step]
            ax.plot(thing_to_plot, 'b-')
            fig.savefig(plots_save_folder + '/' + 'single_error_history' + '.png', dpi=100)

            ax.cla()
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            thing_to_plot = mean_error_history_best[0:best_step]
            ax.plot(thing_to_plot, 'b-')
            fig.savefig(plots_save_folder + '/' + 'best_error_history' + '.png', dpi=100)


def view_data():
    states_history = _load_states_history()

    dim = states_history.shape[1]
    scale_camera_factor = 64

    for k in range(3, states_history.shape[0]):
        autoenc_images = np.zeros((3, dim))
        autoenc_images[0, :] = states_history[k - 2, :]
        autoenc_images[1, :] = states_history[k - 1, :]
        autoenc_images[2, :] = states_history[k - 0, :]

        autoenc_images_color = np.zeros((autoenc_images.shape[0], autoenc_images.shape[1] / 3, 3))

        for r in range(autoenc_images.shape[0]):
            tmp = autoenc_images[r, :].reshape(autoenc_images.shape[1] / 3, 3)
            autoenc_images_color[r, :, 0] = tmp[:, 0]
            autoenc_images_color[r, :, 1] = tmp[:, 1]
            autoenc_images_color[r, :, 2] = tmp[:, 2]

        resized_autoenc = cv2.resize(src=autoenc_images_color, dsize=(0, 0), fx=scale_camera_factor,
                                     fy=scale_camera_factor, interpolation=cv2.INTER_NEAREST)
        cv2.imshow('autoenc', resized_autoenc)
        cv2.waitKey(500)


if __name__ == '__main__':
    #view_data()
    run_experiment()
