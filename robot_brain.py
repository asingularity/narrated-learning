import pickle
from PVM.PVM_framework import MLP
import numpy as np


def _get_mlp(num_inputs, num_hidden, num_outputs, learning_rate):
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


class RobotBrain(object):
    def __init__(self, params):
        sensors_params = params['sensors_params']
        num_sensory_inputs = sensors_params['num_rays']
        autoenc_heirarchy_compression = params['autoenc_heirarchy_compression']
        error_average_steps = params['error_average_steps']
        max_history_length = 100000000
        self.save_steps = params['save_steps']
        self.autoenc_learning_rate = params['autoenc_learning_rate']
        self.save_folder = params['save_folder']
        self.autoenc_learning_disable_step = params['autoenc_learning_disable_step']

        layer_index = 0
        layer_num_inputs = num_sensory_inputs

        self.num_autoenc_networks = len(autoenc_heirarchy_compression)
        self.autoenc_networks = []

        self.autoenc_error_histories = []
        self.autoenc_error_history_step = 0
        self.error_histories_average_steps = error_average_steps

        # ***** autoenc nets *****
        compression_levels_dimensions = [num_sensory_inputs]

        for ratio in autoenc_heirarchy_compression:
            layer_num_hidden = int(ratio * layer_num_inputs)
            layer_num_outputs = layer_num_inputs
            compression_levels_dimensions.append(layer_num_hidden)

            print 'autoenc layer ', layer_index, '[' + str(layer_num_inputs) + '] -> [' + str(layer_num_hidden) + '] -> [' + str(layer_num_outputs) + ']'
            self.autoenc_networks.append(_get_mlp(num_inputs=layer_num_inputs,
                                                  num_hidden=layer_num_hidden,
                                                  num_outputs=layer_num_outputs,
                                                  learning_rate=self.autoenc_learning_rate))

            layer_index += 1
            layer_num_inputs = layer_num_hidden

            self.autoenc_error_histories.append(np.zeros(max_history_length))

        self.averaged_autoenc_error_histories = np.zeros((len(autoenc_heirarchy_compression), max_history_length))
        self.autoenc_images = np.zeros((len(autoenc_heirarchy_compression), num_sensory_inputs))

        # ***** prediction nets *****

        self.predict_time_steps = params['predict_time_steps']
        self.predict_nets_input_compression_levels = params['predict_nets_input_compression_levels']
        self.predict_nets_context_compression_levels = params['predict_nets_context_compression_levels']
        self.predict_nets_output_compression_levels = params['predict_nets_output_compression_levels']
        self.predict_nets_learning_rate = params['predict_nets_learning_rate']
        self.predictor_learning_disable_step = params['predict_nets_learning_disable_step']

        num_nets = len(self.predict_nets_input_compression_levels)
        self.predictor_networks = []

        self.predictor_training_history_length = np.sum(np.array(self.predict_time_steps))
        self.predictor_training_histories = []
        for dim in compression_levels_dimensions:
            self.predictor_training_histories.append(np.zeros((self.predictor_training_history_length, dim)))

        self.predictor_error_histories = []
        self.predictor_error_history_step = 0

        for n in range(num_nets):
            c_index_input = self.predict_nets_input_compression_levels[n]
            c_index_context = self.predict_nets_context_compression_levels[n]
            c_index_output = self.predict_nets_output_compression_levels[n]

            dim_input = compression_levels_dimensions[c_index_input]
            dim_context = compression_levels_dimensions[c_index_context]
            dim_output = compression_levels_dimensions[c_index_output]

            dim_hidden = params['predict_nets_hidden_dim']

            # 0 [40], 1 [20] -> 0 [40]
            print 'predictor layer ', layer_index, '[' + str(dim_input) + '], [' + str(dim_context) + '] -> (' + str(dim_hidden) + ') -> [' + str(dim_output) + ']'

            self.predictor_networks.append(_get_mlp(num_inputs=dim_input + dim_context,
                                                    num_hidden=dim_hidden,
                                                    num_outputs=dim_output,
                                                    learning_rate=self.predict_nets_learning_rate))

            self.predictor_error_histories.append(np.zeros(max_history_length))

        self.averaged_predictor_error_histories = np.zeros((len(autoenc_heirarchy_compression), max_history_length))

        # ***** other *****

        self.last_sensory_input = None
        self.steps = 0

    def get_error_names_histories(self):
        error_names_autoenc = []
        for net_index in range(len(self.autoenc_networks)):
            error_names_autoenc.append('autoenc_' + str(net_index))
        error_histories_autoenc = self.averaged_autoenc_error_histories[:, self.error_histories_average_steps:self.autoenc_error_history_step]
        error_names_predictor = []
        for net_index in range(len(self.predictor_networks)):
            error_names_predictor.append('predictor_' + str(net_index))
        error_histories_predictor = self.averaged_predictor_error_histories[:, self.error_histories_average_steps:self.predictor_error_history_step]

        return error_names_autoenc, error_histories_autoenc, error_names_predictor, error_histories_predictor

    def get_autoenc_images(self):
        return self.autoenc_images

    def process_input(self, robot_sensors):
        rays = robot_sensors.get_rays()
        ray_radians = rays['ray_radians']
        ray_colors = rays['ray_colors']
        ray_lengths = rays['ray_lengths']

        for m_index in range(len(self.predictor_training_histories)):
            self.predictor_training_histories[m_index] = np.roll(self.predictor_training_histories[m_index], 1, 0)

        net_input = ray_colors.copy()
        net_index = 0
        self.predictor_training_histories[0][0, :] = net_input[:]

        for net in self.autoenc_networks:
            net_output = net.evaluate(net_input)

            error = net_output - net_input
            error = np.mean(np.fabs(error))

            e_step = self.autoenc_error_history_step
            self.autoenc_error_histories[net_index][e_step] = error
            if e_step > self.error_histories_average_steps:
                e_ave = self.error_histories_average_steps
                mean_error = np.mean(self.autoenc_error_histories[net_index][e_step - e_ave:e_step])
                self.averaged_autoenc_error_histories[net_index][e_step] = mean_error

            # evaluate hidden -> out
            # technically we could skip one redundant compute per layer (initial hidden->output is given above)

            hidden = net.layers[1]['activation'][:-1].copy()

            for tmp_layer in range(net_index, -1, -1):  # [2, 1, 0] for net_index = 2
                tmp_output = self.autoenc_networks[tmp_layer].evaluate_from_hidden(hidden)
                hidden = tmp_output
            self.autoenc_images[net_index, :] = tmp_output[:]

            if self.steps < self.autoenc_learning_disable_step:
                net.train(net_input, net_input)

            # copy necessary here
            net_input = net.layers[1]['activation'][:-1].copy()
            self.predictor_training_histories[net_index + 1][0, :] = net_input[:]
            net_index += 1

        self.autoenc_error_history_step += 1

        net_index = 0
        for net in self.predictor_networks:
            dt = self.predict_time_steps[net_index]
            dt_context = self.predict_time_steps[net_index + 1]
            c_index_input = self.predict_nets_input_compression_levels[net_index]
            c_index_context = self.predict_nets_context_compression_levels[net_index]
            c_index_output = self.predict_nets_output_compression_levels[net_index]

            net_input = np.concatenate((self.predictor_training_histories[c_index_input][0, :],
                                        self.predictor_training_histories[c_index_context][dt + dt_context, :]))
            net_output = self.predictor_training_histories[c_index_output][dt, :]

            # TODO evaluate first, store error
            net_output_eval = net.evaluate(net_input)
            error = net_output - net_output_eval
            error = np.mean(np.fabs(error))

            e_step = self.predictor_error_history_step
            self.predictor_error_histories[net_index][e_step] = error
            if e_step > self.error_histories_average_steps:
                e_ave = self.error_histories_average_steps
                mean_error = np.mean(self.predictor_error_histories[net_index][e_step - e_ave:e_step])
                self.averaged_predictor_error_histories[net_index][e_step] = mean_error

            if self.steps < self.predictor_learning_disable_step:
                net.train(net_input, net_output)

            net_index += 1

        self.predictor_error_history_step += 1

        if self.steps % self.save_steps == 0:
            print 'saving autoencoders...'
            f = open(self.save_folder + '/autoencoders.pkl', 'w')
            pickle.dump(self.autoenc_networks, f)
            f.close()
            print 'done saving autoencoders.'
            print 'saving predictors...'
            f = open(self.save_folder + '/predictors.pkl', 'w')
            pickle.dump(self.predictor_networks, f)
            f.close()
            print 'done saving predictors.'

        #self.last_sensory_input = current_sensor_input

        self.steps += 1
