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
        self.learning_rate = params['learning_rate']
        self.save_folder = params['save_folder']
        self.learning_disable_step = params['learning_disable_step']

        layer_index = 0
        layer_num_inputs = num_sensory_inputs

        self.num_autoenc_networks = len(autoenc_heirarchy_compression)
        self.autoenc_networks = []

        self.autoenc_error_histories = []
        self.autoenc_error_history_step = 0
        self.error_histories_average_steps = error_average_steps

        for ratio in autoenc_heirarchy_compression:
            layer_num_hidden = int(ratio * layer_num_inputs)
            layer_num_outputs = layer_num_inputs

            print 'layer ', layer_index, '[' + str(layer_num_inputs) + '] -> [' + str(layer_num_hidden) + '] -> [' + str(layer_num_outputs) + ']'
            self.autoenc_networks.append(_get_mlp(num_inputs=layer_num_inputs,
                                                  num_hidden=layer_num_hidden,
                                                  num_outputs=layer_num_outputs,
                                                  learning_rate=self.learning_rate))

            layer_index += 1
            layer_num_inputs = layer_num_hidden

            self.autoenc_error_histories.append(np.zeros(max_history_length))

        self.averaged_autoenc_error_histories = np.zeros((len(autoenc_heirarchy_compression), max_history_length))
        self.autoenc_images = np.zeros((len(autoenc_heirarchy_compression), num_sensory_inputs))

        self.last_sensory_input = None
        self.steps = 0

    def get_error_names_histories(self):
        error_names = []
        for net_index in range(len(self.autoenc_networks)):
            error_names.append('autoenc_' + str(net_index))
        error_histories = self.averaged_autoenc_error_histories[:, self.error_histories_average_steps:self.autoenc_error_history_step]

        return error_names, error_histories

    def get_autoenc_images(self):
        return self.autoenc_images

    def process_input(self, robot_sensors):
        rays = robot_sensors.get_rays()
        ray_radians = rays['ray_radians']
        ray_colors = rays['ray_colors']
        ray_lengths = rays['ray_lengths']

        net_input = ray_colors.copy()
        net_index = 0

        #print '************************'
        for net in self.autoenc_networks:

            #if net_index == 2:
            #    print 'net_input begin: ' + str(net_input)

            # TODO why did copy here not fix it???
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

            #if net_index == 2:
            #    print 'net_input before: ' + str(net_input)

            if True:
                # TODO added copy here but that didn't solve it
                hidden = net.layers[1]['activation'][:-1].copy()

                for tmp_layer in range(net_index, -1, -1):  # [2, 1, 0] for net_index = 2
                    tmp_output = self.autoenc_networks[tmp_layer].evaluate_from_hidden(hidden)
                    hidden = tmp_output

                # display tmp_output
                #print net_index, tmp_output.shape
                self.autoenc_images[net_index, :] = tmp_output[:]

            #if net_index == 2:
            #    print 'net_input after: ' + str(net_input)

            if self.steps < self.learning_disable_step:
                net.train(net_input, net_input)

            net_index += 1
            # THIS WAS WHERE COPY WAS NECESSARY TO AVOID BUG
            net_input = net.layers[1]['activation'][:-1].copy()

        if self.steps % self.save_steps == 0:
            print 'saving autoencoders...'
            f = open(self.save_folder + '/autoencoders.pkl', 'w')
            pickle.dump(self.autoenc_networks, f)
            f.close()
            print 'done saving autoencoders.'

        self.autoenc_error_history_step += 1
        #self.last_sensory_input = current_sensor_input

        self.steps += 1
