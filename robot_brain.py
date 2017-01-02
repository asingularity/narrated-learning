import pickle
from PVM.PVM_framework import MLP
import numpy as np
np.set_printoptions(suppress=True)
#from sklearn.neighbors import KDTree
from brute_force_knn import knn as KDTree


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
        max_history_length = params['max_history_length']
        self.save_steps = params['save_steps']
        self.autoenc_learning_rate = params['autoenc_learning_rate']
        self.save_folder = params['save_folder']
        self.test_predictor_every_k_steps = params['test_predictor_every_k_steps']
        layer_index = 0
        layer_num_inputs = num_sensory_inputs

        self.num_autoenc_networks = len(autoenc_heirarchy_compression)
        self.autoenc_networks = []

        self.autoenc_error_histories = []
        self.autoenc_error_history_step = 0
        self.error_histories_average_steps = error_average_steps

        # ***** autoenc nets *****
        compression_levels_dimensions = [num_sensory_inputs]
        self.compression_levels_dimensions = compression_levels_dimensions

        for ratio in autoenc_heirarchy_compression:
            layer_num_hidden = int(ratio * layer_num_inputs)
            layer_num_outputs = layer_num_inputs
            compression_levels_dimensions.append(layer_num_hidden)

            print 'autoenc layer ', layer_index, '[' + str(layer_num_inputs) + '] -> [' + str(layer_num_hidden) + '] -> [' + str(layer_num_outputs) + ']'

            if not params['load_autoenc_from_file']:
                self.autoenc_networks.append(_get_mlp(num_inputs=layer_num_inputs,
                                                      num_hidden=layer_num_hidden,
                                                      num_outputs=layer_num_outputs,
                                                      learning_rate=self.autoenc_learning_rate))

            layer_index += 1
            layer_num_inputs = layer_num_hidden

            self.autoenc_error_histories.append(np.zeros(max_history_length))

        self.max_history_length = max_history_length
        self.averaged_autoenc_error_histories = np.zeros((len(autoenc_heirarchy_compression), max_history_length))
        self.autoenc_images = np.zeros((len(autoenc_heirarchy_compression), num_sensory_inputs))
        self.ctx_predictor_debug_images = np.zeros((5, num_sensory_inputs))

        if params['load_autoenc_from_file']:
            self.autoenc_learning_disable_step = -1
            f = open(params['load_autoenc_filename'], 'r')
            self.autoenc_networks = pickle.load(f)
            f.close()
        else:
            self.autoenc_learning_disable_step = params['autoenc_learning_disable_step']

        # ***** prediction nets *****

        self.predict_time_steps = params['predict_time_steps']
        self.predict_nets_input_compression_levels = params['predict_nets_input_compression_levels']
        self.predict_nets_context_compression_levels = params['predict_nets_context_compression_levels']
        self.predict_nets_output_compression_levels = params['predict_nets_output_compression_levels']
        self.predictor_learning_disable_step = params['predict_nets_learning_disable_step']
        self.predict_nets_training_interval = params['predict_nets_training_interval']
        self.concat_predictor_input_histories = None
        self.predictor_output_histories = None

        num_nets = len(self.predict_nets_input_compression_levels)
        self.predictor_networks = []
        self.no_context_predictor_networks = []

        # TODO: this needs to be all history
        self.predictor_training_history_length = max_history_length
        # np.sum(np.array(self.predict_time_steps))
        self.predictor_training_histories = []
        for dim in compression_levels_dimensions:
            self.predictor_training_histories.append(np.zeros((self.predictor_training_history_length, dim)))
        self.predictor_training_history_step = 0

        self.predictor_error_histories = []
        self.predictor_error_history_step = 0
        self.no_context_predictor_error_histories = []
        self.no_context_predictor_error_history_step = 0

        for n in range(num_nets):
            c_index_input = self.predict_nets_input_compression_levels[n]
            c_index_context = self.predict_nets_context_compression_levels[n]
            c_index_output = self.predict_nets_output_compression_levels[n]

            dim_input = compression_levels_dimensions[c_index_input]
            dim_context = compression_levels_dimensions[c_index_context]
            dim_output = compression_levels_dimensions[c_index_output]

            # create KD tree later, after some data is present
            # data storage created above
            # what happens here? context and no context predictors? nothing for now, except errors initialized:

            self.predictor_networks.append(None)
            self.predictor_error_histories.append(np.zeros(max_history_length))
            self.no_context_predictor_error_histories.append(np.zeros(max_history_length))

        self.averaged_predictor_error_histories = np.zeros((len(autoenc_heirarchy_compression), max_history_length))
        self.averaged_no_context_predictor_error_histories = np.zeros((len(autoenc_heirarchy_compression), max_history_length))

        # ***** other *****

        self.last_sensory_input = None
        self.steps = 0

    def get_error_names_histories(self):

        error_names_autoenc = []
        for net_index in range(len(self.autoenc_networks)):
            error_names_autoenc.append('autoenc_' + str(net_index))
        error_histories_autoenc = self.averaged_autoenc_error_histories[:, self.error_histories_average_steps + 1:self.autoenc_error_history_step]

        error_names_predictor = []
        for net_index in range(len(self.predictor_networks)):
            error_names_predictor.append('predictor_' + str(net_index))
        error_histories_predictor = self.averaged_predictor_error_histories[:, self.error_histories_average_steps + 1:self.predictor_error_history_step]

        error_names_no_context_predictor = []
        for net_index in range(len(self.predictor_networks)):
            error_names_no_context_predictor.append('no_context_predictor_' + str(net_index))
        error_histories_no_context_predictor = self.averaged_no_context_predictor_error_histories[:, self.error_histories_average_steps + 1:self.no_context_predictor_error_history_step]

        return error_names_autoenc, error_histories_autoenc, \
               error_names_predictor, error_histories_predictor,\
               error_names_no_context_predictor, error_histories_no_context_predictor

    def get_autoenc_images(self):
        return self.autoenc_images

    def get_predictor_images(self):
        return self.ctx_predictor_debug_images

    def process_input(self, robot_sensors):
        rays = robot_sensors.get_rays()
        ray_radians = rays['ray_radians']
        ray_colors = rays['ray_colors']
        ray_lengths = rays['ray_lengths']

        net_input = ray_colors.copy()
        net_index = 0
        self.predictor_training_histories[0][self.predictor_training_history_step, :] = net_input[:]

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
            self.predictor_training_histories[net_index + 1][self.predictor_training_history_step, :] = net_input[:]
            net_index += 1

        self.predictor_training_history_step += 1
        self.autoenc_error_history_step += 1

        num_nets = len(self.predictor_networks)

        for net_index in range(num_nets):
            dt_output = self.predict_time_steps[net_index]
            dt_context = self.predict_time_steps[net_index + 1]
            c_index_input = self.predict_nets_input_compression_levels[net_index]
            c_index_context = self.predict_nets_context_compression_levels[net_index]
            c_index_output = self.predict_nets_output_compression_levels[net_index]
            t_input = self.predictor_training_history_step - dt_context - 1

            net_input = np.concatenate((self.predictor_training_histories[c_index_input][t_input, :],
                                        self.predictor_training_histories[c_index_context][t_input + dt_context, :]))

            net_output = self.predictor_training_histories[c_index_output][t_input + dt_output, :]

            if self.concat_predictor_input_histories is None:
                self.concat_predictor_input_histories = [None] * num_nets

            if self.concat_predictor_input_histories[net_index] is None:
                self.concat_predictor_input_histories[net_index] = np.zeros((self.max_history_length, net_input.shape[0]))

            debug_print_every_step = False
            if debug_print_every_step:
                print 'input level: ', c_index_input, 'context level: ', c_index_context, 'output level: ', c_index_output
                print 'input time: ', t_input, 'context time: ', t_input + dt_context, 'output time: ', t_input + dt_output
                print 'net_input: ', net_index, net_input.shape[0], net_input
                print 'net_input: ', net_index, net_output.shape[0], net_output

            self.concat_predictor_input_histories[net_index][self.predictor_training_history_step, :] = net_input[:]

            if self.predictor_output_histories is None:
                self.predictor_output_histories = [None] * num_nets

            if self.predictor_output_histories[net_index] is None:
                self.predictor_output_histories[net_index] = np.zeros((self.max_history_length, net_output.shape[0]))

            self.predictor_output_histories[net_index][self.predictor_training_history_step, :] = net_output[:]

            net = self.predictor_networks[net_index]

            if (net is not None) and (self.steps % self.test_predictor_every_k_steps == 0):

                dist, ind = net.query([net_input], k=1)
                #print 'query: ', dist, ind

                #net_output_eval = net.evaluate(net_input)
                # TODO verify proper index here!!!!!!!!!!!!!!!!!!!!!
                net_output_eval = self.predictor_output_histories[net_index][ind, :]

                error = net_output - net_output_eval
                error = np.mean(np.fabs(error))
                e_step = self.predictor_error_history_step
                self.predictor_error_histories[net_index][e_step] = error

                if net_index == num_nets - 1:
                    self.predictor_error_history_step += 1

                enable_ctx_predictor_debug = False
                if net_index == 0 and enable_ctx_predictor_debug:
                    self.ctx_predictor_debug_images[0, :] = self.predictor_training_histories[c_index_input][t_input, :]
                    self.ctx_predictor_debug_images[1, :] = self.predictor_training_histories[c_index_input][t_input + dt_context, :]
                    self.ctx_predictor_debug_images[2, :] = net_output[:]
                    self.ctx_predictor_debug_images[3, :] = net_output_eval[:]

                if e_step > self.error_histories_average_steps:
                    e_ave = self.error_histories_average_steps
                    mean_error = np.mean(self.predictor_error_histories[net_index][e_step - e_ave:e_step])
                    self.averaged_predictor_error_histories[net_index][e_step] = mean_error

            if self.steps < self.predictor_learning_disable_step and self.predictor_training_history_step % self.predict_nets_training_interval == 0:
                # TODO build KD tree here! full list not just current input, output!
                if self.predict_nets_training_interval > 1:
                    print 'Building KD Tree for net: ', net_index, ' step: ', self.steps
                self.predictor_networks[net_index] = KDTree(self.concat_predictor_input_histories[net_index][0:self.predictor_training_history_step, :])
                # X : array-like, shape = [n_samples, n_features]

                #self.predictor_networks[net_index] = train(net_input, net_output)

        # TODO NO CONTEXT:
        if self.save_steps is not None:
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
