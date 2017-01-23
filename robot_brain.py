import pickle
from PVM.PVM_framework import MLP
import numpy as np
np.set_printoptions(suppress=True)

from fast_save_matrix import savetxt
from knn_parallel import knn_query as knn_parallel_query


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


class Autoencoder(object):
    def __init__(self, params):

        self.net = _get_mlp(num_inputs=params['num_inputs'],
                            num_hidden=params['num_hidden'],
                            num_outputs=params['num_inputs'],
                            learning_rate=params['learning_rate'])

        self.error_t = None
        self.mean_error_t = None

        self.error_history = None
        self.mean_error_history = None
        self.error_average_steps = None
        self.max_history_length = None

    def initialize_but_keep_nets(self, simulation_params):
        '''
        initializes variables, but keeps networks as they are
        :return:
        '''

        self.error_t = None
        self.mean_error_t = None

        self.error_average_steps = simulation_params['error_average_steps']
        self.max_history_length = simulation_params['max_history_length']

        self.error_history = np.zeros(self.max_history_length)
        self.mean_error_history = np.zeros(self.max_history_length)

    def evaluate_and_store_error(self, net_input):
        '''
        must evaluate net_output against net_input and store that error
        returns hidden
        :param net_input:
        :return:
        '''

        net_output = self.net.evaluate(net_input)
        hidden = self.net.layers[1]['activation'][:-1].copy()

        error = net_output - net_input
        error = np.mean(np.fabs(error))

        if self.error_t is None:
            self.error_t = 0

        self.error_history[self.error_t] = error
        self.error_t += 1

        if self.error_t > self.error_average_steps:
            if self.mean_error_t is None:
                self.mean_error_t = 0
            mean_error = np.mean(self.error_history[self.error_t - self.error_average_steps:self.error_t])
            self.mean_error_history[self.mean_error_t] = mean_error
            self.mean_error_t += 1

        return hidden

    def train(self, net_input):
        self.net.train(net_input, net_input)

    def get_mean_error_history(self):
        return self.mean_error_history[0:self.mean_error_t]


class StatesHistory(object):
    def __init__(self, params):
        self.max_history_length = params['max_history_length']
        self.states_dim_list = params['states_dim_list']
        self.state_arrays_list = []
        for k in range(len(self.states_dim_list)):
            self.state_arrays_list.append(np.zeros((self.max_history_length, self.states_dim_list[k])).astype(np.float))
        self.t = 0

    def process_new_states(self, newest_states_list):

        assert len(newest_states_list) == len(self.state_arrays_list), 'Error: invalid states length!'

        state_index = 0
        for state in newest_states_list:
            self.state_arrays_list[state_index][self.t, :] = state[:]
            state_index += 1
        self.t += 1

    def get_state(self, state_index, delay):
        state = self.state_arrays_list[state_index][self.t - 1 - delay, :]
        return state


class Predictor(object):
    def __init__(self, params):
        self.dt_output = params['dt_output']
        self.dt_context = params['dt_context']
        self.state_index_input = params['state_index_input']
        self.state_index_context = params['state_index_context']
        self.state_index_output = params['state_index_output']

        self.input_history = None
        self.output_history = None
        self.input_history_t = None

        self.error_t = None
        self.mean_error_t = None

        self.error_history = None
        self.mean_error_history = None
        self.error_average_steps = None
        self.max_history_length = None

        self.temp_array = None

    def initialize_but_keep_nets(self, simulation_params):

        self.error_t = None
        self.mean_error_t = None

        # TODO fix conflict between max_history_length (for error history here) vs. same variable for input_history
        self.error_average_steps = simulation_params['error_average_steps']
        self.max_history_length = simulation_params['max_history_length']

        self.error_history = np.zeros(self.max_history_length)
        self.mean_error_history = np.zeros(self.max_history_length)

        # TODO fix conflict between max_history_length (for error history here) vs. same variable for input_history
        self.temp_array = np.zeros(self.max_history_length).astype(np.float)

    def _get_input_context_output(self, states_history, training_delay):
        input_state = states_history.get_state(state_index=self.state_index_input, delay=training_delay + self.dt_context)
        context_state = states_history.get_state(state_index=self.state_index_context, delay=training_delay + 0)
        output_state = states_history.get_state(state_index=self.state_index_output, delay=training_delay + self.dt_context - self.dt_output)
        return input_state, context_state, output_state

    def train(self, states_history, training_delay):
        '''
        predictor must decide if it has enough history to train
        training: sets part of proper states_history array to its own knn data
        :param states_history:
        :param training_delay:
        :return:
        '''

        assert training_delay > self.dt_context, \
            'training_delay must be greater than self.dt_context ' + str(training_delay) + str(self.dt_context)
        assert self.dt_context > self.dt_output, 'dt_context must be > dt_output'

        input_state, context_state, output_state = self._get_input_context_output(states_history=states_history,
                                                                                  training_delay=training_delay)

        if input_state is not None:
            net_input = np.concatenate((input_state, context_state))
            net_output = output_state

            if self.input_history is None:
                self.input_history = np.zeros((self.max_history_length, net_input.shape[0]))
                self.output_history = np.zeros((self.max_history_length, net_output.shape[0]))
                self.input_history_t = 0

            self.input_history[self.input_history_t, :] = net_input
            self.output_history[self.input_history_t, :] = net_output
            self.input_history_t += 1

    def test_newest_point_and_store_error(self, states_history):
        '''
        predictor must decide if it has enough history to test

        :param states_history:
        :param training_delay:
        :return:
        '''

        if self.input_history_t is not None:
            data_set = self.input_history
            data_frames = self.input_history_t
            dim = data_set.shape[1]
            tmp = self.temp_array

            input_state, context_state, output_state = self._get_input_context_output(states_history=states_history,
                                                                                      training_delay=0)
            net_input = np.concatenate((input_state, context_state))
            net_output_actual = output_state

            dist, ind = knn_parallel_query(data_set, net_input, tmp, data_frames, dim)
            net_output_predicted = self.output_history[ind, :]

            error = net_output_actual - net_output_predicted
            error = np.mean(np.fabs(error))

            if self.error_t is None:
                self.error_t = 0

            self.error_history[self.error_t] = error
            self.error_t += 1

            if self.error_t > self.error_average_steps:
                if self.mean_error_t is None:
                    self.mean_error_t = 0
                mean_error = np.mean(self.error_history[self.error_t - self.error_average_steps:self.error_t])
                self.mean_error_history[self.mean_error_t] = mean_error
                self.mean_error_t += 1

    def get_mean_error_history(self):
        return self.mean_error_history[0:self.mean_error_t]


class RobotBrain(object):
    def __init__(self, params):
        self._init_globals(params)
        self.config = self._init_config(params)
        self.autoencoders_list, states_dim_list = self._init_autoencoders(params)
        if self.predictors_enable:
            self.predictors_list = self._init_predictors(params)
            self.states_history = self._init_states_history(params, states_dim_list)

    def _init_globals(self, params):
        self.t = 0
        self.predictors_enable = params['predictors_enable']

    def _init_config(self, params):
        config = {}
        config['training_delay'] = params['training_delay']

        return config

    def _init_autoencoders(self, params):

        self.autoencoders_training_time_range = params['autoencoders_training_time_range']
        self.autoencoders_save_every_k_steps = params['autoencoders_save_every_k_steps']
        self.autoencoders_enable_training = params['autoencoders_enable_training']

        if params['autoencoders_load_from_file']:
            f = open(params['autoencoders_load_filename'], 'r')
            autoencoders_list = pickle.load(f)
            f.close()
        else:
            autoencoders_list = []
            for autoenc_params in params['autoencoders']:
                autoencoders_list.append(Autoencoder(autoenc_params))

        self.autoencoder_images = np.zeros((len(autoencoders_list), params['autoencoders'][0]['num_inputs']))
        states_dim_list = [params['autoencoders'][0]['num_inputs']]
        net_index = 0
        for autoencoder in autoencoders_list:
            autoenc_sim_params = {}
            autoenc_sim_params['error_average_steps'] = params['error_average_steps']
            autoenc_sim_params['max_history_length'] = params['max_history_length']
            autoencoder.initialize_but_keep_nets(autoenc_sim_params)
            states_dim_list.append(params['autoencoders'][net_index]['num_hidden'])
            net_index += 1
        return autoencoders_list, states_dim_list

    def _init_predictors(self, params):

        self.predictors_training_time_range = params['predictors_training_time_range']
        self.predictors_test_every_k_steps = params['predictors_test_every_k_steps']
        self.predictors_save_every_k_steps = params['predictors_save_every_k_steps']
        self.predictors_enable_training = params['predictors_enable_training']

        if params['predictors_load_from_file']:
            f = open(params['predictors_load_filename'], 'r')
            predictors_list = pickle.load(f)
            f.close()
        else:
            predictors_list = []
            for predictor_params in params['predictors']:
                predictors_list.append(Predictor(predictor_params))

        for predictor in predictors_list:
            predictor_sim_params = {}
            predictor_sim_params['error_average_steps'] = params['error_average_steps']
            predictor_sim_params['max_history_length'] = params['max_history_length']
            predictor.initialize_but_keep_nets(predictor_sim_params)

        return predictors_list

    def _init_states_history(self, params, states_dim_list):
        states_history_params = {}
        states_history_params['max_history_length'] = params['max_history_length']
        states_history_params['states_dim_list'] = states_dim_list

        states_history = StatesHistory(states_history_params)
        return states_history

    def process_input(self, robot_sensors, sim_folder_manager):

        net_input = self._process_net_input(robot_sensors)

        newest_states_list = self._process_autoencoders(autoencoders_list=self.autoencoders_list,
                                                        net_input=net_input,
                                                        sim_folder_manager=sim_folder_manager)

        if self.predictors_enable:
            self._process_states_history(newest_states_list=newest_states_list,
                                         states_history=self.states_history)

            self._process_predictors(predictors_list=self.predictors_list,
                                     states_history=self.states_history,
                                     config=self.config,
                                     sim_folder_manager=sim_folder_manager
                                     )

        self.t += 1

    def _process_net_input(self, robot_sensors):
        rays = robot_sensors.get_rays()
        ray_radians = rays['ray_radians']
        ray_colors = rays['ray_colors']
        ray_lengths = rays['ray_lengths']

        net_input = ray_colors.copy()

        return net_input

    def _process_autoencoders(self, autoencoders_list, net_input, sim_folder_manager):
        newest_states_list = [net_input.copy()]

        net_index = 0
        for autoenc in autoencoders_list:
            hidden = autoenc.evaluate_and_store_error(net_input)

            next_net_input = hidden.copy()
            newest_states_list.append(hidden.copy())

            for tmp_layer in range(net_index, -1, -1):  # [2, 1, 0] for net_index = 2
                tmp_output = autoencoders_list[tmp_layer].net.evaluate_from_hidden(hidden)
                hidden = tmp_output
            self.autoencoder_images[net_index, :] = tmp_output[:]

            if self.autoencoders_enable_training:
                if self.autoencoders_training_time_range[0] <= self.t < self.autoencoders_training_time_range[1]:
                    autoenc.train(net_input)

            net_input = next_net_input
            net_index += 1

        if self.autoencoders_save_every_k_steps is not None:
            if self.t % self.autoencoders_save_every_k_steps == 0 and self.t > 0:
                print 'saving autoencoders...'
                f = open(sim_folder_manager.get_models_save_folder() + '/autoencoders.pkl', 'w')
                pickle.dump(autoencoders_list, f)
                f.close()

        return newest_states_list

    def _process_states_history(self, newest_states_list, states_history):
        states_history.process_new_states(newest_states_list)

    def _process_predictors(self, predictors_list, states_history, config, sim_folder_manager):
        '''
            training_delay: delay such that testing points (at current time) are independent of recent training
                it is otherwise possible to "cheat" if testing on the point the network was just trained on.
        '''

        training_delay = config['training_delay']

        for predictor in predictors_list:
            if self.predictors_enable_training:
                predictor.train(states_history, training_delay)

            if self.predictors_test_every_k_steps is not None:
                if self.t % self.predictors_test_every_k_steps == 0:
                    predictor.test_newest_point_and_store_error(states_history)

        if self.predictors_save_every_k_steps is not None:
            if self.t % self.predictors_save_every_k_steps == 0 and self.t > 0:
                print 'saving predictors...'
                f = open(sim_folder_manager.get_models_save_folder() + '/predictors.pkl', 'w')
                pickle.dump(predictors_list, f)
                f.close()

    # functions for other interfaces to retrieve information:

    def get_error_names_histories(self):

        error_names_autoenc = []
        error_histories_autoenc = []
        for net_index in range(len(self.autoencoders_list)):
            error_names_autoenc.append('autoencoder_' + str(net_index))
            error_histories_autoenc.append(self.autoencoders_list[net_index].get_mean_error_history())

        if self.predictors_enable:
            error_names_predictor = []
            error_histories_predictor = []
            for net_index in range(len(self.predictors_list)):
                error_names_predictor.append('predictor_' + str(net_index))
                error_histories_predictor.append(self.predictors_list[net_index].get_mean_error_history())
            #error_names_no_context_predictor = []
            #for net_index in range(len(self.predictor_networks)):
            #    error_names_no_context_predictor.append('no_context_predictor_' + str(net_index))
            #error_histories_no_context_predictor = self.averaged_no_context_predictor_error_histories[:, self.error_histories_average_steps + 1:self.no_context_predictor_error_history_step]
            error_names_no_context_predictor = None
            error_histories_no_context_predictor = None
        else:
            error_names_predictor = None
            error_histories_predictor = None
            error_names_no_context_predictor = None
            error_histories_no_context_predictor = None

        return error_names_autoenc, error_histories_autoenc, \
               error_names_predictor, error_histories_predictor,\
               error_names_no_context_predictor, error_histories_no_context_predictor

    def get_autoenc_images(self):
        return self.autoencoder_images

    def get_predictor_images(self):
        return self.ctx_predictor_debug_images

    # old / deprecated:

    def __init__OLD(self, params):
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

        self.concat_no_context_predictor_input_histories = None
        self.no_context_predictor_output_histories = None

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
        self.predictor_temp_arrays = []

        for n in range(num_nets):
            c_index_input = self.predict_nets_input_compression_levels[n]
            c_index_context = self.predict_nets_context_compression_levels[n]
            c_index_output = self.predict_nets_output_compression_levels[n]

            dim_input = compression_levels_dimensions[c_index_input]
            dim_context = compression_levels_dimensions[c_index_context]
            dim_output = compression_levels_dimensions[c_index_output]

            self.predictor_temp_arrays.append(np.zeros(max_history_length).astype(np.float))
            # create KD tree later, after some data is present
            # data storage created above
            # what happens here? context and no context predictors? nothing for now, except errors initialized:

            self.predictor_networks.append(None)
            self.no_context_predictor_networks.append(None)

            self.predictor_error_histories.append(np.zeros(max_history_length))
            self.no_context_predictor_error_histories.append(np.zeros(max_history_length))

        self.averaged_predictor_error_histories = np.zeros((len(autoenc_heirarchy_compression), max_history_length))
        self.averaged_no_context_predictor_error_histories = np.zeros((len(autoenc_heirarchy_compression), max_history_length))

        # ***** other *****

        self.last_sensory_input = None
        self.steps = 0

    def process_input_OLD(self, robot_sensors):
        rays = robot_sensors.get_rays()
        ray_radians = rays['ray_radians']
        ray_colors = rays['ray_colors']
        ray_lengths = rays['ray_lengths']

        net_input = ray_colors.copy()

        # autoenc

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

        # predictors

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

                #dist, ind = net.query([net_input], k=1)

                data_set = self.predictor_networks[net_index]
                data_frames = data_set.shape[0]
                dim = data_set.shape[1]
                tmp = self.predictor_temp_arrays[net_index]

                dist, ind = knn_parallel_query(data_set, net_input, tmp, data_frames, dim)
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

            if self.steps < self.predictor_learning_disable_step and self.predictor_training_history_step % self.predict_nets_training_interval == 0 and self.predictor_training_history_step > 2.0 * np.sum(self.predict_time_steps):
                if self.predict_nets_training_interval > 1:
                    print 'Building predictor for net: ', net_index, ' step: ', self.steps
                self.predictor_networks[net_index] = self.concat_predictor_input_histories[net_index][0:self.predictor_training_history_step - 2.0 * np.sum(self.predict_time_steps), :]
                # X : array-like, shape = [n_samples, n_features]

                #self.predictor_networks[net_index] = train(net_input, net_output)

        # saving

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

        if False:
            if self.steps % 16 == 0:
                print '----16'
            if self.steps % 32 == 0:
                print '--------32'
            if self.steps % 64 == 0:
                print '----------------64'

        self.steps += 1
