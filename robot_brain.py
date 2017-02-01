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
            self.state_arrays_list.append(np.zeros((self.max_history_length, self.states_dim_list[k])).astype(np.float32))
        self.t = 0

    def process_new_states(self, newest_states_list):

        assert len(newest_states_list) == len(self.state_arrays_list), 'Error: invalid states length!'

        state_index = 0
        for state in newest_states_list:
            self.state_arrays_list[state_index][self.t, :] = state[:]
            state_index += 1
        self.t += 1

    def get_state(self, state_index, delay):
        if self.t - 1 - delay >= 0:
            #print 'StatesHistory.get_state: self.t: ', self.t, ', index: ', self.t - 1 - delay
            state = self.state_arrays_list[state_index][self.t - 1 - delay, :]
            return state
        else:
            return None


class MotorHistory(object):
    def __init__(self, params):
        self.max_history_length = params['max_history_length']
        self.dim = params['dim']

        self.motor_array = np.zeros((self.max_history_length, self.dim)).astype(np.float32)
        self.t = 0

    def process_new_motor_command(self, motor_command):
        self.motor_array[self.t, :] = motor_command[:]
        self.t += 1

    def get_sequence(self, delay_start, delay_end):
        assert delay_start >= delay_end, 'delay_start must be >= delay_end ' + str(delay_start) + ', ' + str(delay_end)
        #print 'MotorHistory.get_sequence: self.t: ', self.t, ', range: [', self.t - 1 - delay_start, ', ', self.t - delay_end, ')'
        return_arr = self.motor_array[self.t - 1 - delay_start:self.t - delay_end, :].flatten()
        return return_arr


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
        self.temp_array = np.zeros(self.max_history_length).astype(np.float32)

        if self.input_history is not None:
            self.input_history = self.input_history.astype(np.float32)
        if self.output_history is not None:
            self.output_history = self.output_history.astype(np.float32)

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
                self.input_history = np.zeros((self.max_history_length, net_input.shape[0])).astype(np.float32)
                self.output_history = np.zeros((self.max_history_length, net_output.shape[0])).astype(np.float32)
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

            if input_state is not None:
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


class InverseModel(object):
    def __init__(self, params):
        self.state_index_current = params['state_index_current']
        self.state_index_future = params['state_index_future']
        self.dt = params['dt']

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

        self.error_average_steps = simulation_params['error_average_steps']
        self.max_history_length = simulation_params['max_history_length']

        self.error_history = np.zeros(self.max_history_length)
        self.mean_error_history = np.zeros(self.max_history_length)

        # TODO fix conflict between max_history_length (for error history here) vs. same variable for input_history
        self.temp_array = np.zeros(self.max_history_length).astype(np.float32)

        if self.input_history is not None:
            self.input_history = self.input_history.astype(np.float32)
        if self.output_history is not None:
            self.output_history = self.output_history.astype(np.float32)

    def _get_current_future_motor(self, states_history, motor_history, training_delay):
        #print '********* START _get_current_future_motor ***********'
        state_add_delay = 1
        # because:
        #   latest motor_command (T) in motor_history was initiated at T-1, applied [T-1, T],
        #   latest state (T) in states_history is at time T

        current_state = states_history.get_state(state_index=self.state_index_current,
                                                 delay=state_add_delay + training_delay + self.dt)
        future_state = states_history.get_state(state_index=self.state_index_future,
                                                delay=state_add_delay + training_delay + 0)
        motor_sequence = motor_history.get_sequence(delay_start=training_delay + self.dt,
                                                    delay_end=training_delay + 1)

        #print 'current state: delay: ', training_delay + self.dt
        #print 'future state: delay: ', training_delay + 0
        #print 'motor_sequence: delay_start: ', training_delay + self.dt, ', delay_end: ', training_delay + 1

        #print '********* END _get_current_future_motor ***********'
        # current state: delay:  129
        # future state: delay:  128
        # motor_sequence: delay_start:  129, delay_end:  129

        return current_state, future_state, motor_sequence

    def train(self, states_history, motor_history, training_delay):
        assert training_delay > self.dt, \
            'training_delay must be greater than self.dt ' + str(training_delay) + str(self.dt)

        current_state, future_state, motor_sequence = self._get_current_future_motor(states_history=states_history,
                                                                                     motor_history=motor_history,
                                                                                     training_delay=training_delay)

        if current_state is not None:
            net_input = np.concatenate((current_state, future_state))
            net_output = motor_sequence

            if self.input_history is None:
                self.input_history = np.zeros((self.max_history_length, net_input.shape[0])).astype(np.float32)
                self.output_history = np.zeros((self.max_history_length, net_output.shape[0])).astype(np.float32)
                self.input_history_t = 0

            self.input_history[self.input_history_t, :] = net_input
            self.output_history[self.input_history_t, :] = net_output
            self.input_history_t += 1

    def test_newest_point_and_store_error(self, states_history, motor_history):
        if self.input_history_t is not None:
            data_set = self.input_history
            data_frames = self.input_history_t
            dim = data_set.shape[1]
            tmp = self.temp_array

            current_state, future_state, motor_sequence = self._get_current_future_motor(states_history=states_history,
                                                                                         motor_history=motor_history,
                                                                                         training_delay=0)

            if current_state is not None:
                net_input = np.concatenate((current_state, future_state))
                net_output_actual = motor_sequence

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

    # TODO for inverse model, mean error histories, add to plots!


class RobotBrain(object):
    def __init__(self, params):
        self._init_globals(params)
        self.config = self._init_config(params)
        self.autoencoders_list, states_dim_list = self._init_autoencoders(params)
        if self.predictors_enable:
            self.predictors_list = self._init_predictors(params)
            self.states_history = self._init_states_history(params, states_dim_list)
            self.motor_history = self._init_motor_history(params)
            self.inverse_list = self._init_inverse(params)

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
            print 'loading autoencoders...'
            f = open(params['autoencoders_load_filename'], 'r')
            autoencoders_list = pickle.load(f)
            f.close()
            print 'done.'
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
            print 'loading predictors...'
            f = open(params['predictors_load_filename'], 'r')
            predictors_list = pickle.load(f)
            f.close()
            print 'done.'
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

    def _init_motor_history(self, params):
        motor_history_params = {}
        motor_history_params['max_history_length'] = params['max_history_length']
        motor_history_params['dim'] = 2

        motor_history = MotorHistory(motor_history_params)
        return motor_history

    def _init_inverse(self, params):
        self.inverse_training_time_range = params['inverse_training_time_range']
        self.inverse_test_every_k_steps = params['inverse_test_every_k_steps']
        self.inverse_save_every_k_steps = params['inverse_save_every_k_steps']
        self.inverse_enable_training = params['inverse_enable_training']

        if params['inverse_load_from_file']:
            print 'loading inverse...'
            f = open(params['inverse_load_filename'], 'r')
            inverse_list = pickle.load(f)
            f.close()
            print 'done.'
        else:
            inverse_list = []
            for inverse_params in params['inverse_models']:
                inverse_list.append(InverseModel(inverse_params))

        for inverse in inverse_list:
            inverse_sim_params  = {}
            inverse_sim_params['error_average_steps'] = params['error_average_steps']
            inverse_sim_params['max_history_length'] = params['max_history_length']
            inverse.initialize_but_keep_nets(inverse_sim_params)

        return inverse_list

    # ************ process ************

    def process_input(self, robot_sensors, sim_folder_manager):

        current_visual_input, previous_motor_command = self._process_sensors(robot_sensors)
        # previous_motor_command was initiated at T-1, applied [T-1, T],
        # current_visual_input is at time T

        newest_states_list = self._process_autoencoders(autoencoders_list=self.autoencoders_list,
                                                        net_input=current_visual_input,
                                                        sim_folder_manager=sim_folder_manager)

        if self.predictors_enable:
            self._process_states_history(newest_states_list=newest_states_list,
                                         states_history=self.states_history)

            self._process_motor_history(newest_motor_command=previous_motor_command,
                                        motor_history=self.motor_history)

            self._process_predictors(predictors_list=self.predictors_list,
                                     states_history=self.states_history,
                                     config=self.config,
                                     sim_folder_manager=sim_folder_manager
                                     )

            self._process_inverse(inverse_list=self.inverse_list,
                                  states_history=self.states_history,
                                  motor_history=self.motor_history,
                                  config=self.config,
                                  sim_folder_manager=sim_folder_manager
                                  )

        self.t += 1

    def _process_sensors(self, robot_sensors):
        rays = robot_sensors.get_rays()
        previous_motor_command = robot_sensors.get_last_motor_command()
        ray_radians = rays['ray_radians']
        ray_colors = rays['ray_colors']
        ray_lengths = rays['ray_lengths']

        current_visual_input = ray_colors.copy()

        return current_visual_input, previous_motor_command

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

    def _process_motor_history(self, newest_motor_command, motor_history):
        motor_history.process_new_motor_command(newest_motor_command)

    def _process_inverse(self, inverse_list, states_history, motor_history, config, sim_folder_manager):

        training_delay = config['training_delay']

        for inverse in inverse_list:
            if self.inverse_enable_training:
                inverse.train(states_history, motor_history, training_delay)

            if self.inverse_test_every_k_steps is not None:
                if self.t % self.inverse_test_every_k_steps == 0 and self.t > 0:
                    inverse.test_newest_point_and_store_error(states_history, motor_history)

        if self.inverse_save_every_k_steps is not None:
            if self.t % self.inverse_save_every_k_steps == 0 and self.t > 0:
                print 'saving inverse models...'
                f = open(sim_folder_manager.get_models_save_folder() + '/inverse.pkl', 'w')
                pickle.dump(inverse_list, f)
                f.close()

    # ************ functions for other interfaces to retrieve information ************

    def get_error_names_histories(self):

        error_names_autoenc = []
        error_histories_autoenc = []
        for net_index in range(len(self.autoencoders_list)):
            error_names_autoenc.append('autoencoder_' + str(net_index))
            error_histories_autoenc.append(self.autoencoders_list[net_index].get_mean_error_history())

        if self.predictors_enable:
            error_names_predictor = []
            error_histories_predictor = []
            error_names_inverse = []
            error_histories_inverse = []

            for net_index in range(len(self.predictors_list)):
                error_names_predictor.append('predictor_' + str(net_index))
                error_histories_predictor.append(self.predictors_list[net_index].get_mean_error_history())
            #error_names_no_context_predictor = []
            #for net_index in range(len(self.predictor_networks)):
            #    error_names_no_context_predictor.append('no_context_predictor_' + str(net_index))
            #error_histories_no_context_predictor = self.averaged_no_context_predictor_error_histories[:, self.error_histories_average_steps + 1:self.no_context_predictor_error_history_step]

            for net_index in range(len(self.inverse_list)):
                error_names_inverse.append('inverse_' + str(net_index))
                error_histories_inverse.append(self.inverse_list[net_index].get_mean_error_history())

            error_names_no_context_predictor = None
            error_histories_no_context_predictor = None
        else:
            error_names_predictor = None
            error_histories_predictor = None
            error_names_inverse = None
            error_histories_inverse = None
            error_names_no_context_predictor = None
            error_histories_no_context_predictor = None

        return error_names_autoenc, error_histories_autoenc, \
               error_names_predictor, error_histories_predictor, \
               error_names_inverse, error_histories_inverse, \
               error_names_no_context_predictor, error_histories_no_context_predictor

    def get_autoenc_images(self):
        return self.autoencoder_images

    def get_predictor_images(self):
        return self.ctx_predictor_debug_images






























