import pickle
import numpy as np
np.set_printoptions(suppress=True)
from fast_save_matrix import savetxt
from brain_components import Autoencoder, StatesHistory, MotorHistory, Predictor, InverseModel


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
        self.motor_out = None
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

    def process_input(self, rays, last_motor_command, goal_states, models_save_folder):

        current_visual_input = self._process_sensors(rays=rays)
        # previous_motor_command was initiated at T-1, applied [T-1, T],
        # current_visual_input is at time T

        newest_states_list = self._process_autoencoders(autoencoders_list=self.autoencoders_list,
                                                        net_input=current_visual_input,
                                                        models_save_folder=models_save_folder)

        if self.predictors_enable:
            self._process_states_history(newest_states_list=newest_states_list,
                                         states_history=self.states_history)

            self._process_motor_history(newest_motor_command=last_motor_command,
                                        motor_history=self.motor_history)

            self._process_predictors(predictors_list=self.predictors_list,
                                     states_history=self.states_history,
                                     config=self.config,
                                     models_save_folder=models_save_folder
                                     )

            self._process_inverse(inverse_list=self.inverse_list,
                                  states_history=self.states_history,
                                  motor_history=self.motor_history,
                                  config=self.config,
                                  models_save_folder=models_save_folder
                                  )

        if goal_states is not None:
            inv = self.inverse_list[0]  # TODO select inverse model here
            self.motor_out = inv.lookup_motor_to_goal(goal_states, self.states_history)
            print 'motor_out, no index: ', self.motor_out
            self.motor_out = self.motor_out[1]  # TODO this depends on which inverse model?
        else:
            self.motor_out = None

        self.t += 1

    def get_motor_output(self):
        return self.motor_out

    def _process_sensors(self, rays):
        ray_radians = rays['ray_radians']
        ray_colors = rays['ray_colors']
        ray_lengths = rays['ray_lengths']

        current_visual_input = ray_colors.copy()

        return current_visual_input

    def _process_autoencoders(self, autoencoders_list, net_input, models_save_folder, include_in_history=True):
        newest_states_list = [net_input.copy()]

        net_index = 0
        for autoenc in autoencoders_list:
            hidden = autoenc.evaluate_and_store_error(net_input, store_error=include_in_history)

            next_net_input = hidden.copy()
            newest_states_list.append(hidden.copy())

            for tmp_layer in range(net_index, -1, -1):  # [2, 1, 0] for net_index = 2
                tmp_output = autoencoders_list[tmp_layer].net.evaluate_from_hidden(hidden)
                hidden = tmp_output
            if include_in_history:
                self.autoencoder_images[net_index, :] = tmp_output[:]

                if self.autoencoders_enable_training:
                    if self.autoencoders_training_time_range[0] <= self.t < self.autoencoders_training_time_range[1]:
                        autoenc.train(net_input)

            net_index += 1

            net_input = next_net_input

        if include_in_history:
            if self.autoencoders_save_every_k_steps is not None:
                if self.t % self.autoencoders_save_every_k_steps == 0 and self.t > 0:
                    print 'saving autoencoders...'
                    f = open(models_save_folder + '/autoencoders.pkl', 'w')
                    pickle.dump(autoencoders_list, f)
                    f.close()

        return newest_states_list

    def _process_states_history(self, newest_states_list, states_history):
        states_history.process_new_states(newest_states_list)

    def _process_predictors(self, predictors_list, states_history, config, models_save_folder):
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
                f = open(models_save_folder + '/predictors.pkl', 'w')
                pickle.dump(predictors_list, f)
                f.close()

    def _process_motor_history(self, newest_motor_command, motor_history):
        motor_history.process_new_motor_command(newest_motor_command)

    def _process_inverse(self, inverse_list, states_history, motor_history, config, models_save_folder):

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
                f = open(models_save_folder + '/inverse.pkl', 'w')
                pickle.dump(inverse_list, f)
                f.close()

    # ************ functions for other interfaces to retrieve information ************
    def get_autoencoder_states_for_input(self, net_input):
        newest_states_list = self._process_autoencoders(autoencoders_list=self.autoencoders_list,
                                                        net_input=net_input,
                                                        models_save_folder=None,
                                                        include_in_history=False)
        return newest_states_list

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






























