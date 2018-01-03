import pickle
import numpy as np
np.set_printoptions(suppress=True)
from fast_save_matrix import savetxt
from brain_components import Autoencoder, StatesHistory, MotorHistory, Predictor, InverseModel, DebugTopdownInfoHistory


class RobotBrain(object):
    def __init__(self, params):
        self._init_globals(params)
        self.config = self._init_config(params)
        self.autoencoders_list, states_dim_list = self._init_autoencoders(params)
        if self.predictors_enable:
            self.predictors_list = self._init_predictors(params)
            self.states_history = self._init_states_history(params, states_dim_list)
            self.debug_topdown_info_history = self._init_debug_topdown_info_history(params)
            self.motor_history = self._init_motor_history(params)
            self.inverse_list = self._init_inverse(params)

    def _init_globals(self, params):
        self.t = 0
        self.motor_out = None
        self.predictors_enable = params['predictors_enable']
        self.use_advanced_exploration = params['use_advanced_exploration']
        self.max_history_length = params['max_history_length']  # only needed for experimental exploration
        self.explore_mode_switch_time = params['explore_mode_switch_time']
        # for task
        self.last_time_set_8 = -9e4
        self.last_time_set_16 = -9e4
        self.last_time_set_32 = -9e4
        self.dist_2 = 0
        self.dist_1 = 0
        self.dist_0 = 0
        self.always_random_motor_out = False

        self.sum_p1_errors_for_task = 0.0
        self.num_p1_errors_for_task = 0


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
        self.predictors_optimize_training = params['predictors_optimize_training']

        assert self.predictors_enable_training is False or self.predictors_optimize_training is False

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

    def _init_debug_topdown_info_history(self, params):
        debug_topdown_info_history_params = {}
        debug_topdown_info_history_params['max_history_length'] = params['max_history_length']
        debug_topdown_info_history = DebugTopdownInfoHistory(debug_topdown_info_history_params)
        return debug_topdown_info_history

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

    def process_input(self, rays, last_motor_command, goal_states, models_save_folder, debug_topdown_info):

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

            self._process_debug_topdown_info_history(newest_topdown_info=debug_topdown_info,
                                                     debug_topdown_info_history=self.debug_topdown_info_history)

            self._process_predictors(predictors_list=self.predictors_list,
                                     states_history=self.states_history,
                                     config=self.config,
                                     models_save_folder=models_save_folder,
                                     debug_topdown_info_history=self.debug_topdown_info_history
                                     )

            self._process_inverse(inverse_list=self.inverse_list,
                                  states_history=self.states_history,
                                  motor_history=self.motor_history,
                                  config=self.config,
                                  models_save_folder=models_save_folder
                                  )

        if (goal_states is not None) and (not self.always_random_motor_out):

            use_recurrence = False
            if use_recurrence:
                proximal_goal_states = self._get_proximal_goal_states(goal_states=goal_states,
                                                                      states_history=self.states_history,  # includes current input?
                                                                      predictors_list=self.predictors_list)
                                                                      # TODO needs ongoing memory for running online?

                # proximal goal states could be [v0, None, None, None], or [None, v1, None, None] etc.
                # it doesn't necessarily have every level complete, just what the most proximal predictor provides

                inv = self.inverse_list[0]  # TODO select inverse model here
                self.motor_out = inv.lookup_motor_to_goal(goal_states=proximal_goal_states,
                                                          states_history=self.states_history)
                #print 'motor_out, no index: ', self.motor_out
                self.motor_out = self.motor_out[1]  # [v, w; v, w; v, w]
            else:

                # todo test predictor here?
                p1_error_for_task = self.predictors_list[1].get_input_context_error(input_state=self.states_history.get_state(state_index=0, delay=0),
                                                                                    context_state=goal_states[0])
                #print 'input state: ', self.states_history.get_state(state_index=0, delay=0)
                #print 'context state: ', goal_states[0]
                #print 'p1_error_for_task: ', p1_error_for_task
                self.sum_p1_errors_for_task += p1_error_for_task
                self.num_p1_errors_for_task += 1

                #print 'mean p1 error: ', self.sum_p1_errors_for_task / self.num_p1_errors_for_task

                inv = self.inverse_list[0]  # TODO select inverse model here
                self.motor_out = inv.lookup_motor_to_goal(goal_states=goal_states,
                                                          states_history=self.states_history)
                #print 'motor_out, no index: ', self.motor_out
                self.motor_out = self.motor_out[1]  # [v, w; v, w; v, w]
        elif (goal_states is not None) and self.always_random_motor_out:
            # todo test predictor here?
            p1_error_for_task = self.predictors_list[1].get_input_context_error(
                input_state=self.states_history.get_state(state_index=0, delay=0),
                context_state=goal_states[0])
            # print 'input state: ', self.states_history.get_state(state_index=0, delay=0)
            # print 'context state: ', goal_states[0]
            # print 'p1_error_for_task: ', p1_error_for_task
            self.sum_p1_errors_for_task += p1_error_for_task
            self.num_p1_errors_for_task += 1
            #print 'mean p1 error: ', self.sum_p1_errors_for_task / self.num_p1_errors_for_task

            self.motor_out = None
        else:
            if self.use_advanced_exploration:
                self.motor_out = self._get_advanced_exploration_motor_out()
            else:
                # this informs robot model to apply random movement
                self.motor_out = None

        self.t += 1

    def _get_advanced_exploration_motor_out(self):
        '''
        use method 2 in:
            Note from 2017-07-12 20:30:07.515 exploration policy
        :return:
        '''
        # num_predictors = 4

        # allocate certain amount of steps to train each predictor
        # steps_per_predictor = self.max_history_length / num_predictors

        # predictor_0: (t, t+2 -> t+1)
        # predictor_1: (t, t+4 -> t+2)

        num_closest_matches = 5

        predictor_0 = self.predictors_list[0]
        predictor_1 = self.predictors_list[1]

        states_history = self.states_history
        motor_history = self.motor_history

        explore_switch_time = self.explore_mode_switch_time #1000000

        if self.t < explore_switch_time:  # steps_per_predictor:
            # use random movements
            motor_out = None
            return motor_out
        else:
            if self.t == explore_switch_time:
                print 'Switching to advanced exploration...'
            # (1) look ahead to S'(t+2) using P[(t,t+2) -> (t+1)]
            #   generate list of K closest S'(t+2) based on S(t)

            #print 'm hist: ', self.motor_history.get_sequence(delay_start=20, delay_end=0)

            possible_context_list, possible_output_list = predictor_0.look_ahead_get_context_output_list(input_state=states_history.get_state(state_index=0, delay=0),
                                                                                                         num_closest_matches=num_closest_matches)
            input_context_error_array = np.zeros(num_closest_matches)
            ind = 0

            # (2) for each S'(t+2) in list
            for possible_context in possible_context_list:
                # set input "S(t)" (same for each) as historical state S(t + 2 - 4)
                temp_input_state = states_history.get_state(state_index=0, delay=2)

                # set context "S(t+4)" as S'(t+2)
                temp_context_state = possible_context

                # get input&context error from P[(t, t+4) -> (t+8)]
                input_context_error = predictor_1.get_input_context_error(input_state=temp_input_state,
                                                                          context_state=temp_context_state)
                input_context_error_array[ind] = input_context_error
                ind += 1

            # (3) choose S'(t+2) with highest input&context error
            max_error_ind = np.argmax(input_context_error_array)
            goal_state_t_p_2 = possible_context_list[max_error_ind]

            # (4) get motor action toward it, by using appropriate trained predictors and finally inverse model:
            # S(t), S'(t+2) -> P -> S'(t+1)
            #goal_state_t_p_1, _, _ = predictor_0.predict(input_state=states_history.get_state(state_index=0, delay=0),
            #                                             context_state=goal_state_t_p_2)
            goal_state_t_p_1 = possible_output_list[max_error_ind]

            # S(t), S'(t+1) -> Inv -> M(t)
            inv = self.inverse_list[0]  # TODO select inverse model here
            motor_out = inv.lookup_motor_to_goal(goal_states=[goal_state_t_p_1],
                                                 states_history=self.states_history)
            motor_out = motor_out[1]  # [v, w; v, w; v, w]

            return motor_out

        # elif self.t < steps_per_predictor * 2:
        # training_predictor = 1
        #    pass
        # elif self.t < steps_per_predictor * 3:
        # training third predictor: (t, t+8 -> t+4)
        # training_predictor = 2
        # else:
        # training fourth predictor: (t, t+16 -> t+8)
        # training_predictor = 3
        # return motor_out

    def get_plan_position_angle_list(self, rays, goal_states):
        '''
        compute plan here

        procedure:
        for each predictor, starting from farthest in time:
            current input, farthest prediction -> get nearer prediction
            get px, py, theta for nearer prediction
            set nearer prediction as (farthest prediction) for next iteration

        :param rays: current visual input
        :param goal_states: [goal visual input, goal autoencoder level 0, level 1, ...]
        :return: list: [[px, py, theta], [px, py, theta], ...]
        '''

        plan_position_angle = False
        if not plan_position_angle:
            return None

        if goal_states is None:
            return None

        states_history = self.states_history
        current_visual_input = self._process_sensors(rays=rays)

        predictor_2_1 = self.predictors_list[0]
        predictor_4_2 = self.predictors_list[1]
        predictor_8_4 = self.predictors_list[2]
        predictor_16_8 = self.predictors_list[3]

        new_goal, dist, ind, input_td_info_8, context_td_info_8, output_td_info_8 = predictor_16_8.predict_and_get_debug_td_info(input_state=states_history.get_state(state_index=0, delay=0),
                                                                                                                                 context_state=goal_states[0])

        return [list(input_td_info_8), list(context_td_info_8), list(output_td_info_8)]

        #new_goal, dist, ind, output_td_info_4 = predictor_16_8.predict_and_get_debug_td_info(input_state=states_history.get_state(state_index=0, delay=0),
        #                                                                                     context_state=new_goal)

        #new_goal, dist, ind, output_td_info_2 = predictor_16_8.predict_and_get_debug_td_info(input_state=states_history.get_state(state_index=0, delay=0),
        #                                                                                     context_state=new_goal)

        #new_goal, dist, ind, output_td_info_1 = predictor_16_8.predict_and_get_debug_td_info(input_state=states_history.get_state(state_index=0, delay=0),
        #                                                                                     context_state=new_goal)

        #return [list(output_td_info_8), list(output_td_info_4), list(output_td_info_2), list(output_td_info_1)]

    def set_always_random_motor(self, setting):
        self.always_random_motor_out = setting

    def reset_for_new_task(self):
        self.last_time_set_8 = -9e4
        self.last_time_set_16 = -9e4
        self.last_time_set_32 = -9e4

    def _get_proximal_goal_states(self, goal_states, states_history, predictors_list):
        '''

        'predictors': [
            {'state_index_input': 0, 'state_index_context': 1, 'state_index_output': 0,
                                     'dt_context': 16,         'dt_output': 8},
            {'state_index_input': 1, 'state_index_context': 2, 'state_index_output': 1,
                                      'dt_context': 32,        'dt_output': 16},
            {'state_index_input': 2, 'state_index_context': 3, 'state_index_output': 2,
                                     'dt_context': 64,         'dt_output': 32}
        ],

        :param goal_states:
        :param states_history:
        :param predictors_list:
        :return:
        '''

        proximal_goal_states = [None] * len(goal_states)
        # TODO: could be completed at end, None states filled in (these are just autoencoder states)

        #   predictor       input,           context,       output
        #       0            t=0, c=0       t=16, c=1      t=8, c=0
        #       1            t=0, c=1       t=32, c=2      t=16, c=1
        #       2            t=0, c=2       t=64, c=3      t=32, c=2

        # order
        # run predictor 2:
        #       input: states_history: (t=0, c=2)
        #       context: goal_states: (t=64, c=3)
        #       output: (t=32, c=2)
        # run predictor 1:
        #       input: states_history: (t=0, c=1)
        #       context: (t=32, c=2)
        #       output: (t=16, c=1)
        # run predictor 0:
        #       input: states_history: (t=0, c=0)
        #       context: (t=16, c=1)
        #       output: (t=8, c=0)

        predictor_2 = predictors_list[2]
        predictor_1 = predictors_list[1]
        predictor_0 = predictors_list[0]

        if self.t > self.last_time_set_32 + 16:
            input_state = states_history.get_state(state_index=2, delay=0)
            context_state = goal_states[3]
            self.state_t_32_c_2, dist_2 = predictor_2.predict(input_state=input_state,
                                                              context_state=context_state)
            dist_2 = dist_2 * 1.0 / (len(input_state) + len(context_state))
            self.dist_2 = dist_2
            self.last_time_set_32 = self.t

        if self.t > self.last_time_set_16 + 8:
            input_state = states_history.get_state(state_index=1, delay=0)
            # context_state_g = goal_states[2]
            context_state = self.state_t_32_c_2
            self.state_t_16_c_1, dist_1 = predictor_1.predict(input_state=input_state,
                                                              context_state=context_state)
            dist_1 = dist_1 * 1.0 / (len(input_state) + len(context_state))

            #state_t_16_c_1, dist_1_g = predictor_1.predict(input_state=input_state,
            #                                               context_state=context_state_g)
            #dist_1_g = dist_1_g * 1.0 / (len(input_state) + len(context_state))
            #state_t_16_c_1, dist_1_h = predictor_1.predict(input_state=states_history.get_state(state_index=1, delay=0),
            #                                               context_state=context_state_h)
            #dist_1 = dist_1 * 1.0 / (len(input_state) + len(context_state))

            self.dist_1 = dist_1
            self.last_time_set_16 = self.t

        if self.t > self.last_time_set_8 + 4:
            input_state = states_history.get_state(state_index=0, delay=0)
            context_state = self.state_t_16_c_1
            self.state_t_8_c_0, dist_0 = predictor_0.predict(input_state=input_state,
                                                             context_state=context_state)
            dist_0 = dist_0 * 1.0 / (len(input_state) + len(context_state))
            self.dist_0 = dist_0
            self.last_time_set_8 = self.t

        proximal_goal_states[0] = self.state_t_8_c_0

        # todo return best dist, so inv can override if better:
        return proximal_goal_states

#        for predictor in predictors_list:
#            input_state = []
#            context_state = []
#            output_state = predictor.predict(input_state, context_state)

        # memory_state_indices = [0, 1, 2, 3]

        # sTODO maybe better init for intermediate memory states (between init and end)
        # mem_state_0 = states_history.get_state(state_index=0, delay=0)
        #                   t=0, c=0       t=8, c=0      t=16, c=1       t=32, c=2       t=64, c=3
        # memory_states = [mem_state_0, goal_states[0], goal_states[1], goal_states[2], goal_states[3]]
        #                      0            1                2              3               4

        # dTODO missing memory states for compress & copy operation?
        # according to notebook:
        # predictor timescales:
        #   predictor       input, context, output
        #       0             0     (0)+16  (0)+8
        #       1             8     (8)+32  (8)+16
        #       2            8+16

        # flattened?         timescale (compression)

        # refer to memory state indices:
        # predictors_input_mem_state =   [0, ]
        # predictors_context_mem_state = [2, ]
        # predictors_output_mem_state =  [1, ]

#        num_iterations = 5
#        for n in range(num_iterations):

            # each predictor updates states corresponding to its output

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

    def _process_predictors(self, predictors_list, states_history, config, models_save_folder, debug_topdown_info_history):
        '''
            training_delay: delay such that testing points (at current time) are independent of recent training
                it is otherwise possible to "cheat" if testing on the point the network was just trained on.
        '''

        training_delay = config['training_delay']

        for predictor in predictors_list:
            if self.predictors_enable_training:
                predictor.train(states_history, training_delay, debug_topdown_info_history)
            elif self.predictors_optimize_training:
                predictor.optimize_train(states_history, training_delay)

            if self.predictors_test_every_k_steps is not None:
                if self.t % self.predictors_test_every_k_steps == 0:
                    # TODO clean this up: make option
                    # predictor.test_on_random_task_and_store_error()
                    predictor.test_newest_point_and_store_error(states_history)

        if self.predictors_save_every_k_steps is not None:
            if self.t % self.predictors_save_every_k_steps == 0 and self.t > 0:
                print 'saving predictors...'
                f = open(models_save_folder + '/predictors.pkl', 'w')
                pickle.dump(predictors_list, f)
                f.close()

    def _process_motor_history(self, newest_motor_command, motor_history):
        motor_history.process_new_motor_command(newest_motor_command)

    def _process_debug_topdown_info_history(self, newest_topdown_info, debug_topdown_info_history):
        debug_topdown_info_history.process_new_topdown_info(newest_topdown_info)

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

    def save_states_history(self, plots_save_folder, state_indices_list):
        print 'saving states history...'
        self.states_history.save_states(plots_save_folder, state_indices_list)




























