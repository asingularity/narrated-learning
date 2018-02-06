import pickle
import time
import cv2
import numpy as np
np.set_printoptions(suppress=True)
from fast_save_matrix import savetxt
from brain_components import Autoencoder, StatesHistory, MotorHistory, Predictor, InverseModel, DebugTopdownInfoHistory
from brain_components_classes.predictor_ensemble import PredictorEnsemble


class RobotBrain(object):
    def __init__(self, params):
        self._init_globals(params)

        self.states_history = self._init_states_history(params, states_dim_list=[params['input_dim']])
        self.debug_topdown_info_history = self._init_debug_topdown_info_history(params)
        self.motor_history = self._init_motor_history(params)

        if params['predictor_ensemble_load_from_file']:
            print 'loading predictor ensemble...'
            f = open(params['predictor_ensemble_filename'], 'r')
            self.predictor_ensemble = pickle.load(f)
            f.close()
            print 'done loading predictor ensemble.'
        else:
            self.predictor_ensemble = None

    def _init_globals(self, params):
        self.t = 0
        self.motor_out = None
        self.goal_states = None
        self.max_history_length = params['max_history_length']  # only needed for experimental exploration
        self.training_delay = params['training_delay']  # new, not used yet

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

    # ************ process ************

    def process_input(self, rays, last_motor_command, models_save_folder, debug_topdown_info):

        current_visual_input = self._process_sensors(rays=rays)
        # previous_motor_command was initiated at T-1, applied [T-1, T],
        # current_visual_input is at time T

        newest_states_list = [current_visual_input.copy()]

        self._process_states_history(newest_states_list=newest_states_list,
                                     states_history=self.states_history)

        self._process_motor_history(newest_motor_command=last_motor_command,
                                    motor_history=self.motor_history)

        self._process_debug_topdown_info_history(newest_topdown_info=debug_topdown_info,
                                                 debug_topdown_info_history=self.debug_topdown_info_history)

        if self.goal_states is not None and self.predictor_ensemble is not None:
            # this may still be None for current (planning-only) testing
            self.motor_out = None  # 0: straight line
        else:
            # this informs robot model to apply random movement
            self.motor_out = None

        self.t += 1

    def process_new_goal_states(self, goal_states, visualizer, rays, topdown_info, current_goal_position_angle):
        self.goal_states = goal_states
        starting_state = self.states_history.get_state(state_index=0, delay=0)

        if starting_state is not None:
            print 'planning...'

            plan_position_angle_list = self.predictor_ensemble.plan_and_get_debug_position_angle_list(goal_state=goal_states,
                                                                                                      starting_state=starting_state)

            print 'finished planning. showing plan for 5 seconds.'

            # change this to public function. pass in plan_position_angle_list as computed here, and display here as planning converges:
            im = visualizer._get_topdown_map(rays, topdown_info, current_goal_position_angle, plan_position_angle_list)
            cv2.imshow('planned', im)
            cv2.waitKey(1)
            time.sleep(5)
            # later, next motor command will be computed here or in process_input based on proximal states in the plan
        else:
            print 'starting state is None. skipping planning.'

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

    def get_motor_output(self):
        return self.motor_out

    def _process_sensors(self, rays):
        ray_radians = rays['ray_radians']
        ray_colors = rays['ray_colors']
        ray_lengths = rays['ray_lengths']

        current_visual_input = ray_colors.copy()

        return current_visual_input

    def _process_states_history(self, newest_states_list, states_history):
        states_history.process_new_states(newest_states_list)

    def _process_motor_history(self, newest_motor_command, motor_history):
        motor_history.process_new_motor_command(newest_motor_command)

    def _process_debug_topdown_info_history(self, newest_topdown_info, debug_topdown_info_history):
        debug_topdown_info_history.process_new_topdown_info(newest_topdown_info)

    # ************ functions for other interfaces to retrieve information ************
    # TODO undefined now
    def get_error_names_histories(self):

        error_names_autoenc = []
        error_histories_autoenc = []
        for net_index in range(len(self.autoencoders_list)):
            error_names_autoenc.append('autoencoder_' + str(net_index))
            error_histories_autoenc.append(self.autoencoders_list[net_index].get_mean_error_history())

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

        return error_names_autoenc, error_histories_autoenc, \
               error_names_predictor, error_histories_predictor, \
               error_names_inverse, error_histories_inverse, \
               error_names_no_context_predictor, error_histories_no_context_predictor

    # TODO undefined now
    def get_autoenc_images(self):
        return self.autoencoder_images

    # TODO undefined now
    def get_predictor_images(self):
        return self.ctx_predictor_debug_images

    def save_states_history(self, plots_save_folder, state_indices_list):
        print 'saving states history...'
        self.states_history.save_states(plots_save_folder, state_indices_list)

    def save_debug_topdown_info_history(self, plots_save_folder):
        print 'saving debug topdown info history...'
        self.debug_topdown_info_history.save_history(plots_save_folder)




























