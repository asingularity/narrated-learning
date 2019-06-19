import pickle
import time
import cv2
import numpy as np
np.set_printoptions(suppress=True)

from brain_components import StatesHistory, MotorHistory, DebugTopdownInfoHistory


class RobotBrain(object):
    def __init__(self, params):
        self._init_globals(params)

        self.states_history = self._init_states_history(params, states_dim_list=[params['input_dim']])
        self.debug_topdown_info_history = self._init_debug_topdown_info_history(params)
        self.motor_history = self._init_motor_history(params)

        if params['predictor_ensemble_load_from_file']:
            print( 'loading predictor ensemble...')
            f = open(params['predictor_ensemble_filename'], 'r')
            self.predictor_ensemble = pickle.load(f)
            f.close()
            print( 'done loading predictor ensemble.')

            print( 'precomputing...')
            self.predictor_ensemble.precompute()
            print( 'done precomputing.')
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

    def process_input_get_motor(self, rays, last_motor_command, models_save_folder, debug_topdown_info):

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
        return self.motor_out

    def process_new_goal_states(self, goal_states):
        '''
        do any planning if needed
        :return:
        '''
        self.goal_states = goal_states

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

    def save_states_history(self, plots_save_folder, state_indices_list):
        print( 'saving states history...')
        self.states_history.save_states(plots_save_folder, state_indices_list)

    def save_debug_topdown_info_history(self, plots_save_folder):
        print( 'saving debug topdown info history...')
        self.debug_topdown_info_history.save_history(plots_save_folder)




























