import pickle
import time
import cv2
import numpy as np
np.set_printoptions(suppress=True)

from brain_components import StatesHistory, MotorHistory, DebugTopdownInfoHistory, MultiLayerEnsemble


class RobotBrain(object):
    def __init__(self, params):
        self._init_globals(params)

        self.states_history = self._init_states_history(params, states_dim_list=[params['input_dim']])
        self.debug_topdown_info_history = self._init_debug_topdown_info_history(params)
        self.motor_history = self._init_motor_history(params)

        self.ensemble_save_every_k_secs = params['predictor_ensemble_save_every_k_secs']
        self.last_ensemble_save_time = time.time()

        if params['predictor_ensemble_load_from_file']:
            print( 'loading predictor ensemble...')
            f = open(params['predictor_ensemble_filename'], 'rb')
            self.predictor_ensemble = pickle.load(f)
            self.predictor_ensemble.init_after_load()
            f.close()
            print( 'done loading predictor ensemble.')
        else:

            dim = params['input_dim']


            IO_entries_per_layer = params['IO_entries_per_layer']
            n_layers = len(IO_entries_per_layer)

            C_entries_factor = params['C_entries_factor']
            IO_learn_time_factor = params['IO_learn_time_factor']
            C_learn_time_factor = params['C_learn_time_factor']

            C_entries_per_layer = []
            layer_IO_learn_times = []
            layer_C_learn_times = []

            for k in range(n_layers):
                IO_entries = IO_entries_per_layer[k]
                layer_IO_learn_times.append(IO_entries * IO_learn_time_factor)

                if k < n_layers - 1:
                    C_entries = IO_entries * C_entries_factor
                    C_entries_per_layer.append(C_entries)
                    layer_C_learn_times.append(C_entries * C_learn_time_factor)
                else:
                    C_entries_per_layer.append(0)
                    layer_C_learn_times.append(0)

            self.predictor_ensemble = MultiLayerEnsemble(params={
                'input_dim': dim,
                'IO_entries_per_layer': IO_entries_per_layer,
                'C_entries_per_layer': C_entries_per_layer,
                'predict_time_per_layer': [params['predict_time']] * n_layers,
                'layer_IO_learn_times': layer_IO_learn_times,
                'layer_C_learn_times': layer_C_learn_times,
            })

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

        self.predictor_ensemble.step(input_state=newest_states_list[0],
                                     input_x_y_theta=self.debug_topdown_info_history.get_td_info(delay=0))

        if self.goal_states is not None and self.predictor_ensemble is not None:
            # TODO this is where "task mode" is enabled
            self.motor_out = None
        else:
            # this informs robot model to apply random movement
            self.motor_out = None

        # save if needed
        if self.ensemble_save_every_k_secs is not None:
            if time.time() - self.last_ensemble_save_time > self.ensemble_save_every_k_secs:
                print('saving ensemble...')
                self.predictor_ensemble.save_to_pkl(file_path=models_save_folder,
                                                    file_name='ensemble.pkl')
                print('done saving ensemble.')
                self.last_ensemble_save_time = time.time()

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

    def get_table_ims(self):
        IO_im_list, C_im_list, W_im_list = self.predictor_ensemble.get_table_ims()
        return IO_im_list, C_im_list, W_im_list



























