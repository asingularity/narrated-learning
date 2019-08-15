import pickle
import time
import cv2
import numpy as np
np.set_printoptions(suppress=True)

from brain_components import StatesHistory, MotorHistory, DebugTopdownInfoHistory, MultiLayerEnsemble, SimpleMultiLayer


class RobotBrain(object):
    def __init__(self, params):
        self._init_globals(params)

        self.states_history = self._init_states_history(params, states_dim_list=[params['input_dim']])
        self.debug_topdown_info_history = self._init_debug_topdown_info_history(params)
        self.motor_history = self._init_motor_history(params)

        self.ensemble_save_every_k_secs = params['predictor_ensemble_save_every_k_secs']
        self.last_ensemble_save_time = time.time()

        self.max_num_goal_states = params['max_num_goal_states']

        if params['predictor_ensemble_load_from_file']:
            print( 'loading predictor ensemble...')
            f = open(params['predictor_ensemble_filename'], 'rb')
            self.predictor_ensemble = pickle.load(f)
            self.predictor_ensemble.init_after_load()
            f.close()
            print( 'done loading predictor ensemble.')

            if not params['enable_learning']:
                self.predictor_ensemble.disable_learning()

        else:

            dim = params['input_dim']
            goal_states_dim = 1
            table_entries = params['table_entries']

            self.predictor_ensemble = SimpleMultiLayer(params={
                'input_dim': dim,
                'goal_context_dim': goal_states_dim,
                'entries': table_entries,
                'predict_time_per_layer': params['predict_time_per_layer'],
                'enable_learning': params['enable_learning'],
                'table_learn_time': params['table_learn_time'],
                'prediction_learn_time': params['prediction_learn_time'],
                'pre_init_goal_contexts': self._get_goal_contexts_list(),  # this matches _get_context_for_goal_state
                'max_history_length': params['max_history_length']  # so it can check that learn time ranges are within!
            })

            # __________________________ HERE NOW __________________________

    def _init_globals(self, params):
        self.t = 0
        self.motor_out = None
        self.goal_states = None
        self.max_history_length = params['max_history_length']  # only needed for experimental exploration

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

    def _get_goal_contexts_list(self):
        '''
        return list of all goal contexts, matching _get_context_for_goal_state
        :return:
        '''
        return list(range(self.max_num_goal_states + 1))

    def _get_context_for_goal_state(self, goal_index):
        '''

        uses self.num_goal_states

        :param goal_index:
        :return:
        '''
        if goal_index is None:
            return None
        else:
            goal_context = np.zeros(1, np.float32)
            goal_context[0] = goal_index

            return goal_context

    # ************ process ************

    def process_input_get_motor(self, rays, last_motor_command, models_save_folder, debug_topdown_info, goal_index_reached, goal_index_task):
        '''

        :param rays:
        :param last_motor_command:
        :param models_save_folder:
        :param debug_topdown_info:
        :param goal_index:
        :return: motor_out: (linear_velocity, angular_velocity)

        '''

        # TODO properly use goal_index_reached for learning only, goal_index_task for task mode only
        # TODO if goal_index_task is None: assume not in task mode i.e. do default behavior for learning, what it is now, including table look-ups.

        goal_context_state_learning = self._get_context_for_goal_state(goal_index=goal_index_reached)
        assert goal_context_state_learning is not None

        goal_context_state_task = self._get_context_for_goal_state(goal_index=goal_index_task)
        # context state for task might be None if not in task mode

        if goal_context_state_learning[0] > 0:
            debug_print = False
            if debug_print:
                print('goal_context_state:', goal_context_state_learning)

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

        motor_out = self.predictor_ensemble.step(input_state=newest_states_list[0].astype(np.float32),
                                                 input_x_y_theta=self.debug_topdown_info_history.get_td_info(delay=0),
                                                 goal_context_state_learning=goal_context_state_learning,
                                                 goal_context_state_task=goal_context_state_task,
                                                 last_motor_command=last_motor_command)

        if goal_context_state_task is not None and self.predictor_ensemble is not None:
            # TODO this is where "task mode" is enabled

            # (linear_velocity, angular_velocity)
            # TODO don't hard-code to assume two motor steps! Average over all instead?
            self.motor_out = (motor_out[0] + motor_out[2]) * 0.5, (motor_out[1] + motor_out[3]) * 0.5
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
        ims_lists = self.predictor_ensemble.get_table_ims()
        return ims_lists

    def save_model(self, models_save_folder):
        self.predictor_ensemble.save_to_pkl(file_path=models_save_folder,
                                            file_name='ensemble.pkl')



























