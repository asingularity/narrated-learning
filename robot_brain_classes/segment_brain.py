

import numpy as np
from brain_components import WinRegionTiles, SimpleWeightTiles, SimplePredictionTiles


class SegmentBrain(object):
    def __init__(self, params):
        '''

        uses MultiLayerSharedTiles
        like: original run_demo.py, but with: USE_RAYCAST_IMAGE = True
        which assumed a 32x32 input image, and only one layer

        :param params:
        '''

        dim = params['input_dim']

        self.predictor_ensemble = SimplePredictionTiles(params={
            'color_enabled': params['color_enabled'],
            'enable_weight_bias': params['enable_weight_bias'],
            'input_dim': dim,
            'goal_context_dim': None,
            'enable_learning': params['enable_learning'],
            'tile_entries_per_layer': params['tile_entries_per_layer'],
            'table_learn_time_per_layer': params['table_learn_time_per_layer'],
            'prediction_learn_time_per_layer': params['prediction_learn_time_per_layer'],
            'tiles_per_layer_NxN': params['tiles_per_layer_NxN'],
            'pre_init_goal_contexts': None,  # this matches _get_context_for_goal_state
            'max_history_length': params['max_history_length'],  # so it can check that learn time ranges are within!
            'table_ims_scale_pixels': params['table_ims_scale_pixels']
        })

    # @profile
    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        # print(np.amin(input_im), np.amax(input_im), input_im.dtype)

        motor_out = self.predictor_ensemble.step(raycast_image=input_im,
                                                 input_state=None,
                                                 input_x_y_theta=None,
                                                 goal_context_state_learning=None,
                                                 goal_context_state_task=None,
                                                 last_motor_command=None)

    def save_model(self, models_save_folder):
        '''

        :param models_save_folder:
        :return:
        '''

    def get_table_ims(self):
        ims_lists = self.predictor_ensemble.get_table_ims()
        return ims_lists
