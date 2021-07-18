import numpy as np
from brain_components_classes.states_history import StatesLimitedHistory


#from robot_brain_classes.tiled_rate_control_wta import RateControlWTABRainLayer0, RateControlWTABRainLayerN
from robot_brain_classes.tiled_iter_wta import IterWTABRainLayerN, IterWTABRainLayer0

from cython_layers_viz import cy_weigh_in_with_out, cy_weigh_in_with_out_L0


# TODO how does tiling work for next layer, given sparse output of layer 0?
#   specifically, many input events now have same position r,c
#   !! cy_tile_the_input function will not work as written !!
#   best way is probably as a new class


# TODO pre-processor? set time steps? only first layer has pre processor?
#   or instead does each layer do time integration again (limited history) ? probably that

# TODO should not do input state dim doubling / p/n stuff for upper layers ...


# 'input_im_dim': input_dim, # TODO problem: assumes image dim i.e. NxN
# 'include_n_events': include_n_events,  # TODO add param



class LayeredTiledRateControlWTA(object):
    def __init__(self, params):

        # maybe: for now we will hardcode some layers, assuming 256x256 image.
        # spatial and temporal layers? means replicate same tiling in between these but add integration time
        # tile_dim_NxN: how many of previous layer tiles combine for a tile in this layer

        # layer     tile_dim_NxN        tile dim pixels     num tiles
        # 0         -                   8x8                 32x32
        # 1         2x2                 16x16               16x16
        # 2         2x2                 32x32               8x8
        # 3         2x2                 64x64               4x4
        # 4         2x2                 128x128             2x2
        # 5         2x2                 256x256             1x1

        # hardcode does:

        # layer     tile_dim_NxN        tile dim pixels     num tiles
        # 0         -                   8x8                 32x32
        # 1         4x4                 32x32               8x8
        # 2         4x4                 128x128             2x2
        # 3         2x2                 256x256             1x1

        self._init_layers_hardcode(params=params)

        self.do_raster_plots_every_k_im = 4
        self.ims_since_raster = 0

    def _init_layers_hardcode(self, params):
        '''

        layer     tile_dim_NxN        tile dim pixels     num tiles
        0         -                   8x8                 32x32
        1         4x4                 32x32               8x8
        2         4x4                 128x128             2x2
        3         2x2                 256x256             1x1

        :return:
        '''

        self.input_im_dim = params['input_im_dim']
        layer_params_list = []

        # 0
        layer_params_list.append({
            'input_im_dim': params['input_im_dim'],
            'tile_im_dim': 8,
            'num_rfs': 400,  # per tile
            'lr': 1.0 / 1000,
            'input_concat_timesteps': 1
        })

        # 1
        # TODO we need to manually update input params here based on what we set above !!! specifically "input_num_tiles_NxN"
        layer_params_list.append({
            'input_num_tiles_NxN': 2,  # previous layer (num_tiles X num_tiles)
            'input_num_rfs_per_tile': layer_params_list[-1]['num_rfs'],  # previous layer num rfs
            'tile_dim_NxN': 1,  # relative to previous layer, how many tiles (NxN) to combine to make a tile in this layer
            'num_rfs': 80,  # per tile
            'lr': 1.0 / 1000,
            'input_concat_timesteps': 4
        })

        # 2
        layer_params_list.append({
            'input_num_tiles_NxN': 2,  # previous layer (num_tiles X num_tiles)
            'input_num_rfs_per_tile': layer_params_list[-1]['num_rfs'],  # previous layer num rfs
            'tile_dim_NxN': 2,
            'num_rfs': 400,  # per tile
            'lr': 1.0 / 1000,
            'input_concat_timesteps': 1
        })

        # 3
        layer_params_list.append({
            'input_num_tiles_NxN': 1,  # previous layer (num_tiles X num_tiles)
            'input_num_rfs_per_tile': layer_params_list[-1]['num_rfs'],  # previous layer num rfs
            'tile_dim_NxN': 1,
            'num_rfs': 80,  # per tile
            'lr': 1.0 / 1000,
            'input_concat_timesteps': 4
        })
        #
        # # 4
        # layer_params_list.append({
        #     'input_num_tiles_NxN': 8,  # previous layer (num_tiles X num_tiles)
        #     'input_num_rfs_per_tile': layer_params_list[-1]['num_rfs'],  # previous layer num rfs
        #     'tile_dim_NxN': 8,
        #     'num_rfs': 800,  # per tile
        #     'lr': 1.0 / 1000,
        #     'input_concat_timesteps': 1
        # })
        #
        # # 5
        # layer_params_list.append({
        #     'input_num_tiles_NxN': 1,  # previous layer (num_tiles X num_tiles)
        #     'input_num_rfs_per_tile': layer_params_list[-1]['num_rfs'],  # previous layer num rfs
        #     'tile_dim_NxN': 1,
        #     'num_rfs': 800,  # per tile
        #     'lr': 1.0 / 1000,
        #     'input_concat_timesteps': 4
        # })

        self.num_layers = len(layer_params_list)

        self.layers = []

        print()
        print('input_im_flat_history:: states_dim: ', self.input_im_dim * self.input_im_dim)
        print()
        self.input_im_flat_history = StatesLimitedHistory(params={'max_delay': 1,
                                                                  'states_dim_list': [self.input_im_dim * self.input_im_dim],
                                                                  'store_extra_data': False})

        self.output_event_histories = []

        layer_num = 0
        for layer_params in layer_params_list:
            # common params
            layer_params['rel_lr_bg'] = 0.0
            layer_params['max_time'] = 5000000
            layer_params['do_raster_plots_every_k_im'] = None
            layer_params['network_type'] = 'seq-kmeans'

            if layer_num == 0:
                self.layers.append(IterWTABRainLayer0(params=layer_params))
            else:
                self.layers.append(IterWTABRainLayerN(params=layer_params))

            state_dim_tmp = self.layers[layer_num].get_num_tiles_NxN() * self.layers[layer_num].get_num_tiles_NxN() * self.layers[layer_num].get_num_rfs_per_tile()

            self.output_event_histories.append(StatesLimitedHistory(params={'max_delay': 2,
                                                                            'states_dim_list': [state_dim_tmp],
                                                                            'store_extra_data': False}))

            layer_num += 1

        # print layer info and do asserts

        # layer     tile_dim_NxN        tile dim pixels     num tiles
        # 0         (8x8)               8x8                 32x32
        # 1         4x4                 32x32               8x8
        # 2         4x4                 128x128             2x2
        # 3         2x2                 256x256             1x1

        assert len(self.layers) == self.num_layers, str((len(self.layers), self.num_layers))

        print()
        print('**** _init_layers_hardcode ****')

        for layer_num in range(self.num_layers):
            wta_layer = self.layers[layer_num]
            print()
            print('layer:', layer_num)
            print('    input_num_tiles_NXN:', wta_layer.get_input_num_tiles_NxN())
            print('    input_num_rfs_per_tile:', wta_layer.get_input_num_rfs_per_tile())
            print()
            print('    tile_dim_NxN:', wta_layer.get_tile_dim_NxN())
            print('    num_tiles_NxN:', wta_layer.get_num_tiles_NxN())
            print('    num_rfs_per_tile:', wta_layer.get_num_rfs_per_tile())

        print()

    def get_final_errors_dict(self):
        d = {}
        return d

    def set_plots_folder(self, folder):
        self.plots_folder = folder

        print()
        print('setting plots folder: ', self.plots_folder)
        print()

        for layer_n in range(self.num_layers):
            self.layers[layer_n].set_plots_folder(folder=folder)

    def process_input(self, input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image):

        input_events = None
        output_events = None

        self.last_input_events = (input_events_p.copy(), input_events_n.copy(), event_coords_r.copy(), event_coords_c.copy(), original_input_image.copy())
        self.last_output_events = []

        self.input_im_flat_history.process_new_states([original_input_image.flatten()])

        for layer_n in range(self.num_layers):

            if layer_n == 0:
                output_events = self.layers[layer_n].process_input(input_events_p=input_events_p,
                                                                   input_events_n=input_events_n,
                                                                   event_coords_r=event_coords_r,
                                                                   event_coords_c=event_coords_c,
                                                                   original_input_image=original_input_image)
            else:
                output_events = self.layers[layer_n].process_input(input_events)

            # print(layer_n, output_events.shape)

            input_events = output_events.copy()
            self.last_output_events.append(output_events.copy())

            self.output_event_histories[layer_n].process_new_states([output_events.flatten()])

    def do_plots(self):
        # TODO skip layer 0 plots for now
        for layer_num in range(1, self.num_layers):
            self.layers[layer_num].do_plots(extra_info=str(layer_num))

    def get_table_ims(self):

        # FIVE LAYER

        # NOT SHOWING FULL TEMPORAL RF JUST 1-2 STEPS INTO PAST!!!

        # only last step for now: not weighing/displaying anything further back

        if self.do_raster_plots_every_k_im is not None:
            self.ims_since_raster += 1
            if self.ims_since_raster > self.do_raster_plots_every_k_im:
                self.do_plots()
                self.ims_since_raster = 0


        ims_list = []
        ims_names_list = []

        # get ims for layer 0

        ims_list_0, ims_names_list_0 = self.layers[0].get_table_ims()
        ims_list.extend(ims_list_0)
        ims_names_list.extend(ims_names_list_0)

        # layer 0

        num_tiles_NxN_L0 = self.layers[0].get_num_tiles_NxN()
        num_rfs_per_tile_L0 = self.layers[0].get_num_rfs_per_tile()
        tile_dim_NxN_L0 = self.layers[0].get_tile_dim_NxN()
        output_events_L0 = self.output_event_histories[0].get_state(delay=0).copy().reshape((num_tiles_NxN_L0, num_tiles_NxN_L0, num_rfs_per_tile_L0))
        weights_L0 = self.layers[0].weights

        num_tiles_NxN_L1 = self.layers[1].get_num_tiles_NxN()
        num_rfs_per_tile_L1 = self.layers[1].get_num_rfs_per_tile()
        tile_dim_NxN_L1 = self.layers[1].get_tile_dim_NxN()
        output_events_L1 = self.output_event_histories[1].get_state(delay=0).copy().reshape((num_tiles_NxN_L1, num_tiles_NxN_L1, num_rfs_per_tile_L1))
        weights_L1 = self.layers[1].weights

        # num_tiles_NxN_L2 = self.layers[2].get_num_tiles_NxN()
        # num_rfs_per_tile_L2 = self.layers[2].get_num_rfs_per_tile()
        # tile_dim_NxN_L2 = self.layers[2].get_tile_dim_NxN()
        # output_events_L2 = self.output_event_histories[2].get_state(delay=0).copy().reshape((num_tiles_NxN_L2, num_tiles_NxN_L2, num_rfs_per_tile_L2))
        # weights_L2 = self.layers[2].weights

        # num_tiles_NxN_L3 = self.layers[3].get_num_tiles_NxN()
        # num_rfs_per_tile_L3 = self.layers[3].get_num_rfs_per_tile()
        # tile_dim_NxN_L3 = self.layers[3].get_tile_dim_NxN()
        # output_events_L3 = self.output_event_histories[3].get_state(delay=0).copy().reshape((num_tiles_NxN_L3, num_tiles_NxN_L3, num_rfs_per_tile_L3))
        # weights_L3 = self.layers[3].weights
        #
        # num_tiles_NxN_L4 = self.layers[4].get_num_tiles_NxN()
        # num_rfs_per_tile_L4 = self.layers[4].get_num_rfs_per_tile()
        # tile_dim_NxN_L4 = self.layers[4].get_tile_dim_NxN()
        # output_events_L4 = self.output_event_histories[4].get_state(delay=0).copy().reshape((num_tiles_NxN_L4, num_tiles_NxN_L4, num_rfs_per_tile_L4))
        # weights_L4 = self.layers[4].weights
        #
        # num_tiles_NxN_L5 = self.layers[5].get_num_tiles_NxN()
        # num_rfs_per_tile_L5 = self.layers[5].get_num_rfs_per_tile()
        # tile_dim_NxN_L5 = self.layers[5].get_tile_dim_NxN()
        # output_events_L5 = self.output_event_histories[5].get_state(delay=0).copy().reshape((num_tiles_NxN_L5, num_tiles_NxN_L5, num_rfs_per_tile_L5))
        # weights_L5 = self.layers[5].weights

        # 5 -> 4

        # cy_weigh_in_with_out(output_events_L4,  # input events to weigh (multiply with above layer's winning RF weights)
        #                      num_tiles_NxN_L4,
        #                      num_rfs_per_tile_L4,
        #                      output_events_L5,  # get winning RF index of output layer, per tile: this is in the space of input layer events
        #                      weights_L5,  # winning (and all other) RF weights
        #                      num_tiles_NxN_L5,
        #                      num_rfs_per_tile_L5,
        #                      tile_dim_NxN_L5)
        #
        # cy_weigh_in_with_out(output_events_L3,  # input events to weigh (multiply with above layer's winning RF weights)
        #                      num_tiles_NxN_L3,
        #                      num_rfs_per_tile_L3,
        #                      output_events_L4,  # get winning RF index of output layer, per tile: this is in the space of input layer events
        #                      weights_L4,  # winning (and all other) RF weights
        #                      num_tiles_NxN_L4,
        #                      num_rfs_per_tile_L4,
        #                      tile_dim_NxN_L4)

        # cy_weigh_in_with_out(output_events_L2,  # input events to weigh (multiply with above layer's winning RF weights)
        #                      num_tiles_NxN_L2,
        #                      num_rfs_per_tile_L2,
        #                      output_events_L3,  # get winning RF index of output layer, per tile: this is in the space of input layer events
        #                      weights_L3,  # winning (and all other) RF weights
        #                      num_tiles_NxN_L3,
        #                      num_rfs_per_tile_L3,
        #                      tile_dim_NxN_L3)

        # print(np.sum(output_events_L2), np.nonzero(output_events_L2)[0])
        #
        # cy_weigh_in_with_out(output_events_L1,  # input events to weigh (multiply with above layer's winning RF weights)
        #                      num_tiles_NxN_L1,
        #                      num_rfs_per_tile_L1,
        #                      output_events_L2,  # get winning RF index of output layer, per tile: this is in the space of input layer events
        #                      weights_L2,  # winning (and all other) RF weights
        #                      num_tiles_NxN_L2,
        #                      num_rfs_per_tile_L2,
        #                      tile_dim_NxN_L2)

        cy_weigh_in_with_out(output_events_L0,  # input events to weigh (multiply with above layer's winning RF weights)
                             num_tiles_NxN_L0,
                             num_rfs_per_tile_L0,
                             output_events_L1,  # get winning RF index of output layer, per tile: this is in the space of input layer events
                             weights_L1,  # winning (and all other) RF weights
                             num_tiles_NxN_L1,
                             num_rfs_per_tile_L1,
                             tile_dim_NxN_L1)

        # TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!

        input_im = self.input_im_flat_history.get_state(delay=0).copy().reshape((self.input_im_dim, self.input_im_dim))

        # modifies input_im_d1
        cy_weigh_in_with_out_L0(input_im,
                                self.input_im_dim,
                                1,
                                output_events_L0,
                                weights_L0,
                                num_tiles_NxN_L0,
                                num_rfs_per_tile_L0,
                                tile_dim_NxN_L0)

        # reshape and append input im (delayed, one after another)

        input_im_d0 = input_im.reshape((self.input_im_dim, self.input_im_dim))
        #input_im_d1 = input_im_d1.reshape((self.input_im_dim, self.input_im_dim))
        #input_im_d0_d1 = np.hstack((input_im_d0, 0.5 + np.zeros((self.input_im_dim, 2)), input_im_d1))
        #ims_list.append(input_im_d0_d1)
        ims_list.append(input_im_d0)
        ims_names_list.append('weighed_input')



        return ims_list, ims_names_list

    def get_table_ims_TWO_LAYER(self):

        if self.do_raster_plots_every_k_im is not None:
            self.ims_since_raster += 1
            if self.ims_since_raster > self.do_raster_plots_every_k_im:
                self.do_plots()
                self.ims_since_raster = 0


        ims_list = []
        ims_names_list = []

        # get ims for layer 0

        ims_list_0, ims_names_list_0 = self.layers[0].get_table_ims()
        ims_list.extend(ims_list_0)
        ims_names_list.extend(ims_names_list_0)

        # populate input events for these time steps from history: as input image, black/white
        # get weighted input? already computed as dot. or just get weights given winners of that layer?
        # latter: need to weigh multiple steps as they occurred, whereas at next layer already concatenated


        num_tiles_NxN_L0 = self.layers[0].get_num_tiles_NxN()
        num_rfs_per_tile_L0 = self.layers[0].get_num_rfs_per_tile()
        tile_dim_NxN_L0 = self.layers[0].get_tile_dim_NxN()

        num_tiles_NxN_L1 = self.layers[1].get_num_tiles_NxN()
        num_rfs_per_tile_L1 = self.layers[1].get_num_rfs_per_tile()
        tile_dim_NxN_L1 = self.layers[1].get_tile_dim_NxN()

        weights_1 = self.layers[1].weights
        output_events_1 = self.output_event_histories[1].get_state(delay=0).copy().reshape((num_tiles_NxN_L1, num_tiles_NxN_L1, num_rfs_per_tile_L1))

        # print('weights_1.shape', weights_1.shape)
        # print('weights_0.shape', weights_0.shape)
        # weights_1.shape(800, 3200)
        # weights_0.shape(800, 128)

        weights_0 = self.layers[0].weights
        output_events_0_d0 = self.output_event_histories[0].get_state(delay=0).copy().reshape((num_tiles_NxN_L0, num_tiles_NxN_L0, num_rfs_per_tile_L0))
        output_events_0_d1 = self.output_event_histories[0].get_state(delay=1).copy().reshape((num_tiles_NxN_L0, num_tiles_NxN_L0, num_rfs_per_tile_L0))

        # modifies output_events_0_d0
        cy_weigh_in_with_out(output_events_0_d0,  # input events to weigh (multiply with above layer's winning RF weights)
                             num_tiles_NxN_L0,
                             num_rfs_per_tile_L0,
                             output_events_1,  # get winning RF index of output layer, per tile: this is in the space of input layer events
                             weights_1,  # winning (and all other) RF weights
                             num_tiles_NxN_L1,
                             num_rfs_per_tile_L1,
                             tile_dim_NxN_L1)

        # modifies output_events_0_d1
        cy_weigh_in_with_out(output_events_0_d1,
                             num_tiles_NxN_L0,
                             num_rfs_per_tile_L0,
                             output_events_1,  # get winning RF index of output layer, per tile: this is in the space of input layer events
                             weights_1,  # winning (and all other) RF weights
                             num_tiles_NxN_L1,
                             num_rfs_per_tile_L1,
                             tile_dim_NxN_L1)

        input_im_d0 = self.input_im_flat_history.get_state(delay=0).copy().reshape((self.input_im_dim, self.input_im_dim))
        input_im_d1 = self.input_im_flat_history.get_state(delay=1).copy().reshape((self.input_im_dim, self.input_im_dim))

        # modifies input_im_d0
        # could this be same cython function as above?
        # yes but need to add newaxis to input_im_d0, input_im_d1, then take it away before append and show (num_rfs_per_tile == 1)

        # TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
        # TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
        # TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!
        # TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!# TODO NEED TO FIX!!!!!!!!!!!!! THIS IS NOT ACCOUNTING N WEIGHTS JUST P WEIGHTS !!!!!!!!!!!!!!!!!!

        cy_weigh_in_with_out_L0(input_im_d0,
                                self.input_im_dim,
                                1,
                                output_events_0_d0,
                                weights_0,
                                num_tiles_NxN_L0,
                                num_rfs_per_tile_L0,
                                tile_dim_NxN_L0)

        # modifies input_im_d1
        cy_weigh_in_with_out_L0(input_im_d1,
                                self.input_im_dim,
                                1,
                                output_events_0_d1,
                                weights_0,
                                num_tiles_NxN_L0,
                                num_rfs_per_tile_L0,
                                tile_dim_NxN_L0)

        # reshape and append input im (delayed, one after another)

        input_im_d0 = input_im_d0.reshape((self.input_im_dim, self.input_im_dim))
        input_im_d1 = input_im_d1.reshape((self.input_im_dim, self.input_im_dim))
        input_im_d0_d1 = np.hstack((input_im_d0, 0.5 + np.zeros((self.input_im_dim, 2)), input_im_d1))
        ims_list.append(input_im_d0_d1)
        ims_names_list.append('weighed_input')




        return ims_list, ims_names_list


    def ASOFUASF_get_table_ims(self):

        ims_list = []
        ims_names_list = []

        # get ims for layer 0

        ims_list_0, ims_names_list_0 = self.layers[0].get_table_ims()
        ims_list.extend(ims_list_0)
        ims_names_list.extend(ims_names_list_0)

        # for upper layers: project back (write in this class)
        #   p/n display stuff does not work for upper layers

        # project how?
        # make an image for each layer:
        #   in "slow visualization" mode: for each upper layer tile, show a real-time mask of the original video input. i.e. pixels over time weighted by weights of RF given there was an RF event.
        #   it must necessarily be delayed by total integration time number of time steps

        # or simplify, for asynchronous viewing:
        # show last N frames for N concat for a given layer, per RF. last activation sequence in terms of input. so, not per input tile but per RF

        # show highest layer, in terms of pixels

        # this is not enough of this layer, because of concat events:
        (input_events_p, input_events_n, event_coords_r, event_coords_c, original_input_image) = self.last_input_events
        # this is not enough of this layer, because of concat events:
        output_events_0 = self.last_output_events[0]

        output_events_1 = self.last_output_events[1]

        print()
        print('input_events_p.shape', input_events_p.shape, 'event_coords_r.shape', event_coords_r.shape, 'original_input_image.shape', original_input_image.shape)
        print('output_events_0.shape', output_events_0.shape)
        print('output_events_1.shape', output_events_1.shape)
        print()

        '''
        output_events[tile_r, tile_c, best_rf_per_tile] = 1
        256 * 256 = 65536
        
        input_events_p.shape (65536,) event_coords_r.shape (65536,) original_input_image.shape (256, 256)
        output_events_0.shape (32, 32, 800)
        output_events_1.shape (16, 16, 800)
        
        '''

        # thing we are getting out:
        # for every unit active in highest layer (1 per tile), color the input (i.e. show weight as a color intensity vs. gray or another color)
        # meaning: for the input sequence that came before current activations of highest layer, color the input. must show enough frames for all layer's concat

        weighted_inputs_L1 = self.layers[1].get_last_weighted_inputs()  # last concated input to each tile, weighted by W for winning RF in that tile (i.e. dot(weight, last concat input))

        assert weighted_inputs_L1.shape[0] == self.layers[1].get_num_tiles_NxN()
        assert weighted_inputs_L1.shape[1] == self.layers[1].get_num_tiles_NxN()
        assert weighted_inputs_L1.shape[2] == self.layers[1].get_input_num_rfs_per_tile() * self.layers[1].get_tile_dim_NxN() * self.layers[1].get_tile_dim_NxN()

        weighted_inputs_L0 = self.layers[0].get_last_weighted_inputs()

        # these are about pixel events, for L0; i.e. get_input_num_rfs_per_tile() is 1
        assert weighted_inputs_L0.shape[0] == self.layers[0].get_num_tiles_NxN()
        assert weighted_inputs_L0.shape[1] == self.layers[0].get_num_tiles_NxN()
        assert weighted_inputs_L0.shape[2] == self.layers[0].get_input_num_rfs_per_tile() * self.layers[0].get_tile_dim_NxN() * self.layers[0].get_tile_dim_NxN()  # == number of input pixels per tile

        # do weighing of weighted_inputs_L0 by weighted_inputs_L1
        # i.e. for every L0 tile, we already have a weight on every pixel
        # now we just need to multiply that by the appropriate weight from the layer above it (using information about tiling)
        # AND we need info about what RFs weere active in L0
        # TODO what about accouting for concat steps? time / multiple steps? apply same weights over over multiple

        # so combine using:
        weighted_inputs_L1
        weighted_inputs_L0
        output_events_0

        # easiest is to write this as loops in cython (complicated)
        # first write as loop here



        '''
        
        alternative was to try to display all the last activations for every RF in every tile
                
        this is too much info to visualize:
        
        last_i_per_rf = self.layers[1].get_last_input_per_rf_activation()

        assert last_i_per_rf.shape[0] == self.layers[1].get_num_tiles_NxN()
        assert last_i_per_rf.shape[1] == self.layers[1].get_num_tiles_NxN()
        assert last_i_per_rf.shape[2] == self.layers[1].get_num_rfs_per_tile()
        assert last_i_per_rf.shape[3] == self.layers[1].input dim per tile ????
         
        '''


        return ims_list, ims_names_list


    def _ANARBAGH_init_layers_with_loop(self, params):
        '''
        to implement later
        :return:
        '''

        self.num_layers = params['num_layers']
        self.layers = []

        for layer_n in range(self.num_layers):

            if layer_n == 0:

                layer_params = {
                    'input_im_dim': params['input_im_dim'],
                    'tile_im_dim': 8,
                    'num_rfs': 800,  # per tile
                    'lr': 1.0 / 1000,
                    'rel_lr_bg': 0.0,
                    'max_time': 5000000,
                    'do_raster_plots_every_k_im': None,  # or None
                    'network_type': 'seq-kmeans',  # seq-kmeans, seq-knn, nn-inits-kmeans
                }

                layer_wta = RateControlWTABRainLayer0(params=layer_params)

            else:

                layer_params = {
                    'input_num_tiles_NxN': self.layers[-1].get_num_tiles_NxN(),
                    'input_flat_dim_per_tile': self.layers[-1].get_num_rfs_per_tile(),
                    'tile_dim_NxN': 1,  # relative to previous layer, how many tiles (NxN) to combine to make a tile in this layer
                    'num_rfs': 800,  # per tile
                    'lr': 1.0 / 1000,
                    'rel_lr_bg': 0.0,
                    'max_time': 5000000,
                    'do_raster_plots_every_k_im': None,  # or None
                    'network_type': 'seq-kmeans',  # seq-kmeans, seq-knn, nn-inits-kmeans
                }

                layer_wta = RateControlWTABRainLayerN(params=layer_params)

            tile_im_dim = tile_im_dim * 2

            self.layers.append(layer_wta)

