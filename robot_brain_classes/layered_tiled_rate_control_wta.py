
from robot_brain_classes.tiled_rate_control_wta import RateControlWTABRainLayer0, RateControlWTABRainLayerN


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

    def _init_layers_hardcode(self, params):
        '''

        layer     tile_dim_NxN        tile dim pixels     num tiles
        0         -                   8x8                 32x32
        1         4x4                 32x32               8x8
        2         4x4                 128x128             2x2
        3         2x2                 256x256             1x1

        :return:
        '''

        layer_params_list = []

        # 0
        layer_params_list.append({
            'input_im_dim': params['input_im_dim'],
            'tile_im_dim': 8,
            'num_rfs': 800,  # per tile
            'lr': 1.0 / 1000,
            'input_concat_timesteps': 1
        })

        # 1
        layer_params_list.append({
            'input_num_tiles_NxN': 32,  # previous layer (num_tiles X num_tiles)
            'input_num_rfs_per_tile': layer_params_list[-1]['num_rfs'],  # previous layer num rfs
            'tile_dim_NxN': 1,  # relative to previous layer, how many tiles (NxN) to combine to make a tile in this layer
            'num_rfs': 800,  # per tile
            'lr': 1.0 / 1000,
            'input_concat_timesteps': 4
        })

        if False:  # later add more
            # 2
            layer_params_list.append({
                'input_num_tiles_NxN': 8,  # previous layer (num_tiles X num_tiles)
                'input_flat_dim_per_tile': layer_params_list[-1]['num_rfs'],  # previous layer num rfs
                'tile_dim_NxN': 4,  # relative to previous layer, how many tiles (NxN) to combine to make a tile in this layer
                'num_rfs': 800,  # per tile
                'lr': 1.0 / 1000,
                'input_concat_timesteps': 1
            })

            # 3
            layer_params_list.append({
                'input_num_tiles_NxN': 2,  # previous layer (num_tiles X num_tiles)
                'input_flat_dim_per_tile': layer_params_list[-1]['num_rfs'],  # previous layer num rfs
                'tile_dim_NxN': 2,  # relative to previous layer, how many tiles (NxN) to combine to make a tile in this layer
                'num_rfs': 800,  # per tile
                'lr': 1.0 / 1000,
                'input_concat_timesteps': 2
            })

        self.num_layers = 2

        self.layers = []

        layer_num = 0
        for layer_params in layer_params_list:
            # common params
            layer_params['rel_lr_bg'] = 0.0
            layer_params['max_time'] = 5000000
            layer_params['do_raster_plots_every_k_im'] = None
            layer_params['network_type'] = 'seq-kmeans'

            if layer_num == 0:
                self.layers.append(RateControlWTABRainLayer0(params=layer_params))
            else:
                self.layers.append(RateControlWTABRainLayerN(params=layer_params))

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

    def _init_layers_with_loop(self, params):
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

        for layer_n in range(self.num_layers):

            if layer_n == 0:
                output_events = self.layers[layer_n].process_input(input_events_p=input_events_p,
                                                                   input_events_n=input_events_n,
                                                                   event_coords_r=event_coords_r,
                                                                   event_coords_c=event_coords_c,
                                                                   original_input_image=original_input_image)
            else:
                output_events = self.layers[layer_n].process_input(input_events)

            input_events = output_events.copy()

    def get_table_ims(self):

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

        return ims_list, ims_names_list

    def do_plots(self):
        pass


