


import numpy as np


class NBrain(object):
    def __init__(self, params):
        '''

        'image_dim_NxN_pixels': IM_DIM,
            128
        'tile_dim_per_layer': TILE_DIM,
            [8, 16, 32, 64, 128]  # pixels, width & height
        'tile_tau_per_layer': TILE_TAU
            [1, 2,  4,  8,  16]  # spatiotemporal RF history steps
        'max_history_length': MAX_HISTORY_LENGTH,
            10600000 + 1
        'ims_scale_pixels': 800,

        :param params:


        IMPLEMENTATION
            initially, no overlap
                not strictly necessary for MVP
            ideally, minimize for-loops: minimize number of np.dot operations
                aim for: one dot product per layer
            each layer has two steps:
                collect input from previous layer
                dot product and normalize by sum

        '''

        self.im_dim = params['image_dim_NxN_pixels']
        self.tile_dim_per_layer = params['tile_dim_per_layer']
        self.tile_tau_per_layer = params['tile_tau_per_layer']
        self.ims_scale_pixels = params['ims_scale_pixels']
        self.max_time = params['max_history_length']

        self.bins_per_pixel = 6

        self.otm = OneTimeMessages()

        self.t = 0

        self.n_layers = len(self.tile_dim_per_layer)

        self.feature_len_per_layer = np.zeros(self.n_layers)

        for layer_n in range(self.n_layers):
            tile_dim = self.tile_dim_per_layer[layer_n]
            tile_tau = self.tile_tau_per_layer[layer_n]

            # for layer 0, tile dim is directly input
            # for higher layer, need to match with previous layer tiles that are in the pixel dim space

            # feature_len =



    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        # TODO compute actual (binned pixel, per tile) input_features here for layer 0
        layer_activity = None

        for layer_n in range(self.n_layers):

            input_features = self._get_input_features(prev_layer_index=)

            layer_weights = self.weights_per_layer[layer_n]
            layer_gains = self.gains_per_layer[layer_n]

            assert layer_weights.shape[0] == input_features.shape[0], str((layer_weights.shape, input_features.shape))
            assert layer_weights.shape[1] == input_features.shape[1], str((layer_weights.shape, input_features.shape))

            match_no_gain = np.divide(np.sum(np.multiply(layer_weights, input_features), axis=1), np.sum(layer_weights, axis=1))

            match = np.multiply(match_no_gain, layer_gains)



    def _bin_pixels_expand_columns(self, arr, num_bins_per_pixel):
        '''

        keep number of rows, expand in number of columns
        construct binary image: (0.0 or 1.0) per bin

        https://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
            use: digitize, or histogram

        https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.digitize.html
            digitize: returns index of bin to which each pixel belongs
            expanded array will have a set of bins per pixel, in the expanded columns

        :param arr:
        :return:
        '''
        #print('***')
        #print('arr', arr.shape)

        bins = np.linspace(0, 1+1e-9, num_bins_per_pixel + 1)  # TODO to fix binning add 1e-9 to 1 in second argument!
        #print('bins', bins.shape, bins)
        bin_indices = np.digitize(arr, bins) - 1  # same shape as arr; which bin, per pixel
        #print('bin_indices', bin_indices.shape, bin_indices)
        arr_exp = np.zeros((arr.shape[0], arr.shape[1] * num_bins_per_pixel), np.float32)
        #print('arr_exp', arr_exp.shape)

        #term1 = np.tile(self.num_bins_per_pixel * np.arange(arr.shape[1]), 2)  # can't remember why this is np.tile(..., 2)
        term1 = num_bins_per_pixel * np.arange(arr.shape[1])  # only works if arr rows is 1?
        term2 = bin_indices.flatten()
        #print('term1', term1.shape, term1)
        #print('term2', term2.shape, term2)
        c = term1 + term2
        r = np.repeat(np.arange(arr.shape[0]), arr.shape[1])

        arr_exp[r, c] = 1.0

        return arr_exp

    def _collapse_binned_columns_to_pixels(self, arr_exp, num_bins_per_pixel):
        '''

        non-trivial: arr_exp may no longer be binary; need to figure out how to collapse multiple values of different weight to one pixel:
            weighted average? take max value?

        :param arr_exp:
        :return:
        '''

        num_pixels = int((arr_exp.shape[1] / num_bins_per_pixel))
        num_rows = arr_exp.shape[0]

        # reshape so we can take max along one dim
        #   ie each row in this matrix is a pixel, temporarily
        # then reshape back

        tmp = arr_exp.reshape((num_rows * num_pixels, num_bins_per_pixel))

        bins = np.linspace(0, 1 + 1e-9, num_bins_per_pixel + 1)

        # max val for display:
        max_index = np.argmax(tmp, axis=1)
        max_vals = bins[max_index]

        weights = tmp[np.arange(tmp.shape[0]), max_index]

        #print(max_vals.shape, weights.shape)

        # weighted mean val for display:
        # tmp1 = np.multiply(tmp, bins[np.newaxis, 0:num_bins_per_pixel])
        # tmp2 = np.sum(tmp1, axis=1)  # 65536
        # tmp3 = np.divide(tmp2, np.sum(tmp, axis=1))
        # max_vals = tmp3

        weights = weights.reshape(num_rows, num_pixels)
        arr = max_vals.reshape(num_rows, num_pixels)

        return arr, weights

