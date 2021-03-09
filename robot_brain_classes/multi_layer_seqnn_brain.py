'''

this file:
    multi_layer_seqnn_brain.py
    started 3/6/21

goes with:
    run_demo_perception.py
    offline_analyses/...

this is the multi layer, optimized, sped up newer version of:
    [ ]     run_demo_predict.py
    [X]     robot_brain_classes/two_stage_seqnn_brain.py
    [ ]     offline_analyses/try_sparsify_3.py

'''

import time
import cv2
import numpy as np
np.set_printoptions(suppress=True, precision=2)
import random
import pickle
from math import sqrt
from brain_components_classes.states_history import StatesLimitedHistory
from cython_eff import compute_eff

from utils.one_time_messages import OneTimeMessages

from cuda_dist_query import CudaTable

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log






class MultiLayerSeqNNBrain(object):
    def __init__(self, params):
        '''

        :param params:
        '''

    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        return

        for layer_n in range(self.num_layers):

            layer_output = self._run_layer(layer_n=layer_n,
                                           layer_input=layer_input)

    def run_layer(self, layer_n, layer_input):
        '''

        :param layer_n:
        :param layer_input:
        :return:
        '''

        # update seq-nn table

        # periodically, re-train RFs using max-bin-append; library from offline_analyses

        # store output for current frame into a states history, to use as input for next layer

    def get_table_ims(self):
        '''

        :return:
        '''

        ims_list = []
        ims_names_list = []

        return ims_list, ims_names_list





class MultiLayerSeqNNBrainWithPredict(object):
    def __init__(self, params):
        '''

        FOR LATER

        :param params:
        '''

    def process_input(self, input_im):
        '''

        :param input_im:
        :return:
        '''

        for layer_n in range(self.num_layers):

            # define layer_input

            self._process_layer(layer_n=layer_n, layer_input=layer_input)

    def _process_layer(self, layer_input):
        '''

        :param layer_input:
        :return:
        '''

        # layer is composed of two stages:
        #   "sparsify"
        #   "predict"

        self._do_stage_sparsify(layer_input)
        #self._do_stage_predict(layer_input)

    def _do_stage_sparsify(self, layer_input):
        '''

        (1) compute sparsify stage events -> training for predictive stage
        (2) learn sparsify stage given layer input

        :param layer_input:
        :return:
        '''

    def _do_stage_predict(self, layer_input):
        '''

        (1) compute predictive stage events -> input to next layer
        (2) learn predictive stage given sparsify stage events

        :param layer_input:
        :return:
        '''

    def get_table_ims(self):
        '''

        :return:
        '''

        return ims_list, ims_names_list


































