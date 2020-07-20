

import random
import cv2
import numpy as np
import pickle


class DuoPlaybackSensor(object):
    def __init__(self, params):
        '''

        :param params:
        '''

        self.image_dim = params['image_dim']
        self.video_dir = params['video_dir']
        self.video_filename = params['video_filename']

        self.skip_frame_count = params['skip_frame_count']

        self.prop_use_for_holdout = params['prop_use_for_holdout']
        self.switch_to_holdout_frame = params['switch_to_holdout_frame']

        self.partial_frame_factor = params['partial_frame_factor']

        assert 0.0 < self.partial_frame_factor < 1.0

        assert 0.0 < self.prop_use_for_holdout < 1.0

        assert params['return_type'] is np.float32 or params['return_type'] is np.float64
        self.return_type = params['return_type']

        self._sample_frames = []
        self._holdout_frames = []
        self._training_frames = []

        self._frame_index = 0
        self._preload_file()

        self.t = 0  # for switch to holdout time

    def _preload_file(self):

        pkl_filename = self.video_dir + self.video_filename + '_' + str(self.image_dim) + '.pkl'

        print('Trying to load from pkl...')

        try:
            self.load_from_pkl(pkl_filename=pkl_filename)
        except:
            print('pkl not found, loading video...')

            displayed_info = False

            ret = True
            print('Starting Pre-loading...')
            cap = cv2.VideoCapture(self.video_dir + self.video_filename)

            fr = 0

            while ret:
                ret, frame = cap.read()

                if ret:
                    if fr % 1000 == 0:
                        print('    preload frame:', fr)

                    # take only left half of frame, as this is a stereo frame
                    frame = frame[:, 0:int(frame.shape[1] / 2), :]

                    dim_keep_r = int(self.partial_frame_factor * frame.shape[0])
                    r_start = int(0.5 * (frame.shape[0] - dim_keep_r))
                    r_end = r_start + dim_keep_r
                    dim_keep_c = int(self.partial_frame_factor * frame.shape[1])
                    c_start = int(0.5 * (frame.shape[0] - dim_keep_c))
                    c_end = c_start + dim_keep_c
                    sample_im = frame[r_start:r_end, c_start:c_end, :]

                    im_rows = sample_im.shape[0]
                    im_cols = sample_im.shape[1]
                    min_dim = min(im_rows, im_cols)

                    sample_im = sample_im[0:min_dim, 0:min_dim, :]

                    if not displayed_info:
                        print('before resize: ', sample_im.shape)

                    sample_im = cv2.resize(src=sample_im, dsize=(self.image_dim, self.image_dim),
                                           interpolation=cv2.INTER_NEAREST)
                    sample_im = cv2.cvtColor(sample_im, cv2.COLOR_BGR2GRAY)

                    if not displayed_info:
                        print('after resize: ', sample_im.shape)

                    # cv2.imshow('im', sample_im)
                    # cv2.waitKey(1)

                    self._sample_frames.append(sample_im.copy())

                    displayed_info = True

                    fr += 1

            self.save_to_pkl(pkl_filename=pkl_filename)

            print('Save done, now loading as test...')

            self.load_from_pkl(pkl_filename=pkl_filename)

        print('Pre-loading complete.')

        self._frame_index = 0

        print('starting on frame index: ', self._frame_index, 'of', len(self._sample_frames))

    def load_from_pkl(self, pkl_filename):
        print('Loading from pkl...')
        f = open(pkl_filename, 'rb')
        all_frames = pickle.load(f)
        f.close()
        print('Loading complete.')

        num_holdout_frames = int(self.prop_use_for_holdout * len(all_frames))
        tmp_ind = len(all_frames) - num_holdout_frames
        self._training_frames = all_frames[0:tmp_ind]
        self._holdout_frames = all_frames[tmp_ind:tmp_ind + num_holdout_frames]

        print()
        print('_training_frames:', len(self._training_frames))
        print('_holdout_frames:', len(self._holdout_frames))

        self._sample_frames = self._training_frames

    def switch_to_holdout(self):
        self._sample_frames = self._holdout_frames
        self._frame_index = 0

    def save_to_pkl(self, pkl_filename):
        print('Saving to pkl...')
        f = open(pkl_filename, 'wb')
        pickle.dump(self._sample_frames, f)
        f.close()
        print('Saving complete.')

    def read_input(self):
        sample_im = self._sample_frames[self._frame_index]

        self._frame_index += self.skip_frame_count
        if self._frame_index >= len(self._sample_frames):
            print('resetting video')
            self._frame_index = 0

        sample_im = sample_im.astype(self.return_type) * 1.0/255.

        assert sample_im.shape[0] == self.image_dim, (sample_im.shape[0], self.image_dim)
        assert sample_im.shape[1] == self.image_dim, (sample_im.shape[1], self.image_dim)

        self.t += 1

        if self.switch_to_holdout_frame is not None:
            if self.t == self.switch_to_holdout_frame:
                print()
                print('INPUT IS SWITCHING TO HOLDOUT SET')
                print()

                self.switch_to_holdout()

        return sample_im



if __name__ == '__main__':
    IM_DIM = 128
    params_video_playback = {
        'image_dim': IM_DIM,  # sensor class has to figure out subset & scale to achieve this dim
        'video_dir': '/srv/projects/NL-data/',
        'video_filename': 'DUOCapture-19-07-2020-14-26-57-330_20k_frames.avi',  #
        'partial_frame_factor': 0.4,
        'return_type': np.float32,  # 32 or 64
        'skip_frame_count': 4,  # Normal is 1 (not 0!); += K frames from video on each step; so we can see prediction better for high frame rate videos
        'prop_use_for_holdout': 0.4,
        'switch_to_holdout_frame': None
    }
    p = DuoPlaybackSensor(params_video_playback)
    print()
    print('training...')
    print()
    for fr in range(4000):
        im = p.read_input()
        cv2.imshow('im', im)
        cv2.waitKey(10)

    print()
    print('holdout...')
    print()

    p.switch_to_holdout()

    for fr in range(4000):
        im = p.read_input()
        cv2.imshow('im', im)
        cv2.waitKey(10)




















