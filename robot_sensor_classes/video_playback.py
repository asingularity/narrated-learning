import random
import cv2
import numpy as np
import pickle


class VideoPlaybackSensor(object):
    def __init__(self, params):
        '''

        :param params:
        '''

        self.preload_file = True

        self.image_dim = params['image_dim']
        self.video_dir = params['video_dir']
        self.video_filename = params['video_filename']

        self.stop_preload_at_frames = params['stop_preload_at_frames']  # 3000  # None: all frames
        self.use_full_frame = params['use_full_frame']
        self.partial_frame_factor = params['partial_frame_factor']

        self.skip_frame_count = params['skip_frame_count']  # 1 is normal; using 4 for model

        self.prop_use_for_holdout = params['prop_use_for_holdout']
        self.switch_to_holdout_frame = params['switch_to_holdout_frame']

        assert 0.0 < self.prop_use_for_holdout < 1.0

        assert params['return_type'] is np.float32 or params['return_type'] is np.float64
        self.return_type = params['return_type']

        if self.preload_file:
            self._sample_frames = []
            self._holdout_frames = []
            self._training_frames = []

            self._frame_index = 0
            self._preload_file()
        else:
            self.cap = cv2.VideoCapture(self.video_dir + self.video_filename)

        self.t = 0  # for switch to holdout time

    def _preload_file(self):

        pkl_filename = self.video_dir + self.video_filename + '_' + str(self.stop_preload_at_frames) + '_' + str(self.use_full_frame) + '_' + str(self.partial_frame_factor) + '_' + str(self.image_dim) + '.pkl'

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

                if self.stop_preload_at_frames is not None:
                    if fr > self.stop_preload_at_frames:
                        break

                if ret:

                    if fr % 100 == 0:
                        print('    preload frame:', fr)

                    gray = frame

                    mid_pt_r = 2 * gray.shape[0] / 4
                    mid_pt_c = 2 * gray.shape[1] / 4

                    im_rows = gray.shape[0]
                    im_cols = gray.shape[1]
                    min_dim = min(im_rows, im_cols)

                    if self.use_full_frame:
                        sample_im = gray[0:min_dim, 0:min_dim, :]
                    else:
                        factor = self.partial_frame_factor  # 8, but tried 4
                        sample_im = gray[mid_pt_r - min_dim / factor:mid_pt_r + min_dim / factor,
                                         mid_pt_c - min_dim / factor:mid_pt_c + min_dim / factor, :]
                        #

                    if not displayed_info:
                        print('before resize: ', sample_im.shape)
                    sample_im = cv2.resize(src=sample_im, dsize=(self.image_dim, self.image_dim),
                                           interpolation=cv2.INTER_NEAREST)
                    sample_im = cv2.cvtColor(sample_im, cv2.COLOR_BGR2GRAY)

                    if not displayed_info:
                        print('after resize: ', sample_im.shape)

                    self._sample_frames.append(sample_im.copy())

                    displayed_info = True

                    fr += 1

            self.save_to_pkl(pkl_filename=pkl_filename)

        print('Pre-loading complete.')

        # HACK for testing
        # self._frame_index = int(len(self._sample_frames) / 2)
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
        self._holdout_frames = all_frames[0:tmp_ind]
        self._training_frames = all_frames[tmp_ind:tmp_ind+num_holdout_frames]

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

    # @profile
    def read_input(self):
        '''

        :return: input image, color for now
        '''

        if self.preload_file:
            sample_im = self._sample_frames[self._frame_index]

            self._frame_index += self.skip_frame_count
            if self._frame_index >= len(self._sample_frames):
                print('resetting video')
                self._frame_index = 0
        else:

            ret, frame = self.cap.read()

            if not ret:
                # reload video
                print('reloading video')

                # TODO must put in thread!!!
                # TODO and/or make an option to preload sampled & resized video into memory
                self.cap = cv2.VideoCapture(self.video_dir + self.video_filename)
                ret, frame = self.cap.read()

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = frame
            # print(frame.shape)  # (2160, 3840, 3)

            mid_pt_r = 2 * gray.shape[0] / 4
            mid_pt_c = 2 * gray.shape[1] / 4

            # mid_pt_r = random.randint(self.image_dim/2 + 1, gray.shape[0] - self.image_dim/2 - 1)
            # mid_pt_c = random.randint(self.image_dim/2 + 1, gray.shape[1] - self.image_dim/2 - 1)

            im_rows = gray.shape[0]
            im_cols = gray.shape[1]
            min_dim = min(im_rows, im_cols)

            sample = True

            if sample:
                sample_im = gray[mid_pt_r - min_dim/8:mid_pt_r+min_dim/8,
                                 mid_pt_c - min_dim / 8:mid_pt_c + min_dim / 8, :]

                sample_im = cv2.resize(src=sample_im, dsize=(self.image_dim, self.image_dim),
                                       interpolation=cv2.INTER_NEAREST)
            else:
                # take square sample of full size, then scale down

                im_rows = gray.shape[0]
                im_cols = gray.shape[1]
                min_dim = min(im_rows, im_cols)

                sample_im = gray[0:min_dim, 0:min_dim, :]
                sample_im = cv2.resize(src=sample_im, dsize=(self.image_dim, self.image_dim),
                                       interpolation=cv2.INTER_NEAREST)

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
        'video_filename': 'videoplayback',  # 3840x2160
        # 'video_filename': 'P1033727.mp4',  # 3840x2160
        'stop_preload_at_frames': None,  # None: use whole video
        'use_full_frame': False,  # use the whole image
        'partial_frame_factor': 4,  # from center, what factor to use - larger factor ~ smaller part of image
        'return_type': np.float32,  # 32 or 64
        'skip_frame_count': 4,  # Normal is 1 (not 0!); += K frames from video on each step; so we can see prediction better for high frame rate videos
        'prop_use_for_holdout': 0.4,
        'switch_to_holdout_frame': None
    }
    p = VideoPlaybackSensor(params_video_playback)
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



