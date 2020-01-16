import random
import cv2
import numpy as np


class VideoPlaybackSensor(object):
    def __init__(self, params):
        '''

        :param params:
        '''

        self.preload_file = True

        self.image_dim = params['image_dim']
        self.video_filename = params['video_filename']

        self.stop_preload_at_frames = params['stop_preload_at_frames']  # 3000  # None: all frames
        self.use_full_frame = params['use_full_frame']
        self.partial_frame_factor = params['partial_frame_factor']

        if self.preload_file:
            self._sample_frames = []

            self._frame_index = 0
            self._preload_file()
        else:
            self.cap = cv2.VideoCapture(self.video_filename)

    def _preload_file(self):
        displayed_info = False

        ret = True
        print('Starting Pre-loading...')
        cap = cv2.VideoCapture(self.video_filename)

        fr = 0

        while ret:
            ret, frame = cap.read()

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

        print('Pre-loading complete.')

        # HACK for testing
        # self._frame_index = int(len(self._sample_frames) / 2)
        self._frame_index = 0

        print('starting on frame index: ', self._frame_index, 'of', len(self._sample_frames))
    # @profile
    def read_input(self):
        '''

        :return: input image, color for now
        '''

        if self.preload_file:
            sample_im = self._sample_frames[self._frame_index]

            self._frame_index += 1
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
                self.cap = cv2.VideoCapture(self.video_filename)
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

        sample_im = sample_im.astype(np.float) * 1.0/255.

        assert sample_im.shape[0] == self.image_dim
        assert sample_im.shape[1] == self.image_dim

        return sample_im
