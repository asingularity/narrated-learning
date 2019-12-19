import random
import cv2
import numpy as np


class VideoPlaybackSensor(object):
    def __init__(self, params):
        '''

        :param params:
        '''

        self.image_dim = params['image_dim']
        self.video_filename = params['video_filename']
        self.cap = cv2.VideoCapture(self.video_filename)

    def read_input(self):
        '''

        :return: input image, color for now
        '''

        ret, frame = self.cap.read()

        if not ret:
            # reload video
            print('reloading video')
            self.cap = cv2.VideoCapture(self.video_filename)
            ret, frame = self.cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame

        #mid_pt_r = 3 * gray.shape[0] / 4
        #mid_pt_c = 2 * gray.shape[1] / 4

        mid_pt_r = random.randint(self.image_dim/2 + 1, gray.shape[0] - self.image_dim/2 - 1)
        mid_pt_c = random.randint(self.image_dim/2 + 1, gray.shape[1] - self.image_dim/2 - 1)

        sample = True
        if sample:

            sample_im = gray[mid_pt_r - self.image_dim/2:mid_pt_r+self.image_dim/2,
                             mid_pt_c - self.image_dim / 2:mid_pt_c + self.image_dim / 2, :]


        else:
            # TODO take square sample of full size, then scale down

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
