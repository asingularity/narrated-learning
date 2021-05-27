import random
import cv2
import numpy as np
import pickle
import os
import datetime
import shutil
from utils.fps_counter import FPSCounter


class VideoPlaybackSensor(object):
    def __init__(self, params):
        '''

        :param params:
        '''

        # ************************ params ************************

        # parameters that affect pre-loading; changing these forces a new preload and save to new pickle files:

        self.pickle_im_crop = params['pickle_im_square_crop']  # square crop to save to pickle: [r_0, c_0, dim]; None: saves centered largest possible square
        self.pickle_im_resize = params['pickle_im_square_resize']  # scale square crop to this dim; None: saves original size

        self.force_reload_to_pkl = params['force_reload_to_pkl']  # even if pkl folder exists, forces new pkl files from original video

        # these affect only what is output, given the ims saved to pickles. allows further subsampling:

        self.returned_im_crop = params['returned_im_square_crop']  # [r_0, c_0, dim], or None for full pickle im
        self.returned_im_dtype = params['returned_im_dtype']  # i.e. np.float32
        self.returned_im_use_color = params['returned_im_use_color']  # True: BGR, False: grayscale

        self.video_dir = params['video_dir']
        self.video_filename = params['video_filename']

        self.gb_per_pkl_file = params['gb_per_pkl_file']  # target size per pkl

        # asserts for constrained params, may change later:

        assert self.returned_im_use_color is False, 'color not yet supported'  # for now assume grayscale

        if self.force_reload_to_pkl:
            print()
            print('!!! warning: forcing reload of pkl files! this is time consuming! !!!')
            print()

        # ************************ init ************************

        pkl_folder_path = self._init_to_pickle()

        self.pkl_folder_path = pkl_folder_path
        self.last_pickle_done = False

        self.curr_pkl_sequence = None
        self.curr_pkl_num = None
        self.curr_pkl_frame_index = None

    def _init_to_pickle(self):

        video_file_path = self.video_dir + '/' + self.video_filename

        # multiple pickles now, we need to check first if all of them exist; but how to do without loading entire video to check video length?
        # solution:
        #   create a coded folder name based on what used to be the pickle name (params etc.)
        #   if coded folder does not yet exist:
        #       create a temp dir with random string name
        #       write pickle files in batches
        #       rename temp dir to coded folder name

        # how often to create pickles: maintain a constant rough size per pickle; so it depends on dim, color, etc.

        if self.pickle_im_crop is not None:
            crop_str = 'crop_' + str(int(self.pickle_im_crop[0])) + '_' + str(int(self.pickle_im_crop[1])) + '_' + str(int(self.pickle_im_crop[2]))
        else:
            crop_str = 'crop_none'

        if self.pickle_im_resize is not None:
            resize_str = 'resize_' + str(int(self.pickle_im_resize))
        else:
            resize_str = 'resize_none'

        pkl_folder_name = self.video_filename + '_' + crop_str + '_' + resize_str + '_pkls'
        pkl_folder_path = self.video_dir + '/' + pkl_folder_name
        pkl_folder_path_exists = os.path.isdir(pkl_folder_path)

        print()
        print('    pkl_folder_path:', pkl_folder_path)
        print()

        frames_per_pkl = None  # computed on first frame
        frames_this_pkl = 0
        pkl_num = 0
        pkl_sequence = []

        show_debug_frames = False

        if self.force_reload_to_pkl or (not pkl_folder_path_exists):

            if not pkl_folder_path_exists:
                os.makedirs(pkl_folder_path)

            fps = FPSCounter()

            ret = True
            print('    starting pre-loading...')
            cap = cv2.VideoCapture(self.video_dir + '/' + self.video_filename)

            fr = 0

            while ret:
                ret, frame = cap.read()
                if frame is None:
                    ret = False
                else:
                    orig_frame = frame.copy()

                    if self.pickle_im_crop is None:
                        # largest square, centered
                        min_dim = min(frame.shape[0], frame.shape[1])
                        if frame.shape[0] > min_dim:
                            start_i = int(0.5 * (frame.shape[0] - min_dim))
                            end_i = start_i + min_dim
                            frame = frame[start_i:end_i, 0:min_dim, :]
                        else:
                            start_i = int(0.5 * (frame.shape[1] - min_dim))
                            end_i = start_i + min_dim
                            frame = frame[0:min_dim, start_i:end_i, :]
                    else:
                        r_0 = self.pickle_im_crop[0]
                        c_0 = self.pickle_im_crop[1]
                        dim = self.pickle_im_crop[2]
                        frame = frame[r_0:r_0+dim, c_0:c_0+dim, :]

                    if self.pickle_im_resize is not None:
                        frame = cv2.resize(src=frame,
                                           dsize=(self.pickle_im_resize, self.pickle_im_resize),
                                           interpolation=cv2.INTER_NEAREST)

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if fr == 0:
                        bytes_per_pixel = 1  # uint8 is saved
                        bytes_per_frame = 1.0 * frame.shape[0] * frame.shape[1] * bytes_per_pixel
                        bytes_per_pkl_file = self.gb_per_pkl_file * 1e9 * 1.0
                        frames_per_pkl = int(bytes_per_pkl_file / bytes_per_frame)
                        print()
                        print('    frames_per_pkl:', frames_per_pkl)
                        print('    orig frame shape:', orig_frame.shape)
                        print('    pkl frame shape:', frame.shape)
                        print('    pkl frame dtype:', frame.dtype)
                        print()

                    pkl_sequence.append(frame)
                    frames_this_pkl += 1
                    write_pkl_now = frames_this_pkl > frames_per_pkl

                    if write_pkl_now:

                        self._write_pkl_file(pkl_sequence=pkl_sequence,
                                             pkl_num=pkl_num,
                                             pkl_folder_path=pkl_folder_path)

                        pkl_num += 1
                        frames_this_pkl = 0
                        pkl_sequence = []

                    fps.update(display_more='fr: ' + str(fr) + ', t (hours): ' + str(fr / (3600 * 30)))

                    if show_debug_frames:
                        cv2.imshow('orig', orig_frame)
                        cv2.imshow('pkl', frame)
                        cv2.waitKey(1)

                    fr += 1

            fps.update(force_display=True)

            # last pickle file might be smaller than the rest:
            self._write_pkl_file(pkl_sequence=pkl_sequence,
                                 pkl_num=pkl_num,
                                 pkl_folder_path=pkl_folder_path)

        return pkl_folder_path

    def _write_pkl_file(self, pkl_sequence, pkl_num, pkl_folder_path):

        pkl_filename = pkl_folder_path + '/' + 'data_' + str(pkl_num) + '.pkl'
        f = open(pkl_filename, 'wb')
        pickle.dump(pkl_sequence, f)
        f.close()

    def done(self):
        return self.last_pickle_done

    def _load_from_pkl(self, pkl_num, pkl_folder_path):

        try:
            print()
            print('Loading from pkl:', pkl_num)
            pkl_filename = pkl_folder_path + '/' + 'data_' + str(int(pkl_num)) + '.pkl'

            f = open(pkl_filename, 'rb')
            seq = pickle.load(f)
            f.close()
            assert len(seq) > 0
            print('Done loading pkl.')
            print()

        except:
            print()
            print('!!! Could not load pickle! pkl_num:', pkl_num)
            print()
            seq = None  # no pickle files left
            #raise

        return seq

    def read_input(self):
        # init first pickle file if needed
        if self.curr_pkl_sequence is None:
            self.curr_pkl_num = 0
            self.curr_pkl_frame_index = 0
            self.curr_pkl_sequence = self._load_from_pkl(pkl_num=self.curr_pkl_num,
                                                         pkl_folder_path=self.pkl_folder_path)
            self.curr_pkl_num += 1  # really the next pickle num

        # set current frame to return
        output_im = self.curr_pkl_sequence[self.curr_pkl_frame_index]

        if self.returned_im_crop is not None:
            r0 = self.returned_im_crop[0]
            c0 = self.returned_im_crop[1]
            dim = self.returned_im_crop[2]

            output_im = output_im[r0:r0 + dim, c0:c0 + dim]

        output_im = output_im.astype(self.returned_im_dtype) * 1.0 / 255.

        # read from next pickle file if needed:
        if self.curr_pkl_frame_index == len(self.curr_pkl_sequence) - 1:
            self.curr_pkl_sequence = self._load_from_pkl(pkl_num=self.curr_pkl_num,
                                                         pkl_folder_path=self.pkl_folder_path)

            if self.curr_pkl_sequence is None:
                # if no next pickle file found:
                self.last_pickle_done = True

            self.curr_pkl_num += 1
            self.curr_pkl_frame_index = 0
        else:
            self.curr_pkl_frame_index += 1

        return output_im


if __name__ == '__main__':

    take_crop = True
    if take_crop:
        # assumes 256 as pickle_im_square_resize for centering (128 below):
        RF_IM_DIM = 16
        square_crop = (128 - int(RF_IM_DIM/2), 128 - int(RF_IM_DIM/2), RF_IM_DIM)
    else:
        square_crop = None

    vid = VideoPlaybackSensor(params={'pickle_im_square_crop': None,
                                      'pickle_im_square_resize': 256,
                                      'force_reload_to_pkl': False,
                                      'gb_per_pkl_file': 1,
                                      'returned_im_square_crop': square_crop,
                                      'returned_im_dtype': np.float32,  # only np.float32 supported
                                      'returned_im_use_color': False,  # only False supported for now
                                      'video_dir': '/srv/projects/video-downloads',
                                      'video_filename': 'seattle-driving-fkps18H3SXY.mp4'
                                      })

    print()
    print('VideoPlaybackSensor init done!')
    print()

    view_debug = True
    fps = FPSCounter()

    while not vid.done():
        im = vid.read_input()
        if view_debug:
            cv2.imshow('im', im)
            cv2.waitKey(1)
        fps.update()


