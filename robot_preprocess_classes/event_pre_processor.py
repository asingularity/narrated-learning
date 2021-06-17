
import cv2
import numpy as np
from robot_sensor_classes.video_playback import VideoPlaybackSensor
from utils.fps_counter import FPSCounter


class EventPreProcessor(object):
    '''
        This class generates events from frame-based images
        i.e. it simulates an event-based camera
    '''

    def __init__(self, params):
        self.brightness_threshold = params['brightness_threshold']
        self.im_dim = params['im_dim']  # N, for NxN image

        self.state_dim = self.im_dim * self.im_dim
        self.last_event_brightness = np.zeros(self.state_dim)
        self.t = 0

        r = np.tile(np.arange(self.im_dim)[:, np.newaxis], 5).astype(np.int)
        c = np.transpose(r).astype(np.int)
        self.event_coords_r = r.flatten()
        self.event_coords_c = c.flatten()

    def step(self, input_frame):
        '''

        assumes input_frame is grayscale already

        :param input_frame:
        :return:
        '''

        # events_arr: binary array of pixels: 1 if event this time step, 0 if no event this time step
        input_state = input_frame.flatten()

        if self.t == 0:
            self.last_event_brightness[:] = input_state[:]

        nnz_events_p = np.nonzero((input_state - self.last_event_brightness) > self.brightness_threshold)
        nnz_events_n = np.nonzero((input_state - self.last_event_brightness) < -self.brightness_threshold)

        self.last_event_brightness[nnz_events_p] = input_state[nnz_events_p]
        self.last_event_brightness[nnz_events_n] = input_state[nnz_events_n]

        events_arr_p = np.zeros(input_state.shape[0], np.float32)
        events_arr_n = np.zeros(input_state.shape[0], np.float32)

        events_arr_p[nnz_events_p] = 1
        events_arr_n[nnz_events_n] = 1

        self.t += 1

        original_input_image = input_frame.copy()
        return events_arr_p, events_arr_n, self.event_coords_r.copy(), self.event_coords_c.copy(), original_input_image


if __name__ == '__main__':

    RF_IM_DIM = 8  # 8
    ROOT_DIR = '/srv'
    square_crop = (128 - int(RF_IM_DIM / 2), 128 - int(RF_IM_DIM / 2), RF_IM_DIM)

    params_video_playback = {
        'pickle_im_square_crop': None,
        'pickle_im_square_resize': 256,
        'force_reload_to_pkl': False,
        'gb_per_pkl_file': 1,
        'returned_im_square_crop': square_crop,
        'returned_im_dtype': np.float32,  # only np.float32 supported
        'returned_im_use_color': False,  # only False supported for now
        'video_dir': ROOT_DIR + '/projects/video-downloads',
        #'video_filename': 'sea-turtles-yLuEx-XH3Uc.mp4'
        'video_filename': 'seattle-driving-fkps18H3SXY.mp4'
        #'video_filename': 'sea-turtles-11hr-spxtEt6RaS4.mp4'
    }

    # TODO: change color to gray in this class to preserve more dynamic range, and test this
    vp = VideoPlaybackSensor(params=params_video_playback)

    ep = EventPreProcessor(params={
        'brightness_threshold': 10.0 / 255,
        'im_dim': RF_IM_DIM
    })

    fps = FPSCounter()

    t = 0

    event_size_counts = np.zeros(RF_IM_DIM * RF_IM_DIM + 1)

    while True:
        im = vp.read_input()
        events_p, events_n = ep.step(input_frame=im)

        im_event_p = events_p.reshape((im.shape[0], im.shape[1]))
        im_event_n = events_n.reshape((im.shape[0], im.shape[1]))

        im_events = im_event_p - im_event_n  # [-1, 0, 1]
        im_events = 0.5 * im_events  # [-0.5, 0, 0.5]
        im_events = 0.5 + im_events  # [0, 0.5, 1.0]

        cv2.imshow('im', im)
        cv2.imshow('im_events', im_events)
        #print(np.sum(events_p) + np.sum(events_n))
        cv2.waitKey(1)

        event_size_counts[int(np.sum(events_p) + np.sum(events_n))] += 1

        # print(im.dtype, np.amin(im), np.amax(im))
        # print(255 * np.unique(im))

        fps.update(display_more='t: ' + str(t))

        if t % 1000 == 0:
            print(event_size_counts * 100.0/np.sum(event_size_counts))

        t += 1

    # show:
    #   event image on each frame (what pixels had an event? their polarity?)
    #   plot a raster in a tmp plot folder on /srv/projects/


