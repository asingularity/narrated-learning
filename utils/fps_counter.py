
import time

class FPSCounter(object):
    def __init__(self, params={}, sec=None):
        if 'display_every_k_seconds' in params:
            self.display_every_k_seconds = params['display_every_k_seconds']
        else:
            self.display_every_k_seconds = 5

        if sec is not None:
            self.display_every_k_seconds = sec

        self.frames = 0
        self.last_time = time.time()

    def update(self):
        measured_fps = None
        if time.time() - self.last_time > self.display_every_k_seconds:
            measured_fps = self.frames / (time.time() - self.last_time)
            print ('FPS:', measured_fps)
            self.frames = 0
            self.last_time = time.time()

        self.frames += 1

        return measured_fps
