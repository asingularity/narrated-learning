
import time

class FPSCounter(object):
    def __init__(self, params):
        self.display_every_k_seconds = params['display_every_k_seconds']
        self.frames = 0
        self.last_time = time.time()

    def update(self):
        if time.time() - self.last_time > self.display_every_k_seconds:
            print 'FPS:', self.frames / (time.time() - self.last_time)
            self.frames = 0
            self.last_time = time.time()

        self.frames += 1
