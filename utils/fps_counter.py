
# version: 12/20/21

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

    def update(self, delta_frames=1, display_more=None, force_display=False, display_more_same_line=False):
        measured_fps = None
        if (time.time() - self.last_time > self.display_every_k_seconds) or force_display:
            measured_fps = self.frames / (time.time() - self.last_time)
            
            to_disp = 'FPS: ' + str(measured_fps)
            
            if display_more is not None:
                if display_more_same_line:
                    to_disp = to_disp + '    ' + display_more
                else:
                    to_disp = to_disp + '\n' + '    ' + display_more
            
            print(to_disp)

            self.frames = 0
            self.last_time = time.time()

        self.frames += delta_frames

        return measured_fps
