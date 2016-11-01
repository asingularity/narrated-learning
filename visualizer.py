import time
import logging
logging.basicConfig(level=logging.INFO)


class Visualizer(object):
    def __init__(self, params):
        self.fps_frames = 0
        self.frames = 0
        self.last_FPS_time = time.time()
        self.fps_display_interval = params['fps_display_interval']

    def _display_fps(self):
        if time.time() - self.last_FPS_time > self.fps_display_interval:
            fps = self.fps_frames * 1.0 / (time.time() - self.last_FPS_time)
            logging.info('FPS: ' + str(fps))
            self.fps_frames = 0
            self.last_FPS_time = time.time()
        self.fps_frames += 1

    def _display_text_map(self):
        for x in range(W + 1):
            for y in range(H + 1):
                if (x, y) == (px, py):
                    sys.stdout.write("@")  # Player
                elif fov[x][y] == 1:
                    if level[x][y] == 0:  # Seen
                        sys.stdout.write(".")
                    else:
                        sys.stdout.write("#")
                else:
                    sys.stdout.write("?")  # Unseen
            print

    def visualize(self, robot_sensors, robot_brain, robot_model, robot_environment):
        self._display_text_map()
        self._display_fps()
