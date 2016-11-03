import time
import sys
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

    def _display_text_map(self, robot_environment):
        W = robot_environment.W
        H = robot_environment.H
        view = robot_environment.view
        level = robot_environment.level
        px = robot_environment.px
        py = robot_environment.py

        for x in range(W + 1):
            for y in range(H + 1):
                if (x, y) == (px, py):
                    sys.stdout.write("@")  # Player
                elif view[x][y] == 1:
                    if level[x][y] == 0:  # Seen
                        sys.stdout.write(".")
                    else:
                        sys.stdout.write("#")
                else:
                    sys.stdout.write("?")  # Unseen
            print

    def visualize(self, robot_sensors, robot_brain, robot_model, robot_environment):
        if self.frames % 5000 == 0:
            self._display_text_map(robot_environment)
        self._display_fps()

        self.frames += 1