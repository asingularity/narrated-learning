import numpy as np
from utils import sintable, costable


class RobotEnvironment(object):
    def __init__(self, params):
        self.W = 24  # Width/Height
        self.H = 79

        self.RAYS = 360  # Should be 360!
        self.STEP = 6  # The step of for cycle. More = Faster, but large steps may
        # cause artifacts. Step 3 is great for radius 10.

        self.RAD = 100  # FOV radius.

        self.level = []
        self.view = []

        for x in range(self.W + 1):
            self.view.append([])
            self.level.append([])

            for y in range(self.H + 1):
                self.view[x].append(False)
                self.level[x].append(False)  # Every tile is floor

        self.level[5][5] = 1  # Add some walls
        self.level[11][11] = 1
        self.level[10][16] = 1
        self.level[11][17] = 1

        self.px = 10  # Player coordinates
        self.py = 10

        self.camera_image = np.zeros(len(range(0, self.RAYS + 1, self.STEP)))

    def step_environment(self, robot_model):

        pixel_index = -1
        for i in range(0, self.RAYS + 1, self.STEP):
            pixel_index += 1
            ax = sintable[i]  # Get precalculated value sin(x / (180 / pi))
            ay = costable[i]  # cos(x / (180 / pi))

            x = self.px  # Player's x
            y = self.py  # Player's y

            self.camera_image[pixel_index] = 0
            for z in range(self.RAD):  # Cast the ray
                x += ax
                y += ay

                if x < 0 or y < 0 or x > self.W or y > self.H:  # If ray is out of range
                    self.camera_image[pixel_index] = 0
                    break

                self.view[int(round(x))][int(round(y))] = 1  # Make tile visible

                if self.level[int(round(x))][int(round(y))] == 1:  # Stop ray if it hit
                    self.camera_image[pixel_index] = 1
                    break  # a wall.
