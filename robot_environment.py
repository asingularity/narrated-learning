import numpy as np
from utils import sintable, costable


class RobotEnvironment(object):
    def __init__(self, params):
        W = 24  # Width/Height
        H = 79

        RAYS = 360  # Should be 360!

        STEP = 6  # The step of for cycle. More = Faster, but large steps may
        # cause artifacts. Step 3 is great for radius 10.

        RAD = 100  # FOV radius.

        self.level = []
        self.view = []

        for x in range(W + 1):
            self.view.append([])
            self.level.append([])

            for y in range(H + 1):
                fov[x].append(False)
                level[x].append(False)  # Every tile is floor

        level[5][5] = 1  # Add some walls
        level[11][11] = 1
        level[10][16] = 1
        level[11][17] = 1

        px = 10  # Player coordinates
        py = 10

        camera_image = np.zeros(len(range(0, RAYS + 1, STEP)))


    def step_environment(self, robot_model):

        pixel_index = -1
        for i in range(0, RAYS + 1, STEP):
            pixel_index += 1
            ax = sintable[i]  # Get precalculated value sin(x / (180 / pi))
            ay = costable[i]  # cos(x / (180 / pi))

            x = px  # Player's x
            y = py  # Player's y

            camera_image[pixel_index] = 0
            for z in range(RAD):  # Cast the ray
                x += ax
                y += ay

                if x < 0 or y < 0 or x > W or y > H:  # If ray is out of range
                    camera_image[pixel_index] = 0
                    break

                fov[int(round(x))][int(round(y))] = 1  # Make tile visible

                if level[int(round(x))][int(round(y))] == 1:  # Stop ray if it hit
                    camera_image[pixel_index] = 1
                    break  # a wall.
