
import random
import numpy as np


class RobotModel(object):
    def __init__(self, params):
        self.max_angular_velocity = params['max_angular_velocity']
        self.linear_velocity = params['linear_velocity']
        self.last_motor_command = np.zeros(2).astype(np.float)

    def act_upon_processing(self, robot_brain):
        pass

    def get_delta_configuration(self):
        '''

        :return: linear speed, angular speed
        '''

        lin_val = self.linear_velocity
        ang_val = self.max_angular_velocity * 2.0 * (random.random() - 0.5)

        self.last_motor_command[0] = lin_val
        self.last_motor_command[1] = ang_val

        return lin_val, ang_val

    def get_last_motor_command(self):
        return self.last_motor_command
