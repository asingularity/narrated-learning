
import random
import numpy as np


class RobotModel(object):
    def __init__(self, params):
        self.max_angular_velocity = params['max_angular_velocity']
        self.linear_velocity = params['linear_velocity']
        self.last_motor_command = np.zeros(2).astype(np.float)
        self.new_angle = None

    def act_upon_processing(self, robot_brain):
        self.new_angle = robot_brain.get_motor_output()

    def get_delta_configuration(self):
        '''

        :return: linear speed, angular speed
        '''

        lin_val = self.linear_velocity
        if self.new_angle is None:
            ang_val = self.max_angular_velocity * 2.0 * (random.random() - 0.5)
        else:
            ang_val = self.new_angle

        self.last_motor_command[0] = lin_val
        self.last_motor_command[1] = ang_val

        return lin_val, ang_val

    def get_last_motor_command(self):
        return self.last_motor_command
