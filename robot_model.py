
import random
import numpy as np


class RobotModel(object):
    def __init__(self, params):
        self.max_angular_velocity = params['max_angular_velocity']
        self.linear_velocity = params['linear_velocity']
        self.last_motor_command = np.zeros(2).astype(np.float)
        self.new_angle_val = None

        # TODO refactor so that get_delta_configuration can be called multiple times / doesn't change the state

    def act_upon_processing(self, motor_command):
        self.new_angle_val = motor_command

    def get_delta_configuration(self):
        '''

        :return: linear speed, angular speed
        '''

        lin_val = self.linear_velocity
        if self.new_angle_val is None:
            ang_val = self.max_angular_velocity * 2.0 * (random.random() - 0.5)
        else:
            ang_val = self.new_angle_val

        self.last_motor_command[0] = lin_val
        self.last_motor_command[1] = ang_val

        return lin_val, ang_val

    def get_last_motor_command(self):
        return self.last_motor_command
