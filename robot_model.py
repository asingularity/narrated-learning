
import random

class RobotModel(object):
    def __init__(self, params):
        self.max_angular_velocity = params['max_angular_velocity']
        self.linear_velocity = params['linear_velocity']

    def act_upon_processing(self, robot_brain):
        pass

    def get_delta_configuration(self):
        '''

        :return: linear speed, angular speed
        '''

        return self.linear_velocity, self.max_angular_velocity * 2.0 * (random.random() - 0.5)
