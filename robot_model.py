

class RobotModel(object):
    def __init__(self, params):
        self.max_angular_velocity = params['max_angular_velocity']

    def act_upon_processing(self, robot_brain):
        pass

    def get_delta_configuration(self):
        '''

        :return: linear speed, angular speed
        '''

        return 0.0, self.max_angular_velocity
