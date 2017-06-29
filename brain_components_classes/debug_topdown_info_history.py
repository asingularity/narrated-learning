
import numpy as np
np.set_printoptions(suppress=True)


class DebugTopdownInfoHistory(object):
    def __init__(self, params):
        self.max_history_length = params['max_history_length']

        # hard code 3: robot_x, robot_y, robot_theta
        self.td_info_array = np.zeros((self.max_history_length, 3)).astype(np.float32)
        self.t = 0

    def process_new_topdown_info(self, newest_topdown_info):
        '''
        extracts robot_x, robot_y, robot_theta and stores (as in states_history)
        :param newest_topdown_info:
        :return:
        '''
        self.td_info_array[self.t, :] = np.array([newest_topdown_info['robot_x'], newest_topdown_info['robot_y'], newest_topdown_info['robot_theta']])
        self.t += 1

    def get_td_info(self, delay):
        '''
        :param delay:
        :return: array([robot_x, robot_y, robot_theta])
        '''
        if self.t - 1 - delay >= 0:
            #print 'StatesHistory.get_state: self.t: ', self.t, ', index: ', self.t - 1 - delay
            td_info = self.td_info_array[self.t - 1 - delay, :]
            return td_info
        else:
            return None
