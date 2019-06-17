import pickle
import numpy as np
np.set_printoptions(suppress=True)


class MotorHistory(object):
    def __init__(self, params):
        self.max_history_length = params['max_history_length']
        self.dim = params['dim']

        self.motor_array = np.zeros((self.max_history_length, self.dim)).astype(np.float32)
        self.t = 0

    def process_new_motor_command(self, motor_command):
        self.motor_array[self.t, :] = motor_command[:]
        self.t += 1

    def get_sequence(self, delay_start, delay_end):
        assert delay_start >= delay_end, 'delay_start must be >= delay_end ' + str(delay_start) + ', ' + str(delay_end)
        #print 'MotorHistory.get_sequence: self.t: ', self.t, ', range: [', self.t - 1 - delay_start, ', ', self.t - delay_end, ')'
        return_arr = self.motor_array[self.t - 1 - delay_start:self.t - delay_end, :].flatten()
        return return_arr
