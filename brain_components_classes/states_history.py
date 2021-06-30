
import pickle
import numpy as np
np.set_printoptions(suppress=True)


class StatesHistory(object):
    def __init__(self, params):
        self.max_history_length = params['max_history_length']
        self.states_dim_list = params['states_dim_list']
        self.state_arrays_list = []
        for k in range(len(self.states_dim_list)):
            self.state_arrays_list.append(np.zeros((self.max_history_length, self.states_dim_list[k])).astype(np.float64))
        self.t = 0

    def store_new_states(self, newest_states_list):
        self.process_new_states(newest_states_list=newest_states_list)

    def process_new_states(self, newest_states_list):

        assert len(newest_states_list) == len(self.state_arrays_list), 'Error: invalid states length!'

        state_index = 0
        for state in newest_states_list:
            self.state_arrays_list[state_index][self.t, :] = state[:]
            state_index += 1
        self.t += 1

    def get_state(self, state_index, delay):
        if self.t - 1 - delay >= 0:
            #print 'StatesHistory.get_state: self.t: ', self.t, ', index: ', self.t - 1 - delay
            state = self.state_arrays_list[state_index][self.t - 1 - delay, :]
            return state
        else:
            return None

    def save_states(self, plots_save_folder, state_indices_list):
        for state_index in state_indices_list:
            f = open(plots_save_folder + '/states_history_' + str(state_index) + '.pkl', 'wb')
            pickle.dump(self.state_arrays_list[state_index], f)
            f.close()


class StatesLimitedHistory(object):
    def __init__(self, params):
        self.max_delay = params['max_delay'] + 1  # + 1 for no safety in case you think of it off by 1
        self.states_dim_list = params['states_dim_list']
        self.state_arrays_list = []
        self.extra_data_list = []

        self.active = False
        self.store_extra_data = True
        if 'store_extra_data' in params:
            self.store_extra_data = params['store_extra_data']

        if self.states_dim_list[0] > 0:
            self.active = True
            for k in range(len(self.states_dim_list)):
                self.state_arrays_list.append(np.zeros((self.max_delay, self.states_dim_list[k])).astype(np.float32))  # TODO MAKE DTYPE A PARAM, DEFAULT TO np.float64 !!!!!!!
                if self.store_extra_data:
                    self.extra_data_list.append([None] * self.max_delay)
        self.t_mod = 0

    def store_new_states(self, newest_states_list, extra_data_list=None):
        self.process_new_states(newest_states_list=newest_states_list, extra_data_list=extra_data_list)

    def process_new_states(self, newest_states_list, extra_data_list=None):
        if self.active:
            assert len(newest_states_list) == len(self.state_arrays_list), 'Error: invalid states length!'
            if self.store_extra_data:
                assert len(extra_data_list) == len(self.extra_data_list), 'Error: invalid extra data length!'
                assert len(newest_states_list) == len(self.extra_data_list), 'Error: invalid extra data length!'

            self.t_mod += 1
            if self.t_mod == self.max_delay:
                self.t_mod = 0

            state_index = 0
            for state in newest_states_list:
                self.state_arrays_list[state_index][self.t_mod, :] = state[:]
                if self.store_extra_data:
                    self.extra_data_list[state_index][self.t_mod] = extra_data_list[state_index]
                state_index += 1

    def get_state(self, delay, state_index=0):
        if self.active:
            assert delay <= self.max_delay
            assert self.t_mod < self.max_delay
            # t_mod is where most recent data point is stored

            time_index = self.t_mod - delay

            if time_index < 0:
                time_index = self.max_delay + time_index

            state = self.state_arrays_list[state_index][time_index, :]
            if self.store_extra_data:
                extra_data = self.extra_data_list[state_index][time_index]
                return state, extra_data
            else:
                return state
        else:
            if self.store_extra_data:
                return None, None
            else:
                return None

    def get_state_sequence(self, delay_long, delay_short, state_index=0, oldest_first=True):
        # TODO this could be more efficient: just two lookups instead!

        if self.active:
            assert delay_long <= self.max_delay
            assert delay_short <= self.max_delay
            assert delay_short <= delay_long

            state_seq = []
            if oldest_first:
                del_range = range(delay_long, delay_short - 1, -1)
            else:
                del_range = range(delay_short, delay_long + 1)

            for delay in del_range:
                assert self.t_mod < self.max_delay
                # t_mod is where most recent data point is stored

                time_index = self.t_mod - delay

                if time_index < 0:
                    time_index = self.max_delay + time_index

                state = self.state_arrays_list[state_index][time_index, :]
                state_seq.append(state)
            return np.array(state_seq)
        else:
            return None

    def get_newest_states_list(self):
        if self.active:
            newest_states_list = []

            for k in range(len(self.state_arrays_list)):
                newest_states_list.append(self.get_state(state_index=k, delay=0))

            return newest_states_list
        else:
            return None



if __name__ == '__main__':
    tmp = StatesLimitedHistory(params={'max_delay': 10,
                                       'states_dim_list': [3],
                                       'store_extra_data': False})

    np.random.seed(0)
    print()
    for t in range(20):
        rand_data = np.random.random(3)

        print('t', t, 'data', rand_data)
        tmp.store_new_states(newest_states_list=[rand_data])

    print()
    tmp_data = tmp.get_state_sequence(state_index=0, delay_long=9, delay_short=0, oldest_first=False)
    print(tmp_data)

    print()
    tmp_data = tmp.get_state(state_index=0, delay=0)
    print(tmp_data)
    print()
    tmp_data = tmp.get_state(state_index=0, delay=9)
    print(tmp_data)






