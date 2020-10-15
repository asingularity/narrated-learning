import cv2
from os import listdir
from os.path import isfile, join
import subprocess
import random
random.seed(0)
from datetime import datetime, date, time
import math
import numpy as np
import sys
np.set_printoptions(suppress=True, precision=4, threshold=sys.maxsize)
import pickle
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from math import log
from matplotlib.pyplot import cm


class DataSource(object):
    def __init__(self, params):
        '''
        params:
         'file_type': 'twa_bid,ask,bid,twa_ask,volume,bar_size,date_time',
         'names': 'AAPL,MSFT,GOOG,FB,BABA,INTC,ORCL,CSCO',
         'force_reload': False,
         'data_folder': DATA_FOLDER
        '''

        self.file_type = params['file_type']
        self.names = params['names']
        self.force_reload = params['force_reload']
        self.data_folder = params['data_folder']
        self.source_data_file_paths_list = []

    def get_source_data_file_paths_list(self):
        return self.source_data_file_paths_list

    def load_data_from_ib_csv(self):

        data_folder = self.data_folder
        data_groups_str = [self.names]
        force_reload = self.force_reload
        file_type = self.file_type

        print ('data_folder ', data_folder)
        data_groups = [data_group_str.split(',') for data_group_str in data_groups_str]

        all_data_groups_str = ''
        for data_group_str in data_groups_str:
            all_data_groups_str = all_data_groups_str + '---' + data_group_str

        data_tables_pkl_filename = data_folder + '/data_tables_' + str(all_data_groups_str) + '.pkl'
        data_completeness_pkl_filename = data_folder + '/data_completeness_' + str(all_data_groups_str) + '.pkl'
        day_strings_pkl_filename = data_folder + '/data_day_strings_' + str(all_data_groups_str) + '.pkl'

        self.source_data_file_paths_list.append(data_tables_pkl_filename)
        self.source_data_file_paths_list.append(data_completeness_pkl_filename)
        self.source_data_file_paths_list.append(day_strings_pkl_filename)

        self.short_filename = str(all_data_groups_str)

        try:
            if force_reload:
                assert False
            print()
            print(data_tables_pkl_filename)
            print()
            f = open(data_tables_pkl_filename, 'rb')

            data_tables = pickle.load(f, encoding='latin1')
            f.close()

            f = open(data_completeness_pkl_filename, 'rb')
            data_completeness_list = pickle.load(f, encoding='latin1')
            f.close()

            f = open(day_strings_pkl_filename, 'rb')
            day_strings_for_day_num = pickle.load(f, encoding='latin1')
            f.close()

            print ('loaded from file: ', len(data_tables), data_tables[0][0].shape, data_tables[0][1].shape, len(data_completeness_list), data_completeness_list[0].shape)
            self.data_tables = data_tables
            self.data_completeness_list = data_completeness_list
            self.day_strings_for_day_num = day_strings_for_day_num

        except:
            print (all_data_groups_str)
            print (data_groups, len(data_groups))
            num_data_groups = len(data_groups)
            file_info_list = []
            index_for_symbol = []

            for k in range(len(data_groups)):
                file_info_list.append([])

                symbol_dict = {}

                symbol_index = 0
                for symbol in data_groups[k]:
                    symbol_dict[symbol] = symbol_index
                    symbol_index += 1

                index_for_symbol.append(symbol_dict)

            print (file_info_list)
            print (index_for_symbol)

            onlyfiles = [f for f in listdir(data_folder) if isfile(join(data_folder, f)) and len(f) > len('historical_data_')]
            onlyfiles = [f for f in onlyfiles if f[0:16]=='historical_data_']

            print ([data_name_string.strip('\n') for data_name_string in file_type.split(',')])

            min_time = datetime.strptime('9999-11-09 08:40:25', "%Y-%m-%d %H:%M:%S")
            max_time = datetime.strptime('1111-11-09 08:40:25', "%Y-%m-%d %H:%M:%S")

            files = []
            for filename in onlyfiles:
                a = subprocess.check_output(['wc', '-l', data_folder + '/' + filename])
                wc = a.split(' ')[0]
                if int(wc) > 1:
                    # get first line
                    # determine if matches file type (from below)

                    with open(data_folder + '/' + filename, 'r') as f:
                        first_line = f.readline()
                        if first_line[0:len(file_type)] == file_type:
                            files.append(filename)

                            second_line = f.readline()
                            data_line = second_line.split(',')
                            date_time_str = data_line[6].strip('\n')
                            dt = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")

                            if dt < min_time:
                                min_time = dt

                            symbol = filename.split('_')[2]
                            for g in range(len(data_groups)):
                                if symbol in data_groups[g]:
                                    file_info_list[g].append([symbol, int(math.ceil((float(wc) - 1.0) * 1.0 / 23400)), ' days, starting: ', str(dt)])

                            for line in f:
                                pass
                            last = line
                            data_line = last.split(',')

                            #print 'first line ', first_line
                            #print 'second line ', second_line
                            date_time_str = data_line[6].strip('\n')
                            dt = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")

                            if dt > max_time:
                                max_time = dt

            print ('initial date: ', min_time, max_time)
            print ('file info: ')
            for g in range(len(file_info_list)):
                print ('>>> GROUP: ', g)
                for k in range(len(file_info_list[g])):
                    print (file_info_list[g][k])

            reftime = min_time

            days_total = (max_time.date() - min_time.date()).days + 1
            print ('total days: ', days_total)

            data_tables = []
            data_completeness_list = []
            for g in range(len(data_groups)):
                num_symbols = len(data_groups[g])
                #data_table = np.zeros((num_symbols, days_total, 23400))
                data_table_bid = np.zeros((num_symbols, days_total * 23400))
                data_table_ask = np.zeros((num_symbols, days_total * 23400))
                data_completeness = np.zeros((num_symbols, days_total))
                data_tables.append([data_table_bid, data_table_ask])
                data_completeness_list.append(data_completeness)
            day_start = datetime.strptime('2015-1-11 06:30:00', "%Y-%m-%d %H:%M:%S")

            day_strings_for_day_num = days_total * ['----------']

            for filename in files:
                print ('loading: ', filename)
                symbol = filename.split('_')[2]

                data_group = None
                for g in range(len(data_groups)):
                    if symbol in data_groups[g]:
                        data_group = g

                if data_group is not None:
                    f = open(data_folder + '/' + filename, 'r')
                    line_num = 0
                    for line in f:
                        if line_num > 0:
                            data_line = line.split(',')
                            if file_type == 'open,high,low,close,volume,bar_size,date_time':
                                bid = float(data_line[3])
                                ask = bid
                            else:
                                if file_type == 'twa_bid,ask,bid,twa_ask,volume,bar_size,date_time':
                                    bid = float(data_line[2])
                                    ask = float(data_line[1])
                                else:
                                    assert False, 'ERROR: Bad file_type: ' + str(file_type)

                            date_time_str = data_line[6].strip('\n')
                            t1 = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S")
                            day_num = (t1.date() - reftime.date()).days
                            day_strings_for_day_num[day_num] = date_time_str[0:10]
                            second_num = (t1 - day_start).seconds
                            symbol_index = index_for_symbol[data_group][symbol]

                            if data_tables[data_group][0][symbol_index, day_num * 23400 + second_num] == 0.0:
                                data_completeness_list[data_group][symbol_index, day_num] = data_completeness_list[data_group][symbol_index, day_num] + 1
                            data_tables[data_group][0][symbol_index, day_num * 23400 + second_num] = bid
                            data_tables[data_group][1][symbol_index, day_num * 23400 + second_num] = ask
                        line_num += 1

                #datas[sec_name] = data

            print ('data_completeness:')

            data_completeness_all = data_completeness_list[0].astype(np.int)

            for g in range(1, len(data_groups)):

                # overwriting! concatenate tables for groups, then save. also, use %2d and 0 or 1 for complete data (threshold=20k?)
                data_completeness_all = np.concatenate((data_completeness_all, data_completeness_list[g].astype(np.int)))
            #np.savetxt(data_folder + '/data_completeness_txt_' + str(all_data_groups_str), data_completeness_list[g].astype(np.int), '%5d')

            all_day_strings = ''
            for day_string in day_strings_for_day_num:
                all_day_strings = all_day_strings + ' ' + day_string

            np.savetxt(data_folder + '/data_completeness_txt_' + str(all_data_groups_str), data_completeness_all, '%10d', header=all_day_strings)

            #data_tables_pkl_filename = data_folder + '/data_tables_' + str(all_data_groups_str) + '.pkl'
            #data_completeness_pkl_filename = data_folder + '/data_completeness_' + str(all_data_groups_str) + '.pkl'

            f = open(data_tables_pkl_filename, 'w')
            pickle.dump(data_tables, f)
            f.close()

            f = open(data_completeness_pkl_filename, 'w')
            pickle.dump(data_completeness_list, f)
            f.close()

            f = open(day_strings_pkl_filename, 'w')
            pickle.dump(day_strings_for_day_num, f)
            f.close()

            self.data_tables = data_tables
            self.data_completeness_list = data_completeness_list
            self.day_strings_for_day_num = day_strings_for_day_num


        # take out zero days (weekends, holidays)

        data_cmp = self.data_completeness_list[0]

        all_data_bid = self.data_tables[0][0]
        all_data_ask = self.data_tables[0][1]

        good_indices = []

        for day_num in range(0, data_cmp.shape[1]):
            #print data_cmp[:, day_num]
            bad_day = False
            # Take out half-days and empty days!
            for atemp in range(0, data_cmp.shape[0]):
                if data_cmp[atemp, day_num] < 23300:
                    #print 'HERE!!!'
                #if data_cmp[0, day_num] == 0:
                    bad_day = True
            if bad_day:
                pass
            else:
                #print data_cmp[0, day_num]
                good_indices.extend(range(day_num * 23400, (day_num + 1) * 23400))

        all_data_bid = all_data_bid[:, good_indices]
        all_data_ask = all_data_ask[:, good_indices]
        num_weird_replacements = 0

        #steps_replaced = []
        for step in range(1, all_data_ask.shape[1] - 1):
            for atemp in range(0, data_cmp.shape[0]):
                if all_data_ask[atemp, step] == 0.0:
                    if all_data_ask[atemp, step - 1] > 0.0 and all_data_ask[atemp, step + 1] > 0.0: #(all_data_ask[atemp, step + 2] > 0.0 or all_data_ask[atemp, step + 1] > 0.0):
                        #assert all_data_bid[0, step - 1] > 0.0 and all_data_bid[0, step + 1] > 0.0
                        all_data_ask[atemp, step] = all_data_ask[atemp, step - 1]
                        all_data_bid[atemp, step] = all_data_bid[atemp, step - 1]
                        num_weird_replacements += 1

        print ('num_weird_replacements: ', num_weird_replacements)
        # zero times become nans

        data_bad_indices = np.nonzero(all_data_bid == 0)
        #print data_bad_indices
        #print len(data_bad_indices[0])
        assert len(data_bad_indices[0])==0, 'BAD DATA LEFT!'
        #print '*'
        #print data_bad_indices[0]
        #print data_bad_indices[1]
        #min_bad_t = np.amin(data_bad_indices[1])
        #max_bad_t = np.amax(data_bad_indices[1])
        #len_bad = len(data_bad_indices[1])
        #print min_bad_t / 23400, max_bad_t / 23400, len_bad


        # DO NOT COMMIT THIS- UNCOMMENT INSTEAD
        all_data_bid[np.nonzero(all_data_bid == 0)] = np.nan
        all_data_ask[np.nonzero(all_data_ask == 0)] = np.nan


        self.data_tables[0][0] = all_data_bid
        self.data_tables[0][1] = all_data_ask

    def save_to_mat(self, folder):
        from scipy.io import savemat

        savemat(folder + '/data_' + self.short_filename + '_bid.mat', mdict={'data_bid': self.get_all_data_bid()})
        savemat(folder + '/data_' + self.short_filename + '_ask.mat', mdict={'data_ask': self.get_all_data_ask()})

    def get_all_data_bid(self, group=0):
        return self.data_tables[group][0]

    def get_all_data_ask(self, group=0):
        return self.data_tables[group][1]

    def plot_data(self, folder):

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        g = 0

        data_table_bid = self.data_tables[g][0]  # .copy()
        data_table_ask = self.data_tables[g][1]  # .copy()
        prefix = self.day_strings_for_day_num[0] + '_' + self.day_strings_for_day_num[len(self.day_strings_for_day_num) - 1] + '_' + self.names

        fig1 = plt.figure(figsize=(50, 10))

        for r in range(data_table_bid.shape[0]):
            ax1 = fig1.add_subplot(data_table_bid.shape[0], 1, r+1)
            #if data_table_bid[r, 0] > 700:
            #    print '700 being plotted at row: ', r

            if np.sum(data_table_bid[r, :]==0) > 0:
                print ('invalid row! ', r)
                print (data_table_bid[r, 0])

            ax1.plot(data_table_bid[r, :], 'b-')
            ax1.plot(data_table_ask[r, :], 'r-')
            for k in range(int(data_table_bid.shape[1] / 23400)):
                ax1.axvline(x=(k + 1) * 23400, color='g')

        fig1.savefig(folder + '/data_' + prefix + '.png', dpi=100)
        plt.close('all')


class StockDataSensor(object):
    def __init__(self, params):

        self.fig = plt.figure(figsize=(40, 20))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.cla()
        self.ax.get_xaxis().get_major_formatter().set_scientific(False)
        self.ax.get_yaxis().get_major_formatter().set_scientific(False)


        self.image_dim = params['image_dim']
        self.return_type = params['return_type']

        # TODO make parameter
        # for initializing arrays
        self.max_time = 2000000

        # TODO make parameter
        self.override_with_random_walk = True

        self.DATA_FOLDER = '/srv/projects/financial/data'
        self.TEMP_FOLDER = '/srv/projects/financial/temp'

        ds_params = {'file_type': 'twa_bid,ask,bid,twa_ask,volume,bar_size,date_time',
                     'names': 'AAPL,MSFT,GOOG,FB,BABA,INTC,ORCL,CSCO',
                     'force_reload': False,
                     'data_folder': self.DATA_FOLDER}

        data_source = DataSource(ds_params)
        print()
        print('STARTING: data_source.load_data_from_ib_csv()')
        print()
        data_source.load_data_from_ib_csv()
        print()
        print('STARTING: data_source.plot_data(TEMP_FOLDER)')
        print()

        plot_on = False
        if plot_on:
            data_source.plot_data(self.TEMP_FOLDER)

        #data_source.save_to_mat(TEMP_FOLDER)
        print()
        print (data_source.get_all_data_bid().shape)  # (8, 1801800)

        self.dat = data_source.get_all_data_bid()

        if self.override_with_random_walk:
            self.dat = self._create_random_walk_like(real_dat=self.dat.copy())

        self.num_stocks = self.dat.shape[0]

        # TODO CLEANUP ******** HACK ********
        self.num_act_per_rf = None
        self.probs_per_rf_stock = None
        self.sum_per_rf_stock = None
        self.num_rf = None


        self.all_ratios = None

        self.t = 0


    def _create_random_walk_like(self, real_dat):
        print()
        print('creating random walk...')
        random_dat = np.zeros_like(real_dat)

        random_dat[:, 0] = np.random.random(real_dat.shape[0]) * 100

        for t in range(1, real_dat.shape[1]):
            new_dat = random_dat[:, t - 1] + np.random.random(real_dat.shape[0]) * 0.01 - 0.005
            new_dat[new_dat < 1] = 1

            random_dat[:, t] = new_dat

        print()
        print('plotting random walk...')
        # plot the random walk data for all stocks
        colors = ['r', 'b', 'g', 'k']
        ck = 0
        self.ax.cla()
        for k in range(random_dat.shape[0]):
            self.ax.plot(random_dat[k, 0:real_dat.shape[1]], c=colors[ck], marker='.')
            ck += 1
            if ck == 4:
                ck = 0

        self.fig.savefig("random_walk_data" + ".png", dpi=100)

        print()
        print('done random walk.')

        return random_dat

    def read_input(self, last_activities):

        if self.t > self.dat.shape[1] - 1:
            print()
            print('OUT OF DATA')
            print(self.t, self.dat.shape[1])
            print()
            exit(1)

        self.delta_t = 320 * 2

        t_in_day = self.t % 23400

        # assumed given timescale
        max_change = 0.005

        #if self.t == 0:
        #    self.t = self.delta_t + self.image_dim + 1

        if t_in_day == 0:
            self.t += self.delta_t + self.image_dim + 1
            t_in_day = self.t % 23400

        # if self.t > self.delta_t + self.image_dim:
        if t_in_day > self.delta_t + self.image_dim:

            # preprocessing
            #   make sure scaled to [0, 1]
            #   make sure square image of image_dim

            dat_now = self.dat[:, self.t - self.image_dim:self.t]
            dat_past = self.dat[:, self.t - self.image_dim-self.delta_t:self.t-self.delta_t]
            dat = np.divide(dat_now, dat_past) - 1.0  # [-0.01--, 0.01++]

            # scale to [0, 1]
            dat[dat < -max_change] = -max_change
            dat[dat > max_change] = max_change
            dat = (dat + max_change) * 1.0 / (2 * max_change)
            assert self.image_dim / dat.shape[0] == int(self.image_dim / dat.shape[0])

            im = np.repeat(dat, int(self.image_dim / dat.shape[0]), axis=0)

            #print(np.amin(dat), np.amax(dat))

            #print(dat.shape)

            #im = 0.5 * np.ones((self.image_dim, self.image_dim), self.return_type)

            if last_activities is not None:
                hl_0_act = last_activities[0]

                if self.num_act_per_rf is None:

                    self.num_rf = hl_0_act.shape[0]

                    self.probs_per_rf_stock = np.ones((self.num_rf, self.num_stocks))
                    self.sum_per_rf_stock = np.zeros((self.num_rf, self.num_stocks))

                    self.num_act_per_rf = np.zeros(self.num_rf)

                    self.all_ratios = np.ones((self.num_rf, self.num_stocks, self.max_time))

                ratios_future = np.divide(self.dat[:, self.t + self.delta_t * 1], self.dat[:, self.t])

                act_indices = np.nonzero(hl_0_act)[0]

                self.num_act_per_rf[act_indices] += 1
                self.sum_per_rf_stock[act_indices, :] += ratios_future
                self.probs_per_rf_stock[act_indices, :] *= ratios_future

                self.all_ratios[:, :, self.t] = self.probs_per_rf_stock[:, :]

        else:
            assert False
            im = 0.5 * np.ones((self.image_dim, self.image_dim), self.return_type)

        self.t += 1

        return im

    def print_ratios(self):

        if self.num_rf is not None:
            print()
            print('RATIOS')
            #print('COMPUTING MEAN')
            #print('sum per rf per stock')
            # print(self.sum_per_rf_stock)
            mean_ratios_per_rf = np.divide(self.sum_per_rf_stock, self.num_act_per_rf[:, np.newaxis])

            print()
            print('mean per rf:')
            print(mean_ratios_per_rf)
            print()
            # print('prob per rf:')
            # print(self.probs_per_rf_stock)
            # print()
            # print('num per rf:')
            # print(self.num_act_per_rf)
            # print()

            self.ax.cla()
            color = iter(cm.rainbow(np.linspace(0, 1, self.num_rf * self.num_stocks)))
            #print('AFUAFASFASASF')
            #print(np.linspace(0, 1, self.num_rf * self.num_stocks))

            colors = ['r', 'g', 'b', 'k']
            k = 0

            plot_all = True

            if plot_all:
                for rf in range(self.num_rf):
                    for st in range(self.num_stocks):
                        c = next(color)
                        #print(c)
                        self.ax.plot(self.all_ratios[rf, st, 0:self.t], c=colors[k], marker='.')

                        k += 1
                        if k > 3:
                            k = 0
            else:
                highest = np.unravel_index(np.argmax(self.all_ratios[:, :, self.t - 1]), (self.num_rf, self.num_stocks))
                print(highest)
                self.ax.plot(self.all_ratios[highest[0], highest[1], 0:self.t], c=colors[k], marker='.')

                k += 1
                highest = np.unravel_index(np.argmin(self.all_ratios[:, :, self.t - 1]), (self.num_rf, self.num_stocks))
                print(highest)
                self.ax.plot(self.all_ratios[highest[0], highest[1], 0:self.t], c=colors[k], marker='.')

            self.fig.savefig("all_prods" + ".png", dpi=100)

def main():
    sensor = StockDataSensor(params={
        'image_dim': 64,
        'return_type': np.float32
    })

    while True:
        cv2.imshow('im', sensor.read_input())
        cv2.waitKey(10)  # 300)


if __name__ == '__main__':
    main()
