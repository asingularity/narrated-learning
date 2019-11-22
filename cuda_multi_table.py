
import numpy as np
from cuda_dist_query import CudaTable


class DumbCudaMultiTable(object):
    '''
        For speed test purposes: dumb implementation of multiple tables
    '''

    def __init__(self, params):
        '''

        '''

        self.num_entries_arr = np.array(params['num_entries_list'])
        self.input_dim_arr = np.array(params['input_dim_list'])

        # TODO for now, we assume they must all be same dim for one collection!
        assert np.unique(self.input_dim_arr).shape[0] == 1

        self.num_tables = self.num_entries_arr.shape[0]
        self.N = self.num_tables

        self.tables_list = []
        for k in range(self.N):
            self.tables_list.append(CudaTable(num_entries=self.num_entries_arr[k],
                                              input_dim=self.input_dim_arr[k]))

    def get_post_init_done(self):
        '''

        :return: N array of bool: true/false that init is done per table
        '''

    def post_init(self, table_indices=None):
        '''

        do post init on selected table indices; if None, then all

        :param table_indices: list
        :return:
        '''

        if table_indices is None:
            tmp_indices = range(self.N)
        else:
            tmp_indices = table_indices

        for k in tmp_indices:
            self.tables_list[k].post_init()

    def get_num_rows(self):
        '''

        :return: N list of [num rows per table]
        '''

        tmp = []
        for k in range(self.N):
            tmp.append(self.tables_list[k].get_num_rows())

        return tmp

    def get_min_dists(self):
        '''

        :return: N list of [table_min_dist, table_min_dist_r, table_min_dist_c]
        '''

        tmp = []
        for k in range(self.N):
            tmp.append(self.tables_list[k].get_min_dists())
        return None

    def set_matrix_rows(self, table_indices, row_indices, row_inputs, rows_to_table_dists, fast_init):
        '''

        set matrix rows: set 1 row or no rows, per table

        :param table_indices: L list (L<=N) of indices of tables for which we are setting a row (only one row)
        :param row_indices: L list of index to set per table
        :param row_inputs: L list of row_input arrays
        :param rows_to_table_dists: L list of dists arrays
        :return:
        '''

        for k in table_indices:
            self.tables_list[table_indices[k]].set_matrix_row(row_index=row_indices[k],
                                                              row_input=row_inputs[k],
                                                              row_to_table_dists=rows_to_table_dists[k],
                                                              fast_init=fast_init[k])

    def query(self, query_inputs):
        '''

        :param query_inputs:
        :return:
        '''

        tmp = []
        for k in range(self.N):
            tmp.append(self.tables_list[k].query(query_input=query_inputs[k]))

        return tmp


class CudaMultiTable(object):
    def __init__(self, params):
        '''

        '''

        self.num_entries_arr = np.array(params['num_entries_list'])
        self.input_dim_arr = np.array(params['input_dim_list'])

        # TODO for now, we assume they must all be same dim for one collection!
        assert np.unique(self.input_dim_arr).shape[0] == 1

        # also for num entries for now
        assert np.unique(self.num_entries_arr).shape[0] == 1
        self.num_entries_per_table = self.num_entries_arr[0]

        self.num_tables = self.num_entries_arr.shape[0]
        self.N = self.num_tables

        total_entries = np.sum(self.num_entries_arr)

        self.table = CudaTable(num_entries=total_entries,
                               input_dim=self.input_dim_arr[0])

    def get_post_init_done(self):
        '''

        :return: N array of bool: true/false that init is done per table
        '''

    def post_init(self, table_indices=None):
        '''

        do post init on selected table indices; if None, then all

        :param table_indices: list
        :return:
        '''

        # TODO this assumes table_indices cannot be used... if all one table

        assert table_indices is None
        self.table.post_init()

    def get_num_rows(self):
        '''

        :return: N list of [num rows per table]
        '''

        return list(self.num_entries_arr)


    def get_min_dists(self):
        '''

        :return: N list of [table_min_dist, table_min_dist_r, table_min_dist_c]
        '''

        # TODO how to do this??

    def set_matrix_rows(self, table_indices, row_indices, row_inputs, rows_to_table_dists, fast_init):
        '''

        set matrix rows: set 1 row or no rows, per table

        :param table_indices: L list (L<=N) of indices of tables for which we are setting a row (only one row)
        :param row_indices: L list of index to set per table
        :param row_inputs: L list of row_input arrays
        :param rows_to_table_dists: L list of dists arrays
        :return:
        '''

        # TODO - for now, assume fast_init all the same!

        # TODO - convert from table_indices to local indices for single table

        row_indices_total = np.array(table_indices) * self.num_entries_per_table
        row_indices_total = row_indices_total + row_indices

        # TODO inside this function -> inside self.d.set_row_dists -> use same for-loop for multiple rows, if possible
        self.table.set_multiple_rows(row_indices=row_indices_total,
                                     row_inputs=row_inputs,
                                     rows_to_table_dists=rows_to_table_dists,
                                     fast_init=fast_init[0])

    def query(self, query_inputs):
        '''

        :param query_inputs:
        :return:
        '''

        dists_tmp = self.table.query_multiple_rows(query_inputs=query_inputs)
        # what does this return?

        #print('WTF')
        #print()
        #print(len(dists_tmp))
        #print()
        #print(dists_tmp[0].shape, dists_tmp[1].shape)  # (1600,) (1600,)
        #print()

        new_dists_list = []
        for k in range(self.num_tables):
            new_dists_list.append(dists_tmp[k][k * self.num_entries_per_table:(k + 1) * self.num_entries_per_table])

        # TODO now only take appropriate subset of above dists, since currently getting whole thing

        return new_dists_list

        # TODO inside CudaTable:
        #   make a query_multi_rows options
        #   use (skcuda) linalg.dot on all rows with matrix at once as one dot product
        #   i.e. X_gpu is m * dim now, not 1 * dim
        #   -- !!! this is pretty wasteful, as lots of computation is just being disgarded, i.e. most of the table with most of the query_inputs rows
        #   and table is full (all) tables concatenated
        #       TODO alternative to this: another "incorrect" shortcut if needed is to assume that all tables are the same table...
        #       i.e. that you only have "one" table for a given scale, just do multiple queries on it at once..
        #       we try to avoid this for now...
        #       i.e. this is the "dumb" shortcut, would win a lot right now.
        #       remaining question would be how to update if multiple tiles want to update the table on the same timestep
        #           would have to sequence this... or combine if different rows are being set... could optimize
        #       still separate W prediction matrices per tile: what happens at edges of image could be quite different from center























































