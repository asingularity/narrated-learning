




import numpy as np
import cv2


def _bin_pixels_expand_columns(arr, num_bins_per_pixel):
    '''

    keep number of rows, expand in number of columns
    construct binary image: (0.0 or 1.0) per bin

    https://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy
        use: digitize, or histogram

    https://het.as.utexas.edu/HET/Software/Numpy/reference/generated/numpy.digitize.html
        digitize: returns index of bin to which each pixel belongs
        expanded array will have a set of bins per pixel, in the expanded columns

    :param arr:
    :return:
    '''
    # print('***')
    # print('arr', arr.shape)

    bins = np.linspace(0, 1 + 1e-9, num_bins_per_pixel + 1)  # TODO to fix binning add 1e-9 to 1 in second argument!
    # print('bins', bins.shape, bins)
    bin_indices = np.digitize(arr, bins) - 1  # same shape as arr; which bin, per pixel
    # print('bin_indices', bin_indices.shape, bin_indices)
    if self.use_negative_input:
        arr_exp = -np.ones((arr.shape[0], arr.shape[1] * num_bins_per_pixel))
    else:
        arr_exp = np.zeros((arr.shape[0], arr.shape[1] * num_bins_per_pixel))

    # print('arr_exp', arr_exp.shape)

    # term1 = np.tile(self.num_bins_per_pixel * np.arange(arr.shape[1]), 2)  # can't remember why this is np.tile(..., 2)
    term1 = num_bins_per_pixel * np.arange(arr.shape[1])  # only works if arr rows is 1?
    term2 = bin_indices.flatten()
    # print('term1', term1.shape, term1)
    # print('term2', term2.shape, term2)
    c = term1 + term2
    r = np.repeat(np.arange(arr.shape[0]), arr.shape[1])

    arr_exp[r, c] = 1.0

    return arr_exp


def _collapse_binned_columns_to_pixels(arr_exp, num_bins_per_pixel):
    '''

    non-trivial: arr_exp may no longer be binary; need to figure out how to collapse multiple values of different weight to one pixel:
        weighted average? take max value?

    :param arr_exp:
    :return:
    '''

    num_pixels = int((arr_exp.shape[1] / num_bins_per_pixel))
    num_rows = arr_exp.shape[0]

    # reshape so we can take max along one dim
    #   ie each row in this matrix is a pixel, temporarily
    # then reshape back

    tmp = arr_exp.reshape((num_rows * num_pixels, num_bins_per_pixel))

    bins = np.linspace(0, 1 + 1e-9, num_bins_per_pixel + 1)

    # max val for display:
    max_index = np.argmax(tmp, axis=1)
    max_vals = bins[max_index]

    weights = tmp[np.arange(tmp.shape[0]), max_index]

    # print(max_vals.shape, weights.shape)

    # weighted mean val for display:
    # tmp1 = np.multiply(tmp, bins[np.newaxis, 0:num_bins_per_pixel])
    # tmp2 = np.sum(tmp1, axis=1)  # 65536
    # tmp3 = np.divide(tmp2, np.sum(tmp, axis=1))
    # max_vals = tmp3

    weights = weights.reshape(num_rows, num_pixels)
    arr = max_vals.reshape(num_rows, num_pixels)

    return arr, weights


def make_im(input_arrays):

    num_bins_per_pixel = 6
    input_im_dim = 16

    tmp_im_all = None
    tmp_im = None

    arr, weights = _collapse_binned_columns_to_pixels(arr_exp=input_arrays,
                                                      num_bins_per_pixel=6)

    for r_tmp in range(weights.shape[0]):  # per rf

        im0 = arr[r_tmp, :].reshape((input_im_dim, input_im_dim))  # rf_dim
        im1 = weights[r_tmp, :].reshape((input_im_dim, input_im_dim))

        # im1 = 0.5 * (im1 + 1)

        spacer1 = 0.5 * np.ones((im0.shape[0], 3))

        tmp2 = np.hstack((im0, spacer1, im1))

        if tmp_im is None:
            tmp_im = tmp2.copy()
        else:
            tmp3 = 0.2 * np.ones((2, tmp_im.shape[1]))
            tmp_im = np.vstack((tmp_im, tmp3, tmp2))

        if r_tmp > 0 and (r_tmp + 1) % 20 == 0:
            if tmp_im_all is None:
                tmp_im_all = tmp_im.copy()
            else:
                spacer = 0.5 * np.ones((tmp_im_all.shape[0], 3))
                tmp_im_all = np.hstack((tmp_im_all, spacer, tmp_im))

            tmp_im = None

    tmp_all_im = tmp_im_all

    max_dim = max(tmp_all_im.shape[0], tmp_all_im.shape[1])
    imscale = 1600 / max_dim  # 0.2: full table, 2.0
    tmp_im = cv2.resize(tmp_all_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

    return tmp_im


def get_sparse_features_subtractive(arr):
    '''

    does not work, just subtracts average value

    :param arr:
    :return:
    '''

    sparse_arr = np.zeros_like(arr)
    sum_rows = np.sum(arr, axis=0)

    for k in range(arr.shape[0]):
        new_row = 2 * arr[k, :] - sum_rows * 1.0 / arr.shape[0]
        new_row = (new_row - np.amin(new_row)) * 1.0 / (np.amax(new_row) - np.amin(new_row))
        # print(np.amin(new_row), np.amax(new_row))
        sparse_arr[k, :] = new_row[:]

    return sparse_arr


def _get_max_bin_index_and_joint_prob(remainder, cumulative_pattern_indices, bin_layer):
    '''

    find max bin and its joint prob based with cumulative pattern being present in remainder, excluding any bins in cumulative pattern already
    note that cumulative_pattern_indices are only valid up to index: bin_layer-1, in input arguments below:

    :param remainder:
    :param cumulative_pattern_indices:
    :param bin_layer:
    :return:
    '''

    #print()
    #print('remainder.shape', remainder.shape)
    #print('cumulative_pattern_indices[0:bin_layer]', cumulative_pattern_indices[0:bin_layer].astype(np.int))
    #print()

    # (1) find all rows of remainder where current cumulative pattern is present
    #       current cumulative pattern size: <bin_layer>
    #           bin_layer 0: pattern size 0
    #           bin_layer 1: pattern size 1

    cumulative_pattern_exp = np.zeros(remainder.shape[1])
    if bin_layer > 0:

        cumulative_pattern_exp[cumulative_pattern_indices[0:bin_layer].astype(np.int)] = 1
        match_sum = np.sum(np.multiply(remainder, cumulative_pattern_exp), axis=1)
        prev_matched_rows = np.nonzero(np.equal(match_sum, bin_layer))[0]
    else:
        #pass
        # all rows are candidate rows, since all rows matched
        prev_matched_rows = np.arange(remainder.shape[0])

    # (2) zero out current cumulative pattern from those rows, in a remainder copy
    rem_tmp = remainder.copy()

    if bin_layer > 0:
        rem_tmp = rem_tmp[prev_matched_rows, :]
        rem_tmp[:, cumulative_pattern_indices[0:bin_layer].astype(np.int)] = 0

    # (3) find max prob bin in remainder copy after zeroing out above
    sum_bins = np.sum(rem_tmp, axis=0)
    max_bin_index = np.argmax(sum_bins)

    occurences = np.amax(sum_bins)

    #if bin_layer > 0:
    prob_cumulative_pattern = len(prev_matched_rows) / remainder.shape[0]
    prob_max_bin_given_pattern = occurences / len(prev_matched_rows)

    joint_prob = prob_cumulative_pattern * prob_max_bin_given_pattern

    return max_bin_index, joint_prob

def _get_best_pattern(remainder, rf_index_for_debug=None):
    '''

    perfect method: i.e. binary pattern must be wholly contained in the input sample

    :param remainder:
    :return:
    '''

    num_bins_per_pixel = 6
    input_im_dim = 16

    feature_len = remainder.shape[1]

    assert feature_len == num_bins_per_pixel * input_im_dim * input_im_dim

    # find max prob bins, and cumulative prob at each additional bin
    # cutoff at max integral (cumulative prob * cumulative pattern size)

    max_pattern_bins = input_im_dim * input_im_dim
    cumulative_pattern_indices = np.zeros(max_pattern_bins)
    cumulative_prob = np.zeros(max_pattern_bins)
    time_integrals = np.zeros(max_pattern_bins)

    last_time_integral = -1

    for bin_layer in range(max_pattern_bins):
        # find max bin and its joint prob based with cumulative pattern being present in remainder, excluding any bins in cumulative pattern already
        # note that cumulative_pattern_indices are only valid up to index: bin_layer-1, in input arguments below:
        max_bin_index, joint_prob = _get_max_bin_index_and_joint_prob(remainder=remainder, cumulative_pattern_indices=cumulative_pattern_indices, bin_layer=bin_layer)

        #if rf_index_for_debug == 51:
        #    print(bin_layer, joint_prob, (bin_layer + 1) * joint_prob)

        if np.isnan(joint_prob):  #  or bin_layer == 0:  FIX HACK! bin_layer==0 returns higher values...
            time_integral = 0
        else:
            time_integral = (bin_layer + 1) * joint_prob

        #if time_integral < last_time_integral:
        #    # THIS IS WRONG!!! actually this function can go up and down, need to look at max over all samples!!!
        #    break

        cumulative_pattern_indices[bin_layer] = max_bin_index
        cumulative_prob[bin_layer] = joint_prob
        time_integrals[bin_layer] = time_integral

        last_time_integral = time_integral

    # TODO is this the right bin? off by one? verify
    #print(time_integrals)
    bin_layer = np.argmax(time_integrals)

    # TODO is this the right bin? off by one? verify
    print("    best pattern size: ", bin_layer, ', occurence prob: ', cumulative_prob[bin_layer], ', num occurences: ', int(cumulative_prob[bin_layer] * remainder.shape[0]))
    best_pattern = cumulative_pattern_indices[0:(bin_layer + 1)]
    return best_pattern


def get_sparse_features(arr):

    num_rfs = 200
    feature_len = arr.shape[1]

    remainder = arr.copy()
    sparse_arr = np.zeros((num_rfs, feature_len))

    for k in range(num_rfs):
        print()
        print('Finding best pattern for RF: ', k)
        # get "best pattern" from remainder using iterative greedy max prob
        best_pattern = _get_best_pattern(remainder, rf_index_for_debug=k)
        best_pattern_exp = np.zeros(remainder.shape[1])
        best_pattern_exp[best_pattern.astype(np.int)] = 1

        # set "best pattern" as this row of sparse arr
        sparse_arr[k, :] = best_pattern_exp[:]

        # TODO subtract "best pattern" from remainder where present
        match_sum = np.sum(np.multiply(remainder, best_pattern_exp), axis=1)
        prev_matched_rows = np.nonzero(np.equal(match_sum, np.sum(best_pattern_exp)))[0]
        for matched_row in list(prev_matched_rows):
            remainder[matched_row, best_pattern.astype(np.int)] = 0
        #print(prev_matched_rows)

        if k%10 == 0:
            im = make_im(sparse_arr)
            cv2.imshow('RFs', im)
        cv2.waitKey(1)

    return sparse_arr

def main():
    print()
    print('loading file...')
    print()

    arr = np.loadtxt('table_i_800.txt')
    print('table_i shape:', arr.shape)
    print()

    sparse_arr = get_sparse_features(arr=arr)

    im_orig = make_im(arr)
    im = make_im(sparse_arr)

    cv2.imshow('orig', im_orig)
    cv2.imshow('RFs', im)
    cv2.waitKey(0)



if __name__ == '__main__':
    main()