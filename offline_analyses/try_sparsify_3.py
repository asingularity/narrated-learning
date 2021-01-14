




import numpy as np
import cv2


ALLOW_OVERLAP = True


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


def _get_matching_rows(data_set, pattern_exp):
    '''

    :param data_set:
    :param pattern_exp: expanded binary pattern (not indices)
    :return:
    '''

    # if pattern_exp has no bins; i.e. zero-pattern; match to all rows
    if np.count_nonzero(pattern_exp) == 0:
        matching_row_indices = np.arange(data_set.shape[0]).astype(np.int)
        pattern_value = 0
    else:

        # main rule: select RF for frame as long as net reduction of remainder; i.e. most of the pattern is effective

        # total number of bins where (pattern == 1 && remainder == 1), per data_set input pattern row
        match_sum = np.sum(np.multiply(data_set, pattern_exp), axis=1)

        # total number of bins of pattern
        pattern_sum = np.sum(pattern_exp)

        # match if match_sum > pattern_sum / 2; in this case the RF has more bins that reduce the remainder than not
        # we don't care at all about bins where pattern is zero; we care about bins where:
        #   (pattern == 1 && remainder == 1)
        #   (pattern == 1 && remainder == 0)
        # we say pattern is a "match" if most (more than half) of the pattern has remainder == 1 on this frame

        matching_row_indices = np.nonzero(match_sum > pattern_sum / 2)[0]
        pattern_value = np.sum((match_sum - pattern_sum / 2)[matching_row_indices])

        # old, more strict rule:
        # matching_row_indices = np.nonzero(match_sum == pattern_sum)[0]
        # pattern_value = len(matching_row_indices) * pattern_sum

    return matching_row_indices, pattern_value


def _get_best_add_bin_index(current_pattern_exp, current_pattern_match_rows, remainder):
    '''

    get the "best" bin index to add to current_pattern_exp

    :param current_pattern_exp: expanded binary pattern
    :param remainder: data set
    :return:
    '''

    # subtract current_pattern_exp from matched rows of remainder and set negatives to zero
    # (or zero out all ones of current pattern in current matched rows)
    rem_tmp = remainder.copy()

    if np.count_nonzero(current_pattern_exp) > 0:
        # TODO is this still right? reduce set to what already matched? this is not a given now
        # Is it not the case that, with a new bin, new additional rows might match when you add a bin? then this whole conditional may not be accurate here...
        # we are assuming so but this might be overly limiting; exhaustive search would be running _get_matching_rows on every possible additional bin candidate, instead of just getting max
        rem_tmp = rem_tmp[current_pattern_match_rows, :]

        cumulative_pattern_indices = np.nonzero(current_pattern_exp)[0]
        rem_tmp[:, cumulative_pattern_indices.astype(np.int)] = 0

    # (3) find max prob bin in remainder copy after zeroing out above
    sum_bins = np.sum(rem_tmp, axis=0)
    max_bin_index = np.argmax(sum_bins)

    # find max bin

    # return max bin
    return max_bin_index


def _get_best_pattern(remainder, rf_index_for_debug=None, pixel_input=True):
    '''

    perfect method: i.e. binary pattern must be wholly contained in the input sample

    :param remainder:
    :return:
    '''

    num_bins_per_pixel = 6
    input_im_dim = 16

    feature_len = remainder.shape[1]

    if pixel_input:
        assert feature_len == num_bins_per_pixel * input_im_dim * input_im_dim

    max_pattern_bins = input_im_dim * input_im_dim

    cumulative_pattern_exp = np.zeros(feature_len)
    cumulative_pattern_indices = []
    cumulative_pattern_values = np.zeros(max_pattern_bins)
    cumulative_matching_rows, pattern_value = _get_matching_rows(data_set=remainder, pattern_exp=cumulative_pattern_exp)

    assert cumulative_matching_rows.shape[0] == remainder.shape[0]  # all rows should match for zero-pattern
    assert pattern_value == 0  # pattern value should be zero for zero-pattern

    for bin_num in range(1, max_pattern_bins):  # why start at one? zero-pattern is already defined
        # get next max bin given cumulative_matching_rows from previous iteration
        max_bin_index = _get_best_add_bin_index(current_pattern_exp=cumulative_pattern_exp, current_pattern_match_rows=cumulative_matching_rows, remainder=remainder)

        # update cumulative_pattern_exp, cumulative_pattern_indices
        cumulative_pattern_indices.append(max_bin_index)  # for being able to pick max pattern at the end
        cumulative_pattern_exp[max_bin_index] = 1

        # get new matching rows and value
        # cumulative_pattern_values, cumulative_matching_rows
        cumulative_matching_rows, pattern_value = _get_matching_rows(data_set=remainder, pattern_exp=cumulative_pattern_exp)
        cumulative_pattern_values[bin_num] = pattern_value

    # TODO ensure not off by one!
    best_pattern_length = np.argmax(cumulative_pattern_values)
    best_pattern = np.array(cumulative_pattern_indices)[0:best_pattern_length + 1].astype(np.int)

    # return indices of best pattern
    return best_pattern


def get_sparse_features(arr, num_rfs, pixel_input):

    feature_len = arr.shape[1]

    remainder = arr.copy()
    sparse_arr = np.zeros((num_rfs, feature_len))

    for k in range(num_rfs):
        print()
        print('Finding best pattern for RF: ', k)
        # get "best pattern" from remainder using iterative greedy max prob

        best_pattern = _get_best_pattern(remainder, rf_index_for_debug=k, pixel_input=pixel_input)
        best_pattern_exp = np.zeros(remainder.shape[1])
        best_pattern_exp[best_pattern.astype(np.int)] = 1

        # set "best pattern" as this row of sparse arr
        sparse_arr[k, :] = best_pattern_exp[:]

        # subtract "best pattern" from remainder where present

        matching_rows, _ = _get_matching_rows(data_set=remainder, pattern_exp=best_pattern_exp)

        print('   num matched rows: ', len(matching_rows))

        for matched_row in list(matching_rows):
            remainder[matched_row, best_pattern.astype(np.int)] = 0
            # consider later: do we need to worry about -1 introduced above if it was a subtraction and not a zeroing?

        if k%10 == 0 and pixel_input:
            im = make_im(sparse_arr)
            cv2.imshow('RFs', im)
        cv2.waitKey(1)

    return sparse_arr


def main():
    print()
    print('loading file...')
    print()

    arr = np.loadtxt('table_i_1.txt')
    # pixel_input = True for hl==0, else False
    pixel_input = False

    print('table_i shape:', arr.shape)
    print()

    sparse_arr = get_sparse_features(arr=arr, num_rfs=240, pixel_input=pixel_input)
    np.savetxt('rfs_balls_240_hl_1.txt', sparse_arr)

    if pixel_input:
        im_orig = make_im(arr)
        im = make_im(sparse_arr)
    else:
        im_orig = arr
        im = sparse_arr

    cv2.imshow('orig', im_orig)
    cv2.imshow('RFs', im)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()




