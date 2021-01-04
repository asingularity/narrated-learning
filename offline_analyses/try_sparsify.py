

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


def make_im(binary_arrays):

    tmp_im_all = None
    tmp_im = None

    num_bins_per_pixel = 6
    input_im_dim = 16

    arr, weights = _collapse_binned_columns_to_pixels(arr_exp=binary_arrays,
                                                      num_bins_per_pixel=6)

    for r_tmp in range(weights.shape[0]):  # per rf

        im0 = arr[r_tmp, :].reshape((input_im_dim, input_im_dim))  # rf_dim
        im1 = weights[r_tmp, :].reshape((input_im_dim, input_im_dim))

        # im1 = 0.5 * (im1 + 1)

        tmp2 = np.hstack((im0, im1))

        if tmp_im is None:
            tmp_im = tmp2.copy()
        else:
            tmp3 = 0.2 * np.ones((2, tmp_im.shape[1]))
            tmp_im = np.vstack((tmp_im, tmp3, tmp2))

        if r_tmp > 0 and (r_tmp + 1) % 10 == 0:
            if tmp_im_all is None:
                tmp_im_all = tmp_im.copy()
            else:
                spacer = 0.5 * np.ones((tmp_im_all.shape[0], 3))
                tmp_im_all = np.hstack((tmp_im_all, spacer, tmp_im))

            tmp_im = None

    tmp_all_im = tmp_im_all

    max_dim = max(tmp_all_im.shape[0], tmp_all_im.shape[1])
    imscale = 2000 / max_dim  # 0.2: full table, 2.0
    tmp_im = cv2.resize(tmp_all_im, dsize=(0, 0), fx=imscale, fy=imscale, interpolation=cv2.INTER_NEAREST)

    return tmp_im


def main():
    print()
    print('loading file...')
    print()

    arr = np.loadtxt('table_i.txt')
    print(arr.shape)

    im = make_im(arr)
    cv2.imshow('RFs', im)
    cv2.waitKey(0)



if __name__ == '__main__':
    main()