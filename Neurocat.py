import sys
import numpy as np
import os


class Loading:
    def __init__(self, nepoch, label=""):
        self.nepoch = nepoch
        self.lastepoch = 0
        if label != "":
            print("\n" + label)
        else:
            print()
        print("Loading %d Epochs" % self.nepoch)

    def loading(self, counter):
        if self.lastepoch != 79 * counter // self.nepoch:
            self.lastepoch = 79 * counter // self.nepoch
            sys.stdout.write("\r" + 
                             " " * self.lastepoch +
                             " " * (79 - self.lastepoch))
            sys.stdout.flush()

    def treshold(self):
        return self.nepoch

    def range(self):
        return range(self.nepoch)

    def in_progress(self, counter):
        return counter < self.nepoch


class Summary:
    RUNS = 1

    @classmethod
    def calc_runs(cls):
        cls.RUNS = 1
        while os.path.exists("./graphs/%d" % cls.RUNS):
            cls.RUNS += 1

    @classmethod
    def folder(cls):
        return "./graphs/%d" % cls.RUNS


def im2col(a, f, s):
    """
    A filter is a submatrix of the matrix 'a', containing a specified (by
    height and width of the filter) neighborhood of a matrix element
    m_{i,j} of matrix 'a'. Meaning: f_{k,l}=(i+k, j+l), respecting the
    matrix dimensions. In this scenario we go through every line of the
    matrix a with stepsize s. After finishing one line, we do the same
    procedure for the next line respecting the stepsize starting at
    (i,j)=(0,0).
    Assuming our matrix fits the filter and stepsize.
    :param a: matrix
    :param f: filtersize (filter height, filter width)
    :param s: stride
    :return: im2col matrix
    """
    # dimensions of Matrix
    m, n = a.shape
    # number of possible filters that fit in a row/column
    row_extend = m - f[0] + 1
    col_extend = n - f[1] + 1

    # start indices of a filter per row (left up)
    start_idx = np.arange(row_extend)[:, None] * n + np.arange(col_extend) * s

    # remaining indices for each filter
    off_idx = np.arange(f[0])[:, None] * n + np.arange(f[1])

    # get the indices we want to have when flatten the matrix
    return np.take(a, off_idx.ravel()[:, None] + start_idx.ravel())
