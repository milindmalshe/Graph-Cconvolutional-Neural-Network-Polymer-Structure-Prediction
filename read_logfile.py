import numpy as np
import pandas as pd
import sys

file_in = str(sys.argv[1])


def read_datafile(file_in):

    count = 0

    with open(file_in) as f:
        for line in f:
            count +=1
            if line.startswith('#step1-start'):
                skip_lines = count



    #skip three lines to get straight to the header

    df = pd.read_table(file_in, delim_whitespace=True, header=None, skiprows=skip_lines+4, nrows=6)

    df = df[[0, 20, 21, 22]]
    df.columns = ['Time', 'c_cntvz', 'c_epxvz', 'f_hold[3]']

    v_mat = df['c_cntvz'].as_matrix()
    v_out = v_mat[-1]

    return v_out


v_out = read_datafile(file_in=file_in)
print v_out
