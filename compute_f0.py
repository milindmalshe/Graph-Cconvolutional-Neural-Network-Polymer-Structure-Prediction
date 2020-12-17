import numpy as np
import sys
import pandas as pd

file_to_read = sys.argv[1]
csvfile = sys.argv[2]


def find_f(file_to_read, csvfile):


    try:

        seq_id = int(file_to_read.split('.')[1])
        mat = pd.read_table(csvfile, delimiter=",").as_matrix()

        f_out = mat[seq_id, 1]


    except:

        f_out = 0


    return f_out




if __name__ == "__main__":

    f_out = find_f(file_to_read=file_to_read, csvfile=csvfile)
    print f_out