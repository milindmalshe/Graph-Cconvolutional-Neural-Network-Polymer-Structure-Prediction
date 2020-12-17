
import numpy as np
import pandas as pd

import sys
import random

file_to_read = str(sys.argv[1])



def read_datafile(file_in):

    count = 0

    with open(file_in) as f:
        for line in f:
            count +=1
            if line.startswith('Atoms #'):
                skip_lines = count

            if line.startswith('Velocities'):
                natoms = count - 3



    df = pd.read_table(file_in, delim_whitespace=True, header=None, skiprows=skip_lines, nrows=natoms-skip_lines)
    df = df.drop(columns=2)
    df.columns = ['id', 'type', 'x', 'y', 'z', 'nx', 'ny', 'nz']



    return df



def check_amines(file_in, primary_N = 9, secondary_N=15, tertiary_N=16):


    df = read_datafile(file_in=file_in)

    count_N1 = len((df.loc[df['type']==primary_N]).as_matrix())
    count_N2 = len((df.loc[df['type']==secondary_N]).as_matrix())
    count_N3 = len((df.loc[df['type']==tertiary_N]).as_matrix())



    return count_N1, count_N2, count_N3



def check_carbon(file_in, epoxide_C=12, reacted_C=14):

    df = read_datafile(file_in=file_in)

    C12 = len((df.loc[df['type'] == epoxide_C]).as_matrix())
    C14 = len((df.loc[df['type'] == reacted_C]).as_matrix())


    return C12, C14



if __name__ == "__main__":

    N1, N2, N3 = check_amines(file_in=file_to_read)
    C12, C14 = check_carbon(file_in=file_to_read)
    print N1
    print N2
    print N3
    print float(N2+2*N3)/float(N1)
    print float(N2 + 2*N3)/float(2*N1 + 2*N2 + 2*N3)

    print "Carbon: "
    print float(C12)/float(C14)


