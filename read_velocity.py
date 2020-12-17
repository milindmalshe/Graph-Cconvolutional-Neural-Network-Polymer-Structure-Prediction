import numpy as np
import sys

file_to_read = str(sys.argv[1])

def read_velocity(file_to_read):

    Z = np.loadtxt(fname=file_to_read, skiprows=1)
    v_f = Z[-1]

    return v_f




if __name__ == "__main__":

    v_f = read_velocity(file_to_read=file_to_read)
    print v_f

