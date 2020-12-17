import numpy as np
import sys
import re
import glob

f_init = float(sys.argv[1]) #initial
delta = float(sys.argv[2])
T_thresh = float(sys.argv[3])


###The idea here is to extract the last timestep


def detect_f(f_init, T_thresh, delta=1e-6, dt=0.1, epsilon=0.03):


    raw_files = './dump.pe1epo.*' #change this when actually employed
    file_to_read =  sorted(glob.glob(raw_files), key=key_func)[-1]

    #read file
    with open(file_to_read) as f:

        text = f.readlines()
        max_t = float(text[1].rstrip('\n'))
        f_critical = f_init + max_t*delta*dt


    ###need to find a flag to see whether we need a new simulation or not
    ###if we perform the pullout test without actually
    if max_t >= T_thresh:
        flag = 0
        f_critical = f_critical - epsilon
    else:
        flag = 1


    return f_critical, flag



#keyfunction to sort filesa
def key_func(x):
    nondigits= re.compile("\D")

    return int(nondigits.sub("", x))



if __name__ == "__main__":

    f_c, flag = detect_f(f_init, T_thresh, delta)
    print f_c
    print flag