import numpy as np
import sys

L = float(sys.argv[1])
d_cnt = float(sys.argv[2])


def check_L(L, d_cnt, L_thresh=1.8, d_thresh=1.5):


    if  L < L_thresh and d_cnt < d_thresh:
        flag = 1

    else :
        flag = 0

    return flag




if __name__ == "__main__":
    flag = check_L(L=L, d_cnt=d_cnt)
    print flag