import numpy as np
import sys

file_to_read = str(sys.argv[1])
f0 = float(sys.argv[2])


def apply_lin(filename, f0, v_lim=0.0035, v_thres=0.0002, gamma=1.5):

    #f0 is dependent on the previous sequence
    #setting up test features

    try:
        X = np.loadtxt(filename)

        f = X[:, 0]
        v = X[:, 1]

        v_lin = np.empty((1, 0))

        if len(X) > 1:
            v_lin = v[v > v_thres]


        print v_lin

        if v_lin.size > 1:
            idx = np.where(v> v_thres)
            f_lin = f[idx]

            z = np.polyfit(f_lin, v_lin, 1)

            f_next = (v_lim - z[1])/z[0]
        else:
            f_next = gamma*f0

    except:
        f_next = f0




    return f_next



if __name__ == "__main__":
    f_next = apply_lin(filename=file_to_read, f0=f0)

    print f_next





