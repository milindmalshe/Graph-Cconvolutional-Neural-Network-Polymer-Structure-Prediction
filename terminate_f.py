import numpy as np
import sys

file_to_read = str(sys.argv[1])
f_val = float(sys.argv[2])
v_val = float(sys.argv[3])

def save_file(filename, f_val, v_val, v_thresh=0.0035, epsilon=0.001, epsilon_f = 0.03):

    #Z = np.transpose(np.array([f_val, v_val]))
    Z = (np.array([f_val, v_val]))
    Z = Z[None, :]
    d = v_thresh - v_val

    if np.absolute(d) < epsilon:
        flag = 1
        if d > 0:
            f_final = f_val - epsilon_f

        else:
            f_final = f_val - 1.5*epsilon_f

    else:
        flag = 0
        f_final = 0


    try:
        test_arr = np.loadtxt(filename)
        test_arr = np.vstack((test_arr, Z))

        np.savetxt(filename, test_arr)

    except:
        np.savetxt(filename, Z)






    return flag, f_final


if __name__ == "__main__":
    flag, f_final = save_file(filename=file_to_read, f_val=f_val, v_val=v_val)
    print flag
    print f_final