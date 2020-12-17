import numpy as np
import pandas as pd

import sys
import random

file_to_read = str(sys.argv[1])
option_fun = sys.argv[2]
fun_id = sys.argv[3]

def gen_filename(file_to_read, opt_fun, fun_id):

    file_old = file_to_read

    if file_to_read.endswith('epo'):
        polymer = file_to_read[-4:]
        file_to_read = file_to_read[:-4]



    opt_fun = int(opt_fun)
    #first decide on the string name to add:
    if opt_fun==0:
        opt_str = 'C'
    elif opt_fun==1:
        opt_str = 'Np'
    elif opt_fun==2:
        opt_str = 'Ns'
    elif opt_fun==3:
        opt_str = 'O'
    else:
        opt_str = 'None'

    gen_rand = str(random.randint(0, 999))
    str_out =  file_to_read + opt_str + gen_rand + polymer
    maintain_data(file_to_read=file_old, gen_rand=gen_rand, fun_id=fun_id)



    return str_out



def maintain_data(file_to_read, gen_rand, fun_id):

    polymer_name = file_to_read[-4:]
    data_filename = polymer_name + 'info' + '.txt'

    if file_to_read.startswith('data.3rr'):


        #if it is a new file then create an array of length 9
        #indicating that a max of 10 insertions is allowed

        #the first element in the array correspond to the random number
        new_array = np.zeros((1, 10))
        new_array[0, 0] = int(gen_rand)
        new_array[0, 1] = int(fun_id)

        with open(data_filename, 'ab') as f:
            np.savetxt(f, new_array)

    else:
        Z = np.loadtxt(data_filename)  #load old array

        if len(Z.flatten()) == 10:
            Z = Z[None, :]
            

        count = np.count_nonzero(Z[-1, :])

        #modify new array to account for
        last_array = Z[-1, :]
        new_array = last_array.copy()
        new_array[0] = int(gen_rand)
        new_array[count] = int(fun_id)
        new_array = new_array[None, :]


        with open(data_filename, 'ab') as f:
            np.savetxt(f, new_array)
            

    return None


if __name__ == "__main__":
    str_out = gen_filename(file_to_read=file_to_read, opt_fun=option_fun, fun_id=fun_id)

    print str_out



