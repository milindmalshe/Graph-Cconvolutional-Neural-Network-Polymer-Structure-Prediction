import numpy as np
import sys

from os import path

import pandas as pd

file_to_read = str(sys.argv[1])
naive_file = str(sys.argv[2])
seq_opt = int(sys.argv[3]) #whether we should let it run randomly or not
opt_fun = int(sys.argv[4]) #opt_fun > 6 -> random

#option_fun = sys.argv[2]

def init_file(file_to_read, naive_file):

    if path.exists(file_to_read):
        df = pd.read_table(file_to_read, delim_whitespace=False, delimiter=",")
        file_list = df.loc[:, ['file_new']].values.flatten().tolist()
        opt_mat = df.loc[:, ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']].as_matrix()

        opt_final = opt_mat[-1, :].flatten()[-1] #last element in opt_final
        file_list.insert(0, naive_file)




    #file_list = Z[:, -1]



    return file_list, opt_final





if __name__ == "__main__":
    file_list, opt_final = init_file(file_to_read=file_to_read, naive_file=naive_file)
    opt_max = 5 #change this for silicon

    if seq_opt == 0 or opt_final != 0:

        #pick a random number
        rand_num = np.random.randint(low=0, high=len(file_list))
        file_choose = file_list[rand_num]
        file_num = rand_num

    else:

        file_choose = file_list[-1]
        file_num = len(file_list) - 1

    if opt_fun < 6:
        opt_choose = opt_fun
    else:
        opt_choose = np.random.randint(low=0, high=opt_max)



    print file_choose
    print file_num
    print opt_choose



