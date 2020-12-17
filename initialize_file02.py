import numpy as np
import sys

from os import path

import pandas as pd

file_to_read = str(sys.argv[1])
naive_file = str(sys.argv[2])
seq_opt = int(sys.argv[3]) #whether we should let it run randomly or not
opt_fun = int(sys.argv[4]) #opt_fun > 6 -> random

#option_fun = sys.argv[2]

def init_file(file_to_read, naive_file, N=5):

    if path.exists(file_to_read):
        df = pd.read_table(file_to_read, delim_whitespace=False, delimiter=",")
        file_list = df.loc[:, ['file_new']].values.flatten().tolist()
        opt_mat = df.loc[:, ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']].as_matrix()

        opt_final = opt_mat[-1, :].flatten()[-1] #last element in opt_final
        file_list.insert(0, naive_file)

        #opt_final decides whether this is a naive structure or not

    else:

        #if path doesn't exist, then create a file of zeros and create a new file
        opt_final = 0
        file_list = [naive_file]

        #create a new file
        cnt_list =[]
        opt_list = []
        fun_list = []
        for i in range(N):
            cntstr = 'cnt' + str(i+1)
            optstr = 'opt' + str(i+1)
            funstr = 'fun' + str(i+1)

            cnt_list.append(cntstr)
            opt_list.append(optstr)
            fun_list.append(funstr)


        #creating a new file
        header_list = ['seq_ID'] + cnt_list + opt_list + fun_list + ['file_old'] + ['file_new'] + ['min_flag']
        df_out = pd.DataFrame(columns=header_list)
        df_out.to_csv(file_to_read, index=False)




    


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



