import sys
import numpy as np
import pandas as pd
from os import path

csvfile = str(sys.argv[1])
ogfile = str(sys.argv[2])

def init_file(file_to_read, fill_num=5):

    if path.exists(file_to_read):
        df = pd.read_table(file_to_read, delim_whitespace=False, delimiter=",")
        file_list = df.loc[:, ['file_new']].values.flatten().tolist()

        #parse filelast to get the last file
        if len(file_list) > 0:
            lastfile = file_list[-1]
            str_choose = lastfile.split('.')[1]
            num_choose = int(str_choose) + 1
            str_choose = str(num_choose).rjust(fill_num, '0')

            str_out = ogfile + '.' + str(str_choose)


        else:

            num_choose = int(1)
            str_choose = str(num_choose).rjust(fill_num, '0')
            str_out = ogfile + '.' + str(str_choose)



    return str_out




if __name__=="__main__":

    str_out = init_file(file_to_read=csvfile)

    print str_out