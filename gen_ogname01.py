import numpy as np
import sys



def gen_ogname(first_str, rand_num):

    str_out = first_str + str(rand_num) + 'epo'


    return str_out



if __name__ == "__main__":

    main_str = 'data.3rr'
    rand_num = np.random.randint(low=0, high=5) + 1

    str_out = gen_ogname(first_str=main_str, rand_num=rand_num)

    print str_out