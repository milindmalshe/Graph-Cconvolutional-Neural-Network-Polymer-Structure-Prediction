import numpy as np

import numpy as np
import pandas as pd

import sys
import random

file_to_read = str(sys.argv[1])
cnt_in = int(sys.argv[2])

def read_datafile(file_in):

    count = 0

    with open(file_in) as f:
        for line in f:
            count +=1
            if line.startswith('Atoms #'):
                skip_lines = count

            if line.startswith('Velocities'):
                natoms = count - 3



    df = pd.read_table(file_in, delim_whitespace=True, header=None, skiprows=skip_lines, nrows=natoms-skip_lines)
    df = df.drop(columns=2)
    df.columns = ['id', 'type', 'x', 'y', 'z', 'nx', 'ny', 'nz']



    return df



def compute_distance(cnt_array, cnt_other_array):

    dist_list = []
    points = cnt_array


    for j in range(0, len(cnt_other_array)):

        single_point = cnt_other_array[j, :]

        dist = np.sum(((points - single_point)**2), axis=1)
        dist = np.sqrt(dist)
        dist_list.append(dist)


    return np.asarray(dist_list)




def locate_cnt(df, type_choose=22, random_var=True, **kwargs):

    df_cnt = df.loc[df['type'] == type_choose]
    df_other = df.loc[df['type'] != type_choose]
    other_mat = df_other.loc[:, ['x', 'y', 'z']].as_matrix()

    if random_var==True:

        cnt_xyz = (df_cnt.loc[:, ['id', 'x', 'y', 'z']]).as_matrix()
        len_cnt= int(len(cnt_xyz))
        rand_num = random.randint(0, len_cnt)

        cnt_choose = cnt_xyz[rand_num, :]
        cnt_id = cnt_choose[0]
        x_out = cnt_choose[1]
        y_out = cnt_choose[2]
        z_out = cnt_choose[3]


    elif 'cnt_id' in kwargs:

        cnt_id = kwargs['cnt_id']
        df_choose = df.loc[df['id']==cnt_id]


        cnt_choose = (df_choose.loc[:, ['x', 'y', 'z']]).as_matrix()
        cnt_choose = cnt_choose.flatten()
        x_out = cnt_choose[0]
        y_out = cnt_choose[1]
        z_out = cnt_choose[2]

    else:

        cnt_id = 0
        x_out = 0
        y_out = 0
        z_out = 0



    ###this block of code to find difference between
    pick_cnt = np.array([x_out, y_out, z_out])
    dist_mat = np.linalg.norm((pick_cnt - other_mat), axis=1)

    idx_mat = np.argmin(dist_mat)

    print "troubleshoot"

    print other_mat[idx_mat]
    print dist_mat[idx_mat]

    print "new coords: "

    other_cords = other_mat[idx_mat]
    new_cords = np.array([(0.5*pick_cnt[0] + 0.5*other_cords[0]), 1*(0.5*pick_cnt[1] + 0.5*other_cords[1]), 1*(0.5*pick_cnt[2] + 0.5*other_cords[2])])

    print new_cords


    return cnt_id, x_out, y_out, z_out



if __name__ == "__main__":

    df = read_datafile(file_in=file_to_read)

    if cnt_in == 0:
        cnt_id, x_out, y_out, z_out = locate_cnt(df=df)
    else:
        cnt_id, x_out, y_out, z_out = locate_cnt(df=df, random_var=False, cnt_id=cnt_in)


    print int(cnt_id)
    print x_out
    print y_out
    print z_out
