#This file will be sent to CHPC for Bayesian optimization

import numpy as np
import pandas as pd

import sys
import random

file_to_read = str(sys.argv[1])
option_fun = sys.argv[2]



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



###drop existing bonds



##function to get rid of functional groups that have already bonded
def remove_used_fun(mat_1):


    idx = np.where(mat_1[:, 2] < 2.00) #locate functional
    fun_id = mat_1[idx, 1]
    idx_2 = np.where(mat_1[:, 1] == fun_id)[1]

    mat_2 = np.delete(mat_1, idx_2, axis=0)



    return mat_2

def detect_locations(df, threshold=4.0, type_choose=22, **kwargs):


    if 'fun_type' in kwargs:
        fun_type = kwargs['fun_type']
        df_fun = df.loc[df['type'] == fun_type]

    elif 'fun_id' in kwargs:
        fun_id = kwargs['fun_id']
        df_fun = df.loc[df['id'] == fun_id]


    df_cnt = df.loc[df['type'] == type_choose]

    index_fun = df_fun.loc[:, ['id']].as_matrix()
    index_cnt = df_cnt.loc[:, ['id']].as_matrix()

    df_1 = df_cnt.loc[:, ['x', 'y', 'z']]
    df_2 = df_fun.loc[:, ['x', 'y', 'z']]
    z1 = compute_distance(df_1.as_matrix(), df_2.as_matrix())



    idx_choose = np.where((np.asarray(z1) < threshold))
    dist_select = z1[idx_choose]



    id_fun = (index_fun[idx_choose[0]]).flatten()
    id_cnt = (index_cnt[idx_choose[1]]).flatten()


    #hstack all the distances and the id's
    mat_out = np.vstack((id_cnt, id_fun, dist_select))
    mat_out = mat_out.transpose()



    df_fun_out = df_fun[df_fun.id.isin(id_fun)]
    df_cnt_out = df_cnt[df_cnt.id.isin(id_cnt)]


    return df_fun_out, df_cnt_out, mat_out



def compute_distance(cnt_array, cnt_other_array):

    dist_list = []
    points = cnt_array


    for j in range(0, len(cnt_other_array)):

        single_point = cnt_other_array[j, :]

        dist = np.sum(((points - single_point)**2), axis=1)
        dist = np.sqrt(dist)
        dist_list.append(dist)


    return np.asarray(dist_list)







def get_R_and_H(df, fun_atom_id, H_type, alkane_type=2):

    df_fun, df_C, _ = detect_locations(df=df, threshold=2.0, type_choose=alkane_type, fun_id=fun_atom_id)
    alkane_id = df_C.loc[:, ['id']].as_matrix()
    df_1 = df_fun.loc[:, ['x', 'y', 'z']]
    df_2 = df_C.loc[:, ['x', 'y', 'z']]
    z_C = compute_distance(df_1.as_matrix(), df_2.as_matrix()).flatten()


    #z_C = z_C[np.where((z_C[:, 2]) > 2.0)]

    idx_C = np.argmin(z_C)
    alkane_choose = int(alkane_id[idx_C][0])


    #Repeat the same process to
    _, df_H, _ = detect_locations(df=df, threshold=2.0, type_choose=H_type, fun_id = fun_atom_id)

    H_id = df_H.loc[:, ['id']].as_matrix()
    df_2 = df_H.loc[:, ['x', 'y', 'z']]
    z_H = compute_distance(df_1.as_matrix(), df_2.as_matrix()).flatten()

    print z_H

    idx_H = np.argmin(z_H)
    H_choose = int(H_id[idx_H][0])


    return alkane_choose, H_choose







def pick_group_and_atom(df, **kwargs):

    option_list = np.array([12, 9, 15, 18])

    if 'option_fun' in kwargs:
        option_fun = kwargs['option_fun']
        fun_type= option_list[int(option_fun)]
    else:
        rand_num = random.randint(0, len(option_list))
        fun_type = option_list[rand_num]

    df_fun, df_cnt, mat_1 = detect_locations(df=df, threshold=4.0, fun_type=fun_type)


    #This block makes sure that the bonded atoms are not


    #mat_1 = remove_used_fun(mat_1)



    print mat_1
    #select id among df_fun


    if 'fun_id' in kwargs:
        fun_id = kwargs['fun_id']
        idx_tmp = np.where(mat_1[:, 1] == fun_id)
        cnt_id = mat_1[idx_tmp, 0]

    elif 'minimize' in kwargs and kwargs['minimize']==True:
        idx_min = np.argmin(mat_1[:, -1])
        fun_id = mat_1[idx_min, 1]
        cnt_id = mat_1[idx_min, 0]

    else:
        rand_num = random.randint(0, len(mat_1)-1)
        fun_id = int(mat_1[rand_num, 1])
        cnt_id = int(mat_1[rand_num, 0])


    #now we can detect the hydrogen type by looking at the fun_type
    alkane_type = 2
    c_choose = 0

    if fun_type == 12:
        H_type = 1

        #Here we see if the epoxide atom works
        alkane_type = 2

    elif fun_type == 9 or fun_type == 15:
        H_type = 11
        alkane_type=4

        if fun_type == 15:
            chain_type = 14
    else:
        H_type = 20

    print "fun Id: ", fun_id

    alkane_choose, H_choose = get_R_and_H(df=df, fun_atom_id=fun_id, H_type=H_type, alkane_type=alkane_type)

    if fun_type == 15:
        c_choose, _ = get_R_and_H(df=df, fun_atom_id=fun_id, H_type=H_type, alkane_type=chain_type)


    return int(cnt_id), int(fun_id), int(alkane_choose), int(H_choose), int(c_choose)


df = read_datafile(file_to_read)

if file_to_read.startswith('data.3rr'):
    choose_del=False
else:
    choose_del=True

polymer_name = file_to_read[4:]

cnt_id, fun_id, R_id, H_id, c_id = pick_group_and_atom(df, option_fun=option_fun, minimize=True, polymer_name=polymer_name, choose_val=choose_del)

print cnt_id
print fun_id
print R_id
print H_id

if int(option_fun)==2:
    print c_id

#detect
