#This file will be sent to CHPC for Bayesian optimization

import numpy as np
import pandas as pd

import sys
import random

file_to_read = str(sys.argv[1])
option_fun = sys.argv[2]
init_file = str(sys.argv[3])
file_num = int(sys.argv[4])



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

def drop_existing_bonds(df, init_file, file_num, threshold=2.0, type_choose=22):

    init_df = pd.read_table(init_file, delim_whitespace=False, delimiter=",")
    option_list = [12, 9, 15, 18, 23]

    if file_num > 0:
        idx = file_num - 1
        used_fun = init_df.loc[:, ['opt1', 'opt2', 'opt3','opt4', 'opt5']].as_matrix()[idx]

        used_fun = np.unique((used_fun[used_fun > 0])) #list of used function
	used_fun = used_fun.astype(int)


        df_cnt = df.loc[df['type']==type_choose]
        idx_cnt = df_cnt.loc[:, ['id']].as_matrix()

	#print "used fun"
	#print used_fun
        #for each functional group in the list, eliminate one by one
        for num in used_fun:

            fun = option_list[num-1]

            df_fun = df.loc[df['type']==fun]
            idx_fun = df_fun.loc[:, ['id']].as_matrix()

            #find where distance < 2.0

            z1 = compute_distance(df_cnt.loc[:, ['x', 'y', 'z']].as_matrix(), df_fun.loc[:, ['x', 'y', 'z']].as_matrix())
            idx_choose = np.where((np.asarray(z1) < threshold))


            id_fun = (idx_fun[idx_choose[0]]).flatten()
            id_cnt = (idx_cnt[idx_choose[1]]).flatten()

            df = df[~df['id'].isin(id_fun)]
            df = df[~df['id'].isin(id_cnt)]





    return df


####check if the recommended changes have already been implemented -> else recommend new changes
def check_init_file(init_file, cnt_id, fun_id, file_num):

    init_df = pd.read_table(init_file, delim_whitespace=False, delimiter=",")
    
    if file_num > 0:	
    	file_num = file_num - 1

    cnt_list = init_df.loc[:, ['cnt1', 'cnt2', 'cnt3', 'cnt4', 'cnt5']].as_matrix()[file_num, :]
    cnt_new = cnt_list.copy()
    cnt_new = cnt_new.astype(int)

    fun_list = init_df.loc[:, ['fun1', 'fun2', 'fun3', 'fun4', 'fun5']].as_matrix()[file_num, :]
    fun_new = fun_list.copy()
    fun_new = fun_new.astype(int)

    idx_zero = (np.where(cnt_list==0))[0]

    cnt_new[idx_zero[0]] = cnt_id
    fun_new[idx_zero[0]] = fun_id
    

    cnt_all = init_df.loc[:, ['cnt1', 'cnt2', 'cnt3', 'cnt4', 'cnt5']].as_matrix()
    fun_all = init_df.loc[:, ['fun1', 'fun2', 'fun3', 'fun4', 'fun5']].as_matrix()

    cnt_all = cnt_all.astype(int)
    fun_all = fun_all.astype(int)

    d_1 = np.linalg.norm(cnt_all - cnt_new, axis=1)
    d_2 = np.linalg.norm(fun_all - fun_new, axis=1)

    #print cnt_all
    #print cnt_new
    #print fun_all
    #print fun_new
    #print d_1
    #print d_2

    bool_out = np.any(d_1 == 0) and np.any(d_2 == 0)


    return bool_out



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
    idx_C = np.argmin(z_C)
    alkane_choose = int(alkane_id[idx_C][0])


    #Repeat the same process to
    _, df_H, _ = detect_locations(df=df, threshold=2.0, type_choose=H_type, fun_id = fun_atom_id)
    H_id = df_H.loc[:, ['id']].as_matrix()
    df_2 = df_H.loc[:, ['x', 'y', 'z']]
    z_H = compute_distance(df_1.as_matrix(), df_2.as_matrix()).flatten()



    if z_H.size != 0:
        idx_H = np.argmin(z_H)
        H_choose = int(H_id[idx_H][0])
    else:
        H_choose = 0


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


    #select id among df_fun
    fun_array = (df_fun.loc[:, ['id']].as_matrix()).flatten()
    cnt_array = (df_cnt.loc[:, ['id']].as_matrix()).flatten()

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
    chain_type = 14
	
    if fun_type == 12:
        H_type = 1
    elif fun_type == 9 or fun_type == 15:
        H_type = 11
	alkane_type = 4
	
	if fun_type == 15:
	    chain_type = 14	

    else:
        H_type = 20


    alkane_choose, H_choose = get_R_and_H(df=df, fun_atom_id=fun_id, H_type=H_type, alkane_type=alkane_type)


    if fun_type == 15:
	    c_choose, _ = get_R_and_H(df=df, fun_atom_id=fun_id, H_type=H_type, alkane_type=chain_type)
	
    return int(cnt_id), int(fun_id), int(alkane_choose), int(H_choose), int(c_choose)




#-------------------------------------------------
df = read_datafile(file_to_read)



#drop existing bonds:
df = drop_existing_bonds(df, init_file=init_file, file_num=file_num)

H_id = 0

while H_id == 0:
    cnt_id, fun_id, R_id, H_id, c_id = pick_group_and_atom(df, option_fun=option_fun, minimize=False)

if file_num > 0:
    check_init_file(init_file=init_file, cnt_id=cnt_id, fun_id=fun_id, file_num=file_num)

print cnt_id
print fun_id
print R_id
print H_id

if int(option_fun)==2:
	print c_id


#detect
