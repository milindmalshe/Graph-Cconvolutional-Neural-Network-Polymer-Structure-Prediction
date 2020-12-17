import numpy as np
import pandas as pd
import hotspot_file_01
import matplotlib.pyplot as plt

import pymp
import math
import multiprocessing


from numpy import linalg as LA
from scipy import optimize
import hotspot_file_01

def link_models(init_file_list, file_list, cnt_data, RBF_data, bound_mat, rbf_1, rbf_2, f=400):

    #df_list -> list of all models that are

    print file_list

    if len(init_file_list) == 0:
        init_df = pd.read_table(init_file_list, delim_whitespace=False, delimiter=",")
    else:
        # concatenating multiple dataframes
        list_temp = []
        for filename in init_file_list:
            df_temp = pd.read_table(filename, delim_whitespace=False, delimiter=",")
            list_temp.append(df_temp)

        init_df = pd.concat(list_temp, axis=0, ignore_index=True)

    file_list = get_split_string(file_list)
    df_file = init_df[init_df['file_new'].isin(file_list)]
    df_file = df_file[df_file['file_old'].isin(file_list)]
    df_file = config_init_files(df_file)


    file_old = df_file.loc[:, ['file_old']].as_matrix()
    file_new = df_file.loc[:, ['file_new']].as_matrix()



    idx_list = np.zeros((len(file_old), ))

    #rbf_1, rbf_coars
    ###read df_list for file_list
    df_list, _, _ = hotspot_file_01.read_MD_model(add_dir(file_old.squeeze().tolist()))
    df_list_02, _, _ = hotspot_file_01.read_MD_model(add_dir(file_new.squeeze().tolist()))

    RBF_O1 = []
    RBF_O2 = []
    RBF_N1 = []
    RBF_N2 = []

    list_A1 = []
    list_A2 = []

    list_A1n = []
    list_A2n = []


    #for i in range(0, 10):
    for i in range(len(file_old)):

        idx_list[i] = i

        print "checkpont 2: "
        #print file_old[i]
        #print file_new[i]


        idx_1 = np.where(file_list==file_old[i])[0][0] #indexing where the file_old meets
        idx_2 = np.where(file_list==file_new[i])[0][0] #indexing where the new file_list

        #print "checkpont 3: "
        #print idx_1
        #print idx_2

        rbf_old1 = rbf_1[idx_1*f:(idx_1+1)*f, :, :, :]  #dimension idx *z*x*columns, along with their augmented copies
        rbf_old2 = rbf_2[idx_1*f:(idx_1+1)*f, :, :, :]

        rbf_new1 = rbf_1[idx_2*f:(idx_2+1)*f, :, :, :]
        rbf_new2 = rbf_2[idx_2*f:(idx_2+1)*f, :, :, :]

        str_choose = 'arr_' + str((idx_1) * f)
        str_choose02 = 'arr_' + str((idx_2)*f)

        R1 = RBF_data[str_choose]
        R2 = RBF_data[str_choose02]

        R1, R2 = troubleshoot_cnt(R1, R2)
        print np.unravel_index(np.argmax(R1 - R2), R1.shape)
        print np.amax(R1 - R2)


        #the problem is we need to augment the actions as well
        #need to keep tract of the cnt atom in question
        #need to find the cnt id

        cnt_list = df_file.loc[:, ['cnt1', 'cnt2', 'cnt3', 'cnt4', 'cnt5']].as_matrix()[i, :]
        cnt_id = cnt_list[cnt_list > 0][-1]

        fun_list = df_file.loc[:, ['fun1', 'fun2', 'fun3', 'fun4', 'fun5']].as_matrix()[i, :]
        fun_id = fun_list[fun_list > 0][-1]

        opt_list = df_file.loc[:, ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']].as_matrix()[i, :]
        opt_type= opt_list[opt_list > 0][-1]


        #### let's start defining the action space
        #first computing the distance
        df = df_list[i]
        df_cnt = df.loc[df['id'] == cnt_id]
        df_fun = df.loc[df['id'] == fun_id]
        dist = compute_distance(df_cnt.loc[:, ['x', 'y', 'z']].as_matrix(), df_fun.loc[:, ['x', 'y', 'z']].as_matrix())[0]

        cnt_xyz = df_cnt.loc[:, ['x', 'y', 'z']].as_matrix()

        #fun id
        fun_xyz = df_fun.loc[:, ['x', 'y', 'z']].as_matrix()



        #now we can simply keep track of the cnt atom in question to observe
        print "i: ", i
        print "cnt_xyz: ", cnt_xyz
        print "distance: ", LA.norm(cnt_xyz.flatten() - fun_xyz.flatten())


        ##now look at the new df and if it tallies up
        df_n = df_list_02[i]
        cnt_new = new_df(df=df_n, cnt_o=cnt_xyz)

        print "cnt_new: ", cnt_new


        ####check at the new

        #print "dist: ", dist
        #print "opt type:", opt_type
        #print "bound_mat: ", bound_mat.shape
        #print "idx_1: ", idx_1
        #print cnt_data['arr_0'].shape
        #print cnt_data['arr_0'][0:10, :, :]
        #print cnt_data['arr_1'][0:10, :, :]




        #feed distance and opt_type to get the naive action matrix:
        A = define_action_space(a=(opt_type-1), r_pick=dist)
        #print A.shape
        if dist < 5.0:
            print "old"
            A1, A2 = cnt_check(cnt_data=cnt_data, cnt_xyz=cnt_xyz, idx_1=idx_1, bound_mat=bound_mat, action_vector=A)
            print "new"
            A1_n, A2_n = cnt_check(cnt_data=cnt_data, cnt_xyz=cnt_new, idx_1=idx_2, bound_mat=bound_mat, action_vector=A)

            ##append to list

                #convert RBF_01 to an array
            #rbf_old1 = rbf_old1.reshape(-1, rbf_old1.shape[3])
            #rbf_old2 = rbf_old2.reshape(-1, rbf_old2.shape[3])
            RBF_O1.append(rbf_old1)
            RBF_O2.append(rbf_old2)

            #rbf_new1 = rbf_new1.reshape(-1, rbf_new1.shape[3])
            #rbf_new2 = rbf_new2.reshape(-1, rbf_new2.shape[3])
            RBF_N1.append(rbf_new1)
            RBF_N2.append(rbf_new2)

            #A1 = A1.reshape(-1, A1.shape[3])
            #A2 = A2.reshape(-1, A2.shape[3])
            list_A1.append(A1)
            list_A2.append(A2)

            list_A1n.append(A1_n)
            list_A2n.append(A2_n)

    return RBF_O1, RBF_O2, RBF_N1, RBF_N2, list_A1, list_A2, list_A1n, list_A2n, file_old, file_new



def compute_distance(cnt_array, cnt_other_array):

    dist_list = []
    points = cnt_array


    for j in range(0, len(cnt_other_array)):

        single_point = cnt_other_array[j, :]

        dist = np.sum(((points - single_point)**2), axis=1)
        dist = np.sqrt(dist)
        dist_list.append(dist)


    return np.asarray(dist_list)



def define_action_space(a, r_pick, R_space=[2.25, 2.75, 3.25, 3.75], action_space=np.arange(0,4)):

    #Rspace was discretized

    #define action matrix:
    A = np.zeros((len(R_space), len(action_space)))

    a_idx = np.where(action_space==a)
    r_idx = np.argmin(np.absolute(R_space - r_pick))

    temp_val = R_space[r_idx] - r_pick
    print "a_idx: ", a_idx
    print "r_idx:", r_idx
    print "temp_val: ", temp_val



    A[r_idx, a_idx ] = 1
    A = A.flatten()

    return A


def get_split_string(file_list):

    filename_out = file_list.copy()

    count = 0

    for filename in file_list:
        filename_out[count] = filename.rsplit('/')[-1]
        count += 1

    return filename_out


###----need a function to remove
def config_init_files(df):

    bool_s = df.index.values[df['file_old'].str.startswith('data.')]

    for num in bool_s:
        file_choose = df.at[num, 'file_old']
        last_name = file_choose.rsplit('.')[-1]
        new_name = last_name + '.00000'

        df.at[num, 'file_old'] = new_name



    return df


def add_dir(file_list):

    file_out = ['./MDfiles/select03/' + s for s in file_list]


    return file_out


def cnt_check(cnt_data, cnt_xyz, idx_1, bound_mat, action_vector, f=400, yc=30.0):

    #define the action matrix A1 and A2
    A1 = np.zeros((f, bound_mat.shape[1], bound_mat.shape[2], len(action_vector))) #lower half
    A2 = np.zeros_like(A1)

    #there are three steps to finding the new matrix
    #first task is to look at the original matrix and see
    str_choose = 'arr_' + str((idx_1)*f)
    cnt_0 = cnt_data[str_choose][0, :, :]



    #print cnt_0
    #print cnt_xyz
    #print "bound mat: "
    #print bound_mat.shape

    diff_mat = cnt_0 - cnt_xyz
    idx_cnt = np.where(LA.norm(diff_mat, axis=1)==0.0)



    #print "dist_mat"
    #print LA.norm(diff_mat, axis=1)
    bottom_count = 0
    top_count = 0

    #A is a one dimensional array

    for i in range(0, f):
        str_1 = 'arr_' + str(idx_1*f + i)
        cnt_1 = cnt_data[str_1][0, :, :]
        cnt_pick = np.squeeze(cnt_1[idx_cnt, :])

        bound_pick = bound_mat[idx_1*f + i, :, :, :]

        if i == 0:
            print "troubleshoot cnt check: "
            print "cnt_pick: ", cnt_pick
            print bound_mat.shape

        #print cnt_pick
        #print bound_pick

        bool_1 = np.logical_and(cnt_pick[0] > bound_pick[:, :, 0], cnt_pick[0] < bound_pick[:, :, 2])
        bool_2 = np.logical_and(cnt_pick[2] > bound_pick[:, :, 1], cnt_pick[2] < bound_pick[:, :, 3])
        bool_g = np.logical_and(bool_1, bool_2)
        idx_g = (np.where(bool_g == True))

        #print 'idx_2: '
       # print idx_g
        #this block shows where cnt_pick

        #first check if its the top or the bottom half
        bool_y = cnt_pick[1] < yc

        if i == 0:
            print "bool: ", bool_y
            print "idx_g: ", idx_g
            print "action_vector: ", action_vector

        if bool_y == True:
            A1[i, idx_g[0], idx_g[1], :] = action_vector
            #print "goes to bottom"
            bottom_count += 1

        else:
            A2[i, idx_g[0], idx_g[1], :] = action_vector
            #print "goes to top"


            top_count += 1
            #print goes to top


        #A1_o = A1.reshape(-1, A1.shape[])
        #picking the cnt:
    #first let's pick out
    print "counts: "
    print top_count
    print bottom_count


    return A1, A2



def convert_local_data(RBF_O1, RBF_O2, RBF_N1, RBF_N2, A1, A2):


    R_v = np.load('R_v.npy')


    #the rbfs and A1/A2 are lists, each list is a data point plus augmented copies
    N = len(RBF_O1)*(RBF_O1[0].shape[0])*(RBF_O1[0].shape[1])*(RBF_O1[0].shape[2])
    #N = len(RBF_O1) * (RBF_O1[0].shape[1]) * (RBF_O1[0].shape[2])

    rbf_o1 = np.zeros((N, RBF_O1[0].shape[3]))
    rbf_o2 = np.zeros_like(rbf_o1)

    rbf_n1 = np.zeros_like(rbf_o1)
    rbf_n2 = np.zeros_like(rbf_o2)

    a1 = np.zeros((N, A1[0].shape[3]))
    a2 = np.zeros_like(a1)

    count = 0
    count_2 = 0
    for m in range(0, len(RBF_O1)):

        for n in range(0, RBF_O1[m].shape[0]):

            for i in range(0, RBF_O1[m].shape[1]):

                for j in range(0,  RBF_O1[m].shape[2]):

                    rbf_o1[count, :] = RBF_O1[m][n, i, j, :]
                    rbf_o2[count, :] = RBF_O2[m][n, i, j, :]

                    rbf_n1[count, :] = RBF_N1[m][n, i, j, :]
                    rbf_n2[count, :] = RBF_N2[m][n, i, j, :]


                    a1[count, :] = A1[m][n, i, j, :]

                    a2[count, :] = A2[m][n, i, j, :]

                    a = A1[m][n, i, j, :]

                    if np.any(a > 0):
                        max_idx = fix_action(rbf_n=RBF_N1[m], a1=a, R_v=R_v, i=i, j=j)
                        #print "count: ", count
                        #print "max_idx: ", max_idx

                        if max_idx == -1 or max_idx > 0:
                            count_2 += 1
                        else:
                            #print RBF_N1[m][n, i, j, :].reshape(14, 7)
                            print "i, j:",  i, j


                    count += 1


    print "count 2: ", count_2


    return rbf_o1, rbf_o2, rbf_n1, rbf_n2, a1, a2



def shuffle_local01(RBF_O, A1, RBF_N):


    indices = np.random.permutation(RBF_O.shape[0])

    rbf_o, a1, rbf_n = RBF_O[indices, :], A1[indices, :], RBF_N[indices, :]


    return rbf_o, a1, rbf_n



def fix_action(rbf_n,  a1, R_v, i, j, R_space=[2.25, 2.75, 3.25, 3.75], action_space=np.arange(0,4), first_idx=7, thres_val=0.2):

    #rbf_n has dimensions len(orignal dataset)*states
    #same for a1
    R_v = R_v[0, :]

    a1_mat = a1.reshape(len(R_space), len(action_space))
    Z_out =  rbf_n[0, i, j, :].copy()
    idx_a = np.where(a1_mat > 0)[1]
    idx_a = identify_idx(idx_a=idx_a)[0]

    #print len(R_v)
    #print int(rbf_n.shape[3]/len(R_v))
    #print rbf_n[0, i, j, :].shape
    #print int(len(rbf_n[0, i, j, :])/len(R_v))

    Z_temp =  rbf_n[0, i, j, :].reshape(len(R_v), int(len(rbf_n[0, i, j, :])/len(R_v)))
    #print a1_mat
    #print "idx_a:"
    #print idx_a
    #print "Z_temp: "
    #print Z_temp[0:first_idx, idx_a]
    ##keep track of Z_temp

    if np.all(Z_temp[0:first_idx, idx_a] < thres_val) :

        #define high values for p and q
        p2 = i+1
        q2 = j+1

        if i + 1 >= rbf_n.shape[1]:
            p2 = 0

        if j+ 1>= rbf_n.shape[2]:
            q2 = 0

        Q_temp = np.zeros((8, len(R_v), int(rbf_n.shape[3]/len(R_v))))
        Q_temp[0, :, :] = rbf_n[0, i-1, j, :].reshape(len(R_v), int(rbf_n.shape[3]/len(R_v)))
        Q_temp[1, :, :] = rbf_n[0, p2, j, :].reshape(len(R_v), int(rbf_n.shape[3] / len(R_v)))
        Q_temp[2, :, :] = rbf_n[0, i, j-1, :].reshape(len(R_v), int(rbf_n.shape[3] / len(R_v)))
        Q_temp[3, :, :] = rbf_n[0, i, q2, :].reshape(len(R_v), int(rbf_n.shape[3] / len(R_v)))
        Q_temp[4, :, :] = rbf_n[0, p2, q2, :].reshape(len(R_v), int(rbf_n.shape[3] / len(R_v)))
        Q_temp[5, :, :] = rbf_n[0, i-1, q2, :].reshape(len(R_v), int(rbf_n.shape[3] / len(R_v)))
        Q_temp[6, :, :] = rbf_n[0, p2, j-1, :].reshape(len(R_v), int(rbf_n.shape[3] / len(R_v)))
        Q_temp[7, :, :] = rbf_n[0, i-1, j - 1, :].reshape(len(R_v), int(rbf_n.shape[3] / len(R_v)))

        #print "Q_temp"
        #print Q_temp[:, :, idx_a]
        max_idx = np.unravel_index(np.argmax(Q_temp[:, 0:first_idx, idx_a], axis=None), Q_temp[:, 0:first_idx, idx_a].shape)[0]
        #print "max idx: ", max_idx
        Q_out = Q_temp[max_idx, :, :].flatten()

    else:
        #print "max_idx: "
        max_idx = -1



    return max_idx




def identify_idx(idx_a, unique_type=np.asarray([2, 4, 9, 12, 15, 18, 16]), action_type=np.asarray([12, 9, 15, 18])):

    action = action_type[idx_a]
    idx_out = np.where(action==unique_type)



    return idx_out







##-----------troubleshoot RDF-----

def troubleshoot_cnt(RBF_o, RBF_n, num_features=7, check_idx=7):

    R1, R2 = RBF_o[0, :, :], RBF_n[0, :, :]
    cnt_count, r_count = R1.shape[0], R1.shape[1]/num_features
    #cnt is a matrix of size (400*1

    #RBF_o and RBF_n are two RBF matrices we're looking to compare


    R1 = R1.reshape(cnt_count, r_count, num_features)
    R2 = R2.reshape(cnt_count, r_count, num_features)

    #R1 -> R_o R2 ->
    #this code is to check whether
    # rbf = z*x*reshape
    r_o = R1[:, 0:check_idx, :]
    r_n = R2[:, 0:check_idx, :]
    #idx_mat = np.where(rbf > 0)





    return r_o, r_n



def new_df(df, cnt_o, opt_list = [12, 9, 15, 18], type_choose=22):

    df_cnt = df.loc[df['type'] == type_choose]

    cnt_mat = df_cnt.loc[:, ['x', 'y', 'z']].as_matrix()
    cnt_id = df_cnt.loc[:, ['id']].as_matrix()

    ###----------------------
    # choose dataframe with a different type
    df_fun = df.loc[df['type'].isin(opt_list)]
    fun_mat = df_fun.loc[:, ['x', 'y', 'z']].as_matrix()

    fun_type = df_fun.loc[:, ['type']].as_matrix()


    D = compute_distance(cnt_array=cnt_mat, cnt_other_array=fun_mat)

    if np.any(D < 2.0):

        #find where D is less than 2.0
        idx = np.where(D < 2.0)

        #index out cnt id where functionalization has taken place
        idx_cnt = idx[1]
        idx_fun = idx[0]

        #print "idx_cnt: ", idx_cnt
        #print "idx_fun: ", idx_fun

        cnt_choose = cnt_mat[idx_cnt, :]

        ##find with respect to original cnt
        min_idx = np.argmin(LA.norm((cnt_choose - cnt_o), axis=1))
        print "LA norm: ",  LA.norm((cnt_choose - cnt_o), axis=1)
        print "cnt_o: ", cnt_o
        print "cnt_choose: ", cnt_choose
        print "min_idx: ", min_idx

        cnt_choose = cnt_choose[min_idx, :]

    return cnt_choose