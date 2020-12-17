import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance
import mpi4py
import pandas as pd
import re
import glob
import math

import cluster_01
import feature_extract_01
from enum import Enum
from scipy import optimize



def read_bonds(chain_file):

    with open(chain_file) as f:
        text = f.readlines()

        count = 0
        skip_count = 0
        vel_count = 0
        bond_count = 0

        for line in text:

            count += 1

            if line.startswith('Atoms'):
                skip_count = count

            if line.startswith('Velocities'):
                vel_count = count - 3

            if line.startswith('Bonds'):
                bond_count = count

    print( "troubleshoot", skip_count, vel_count, bond_count )

    df_xyz = pd.read_table(chain_file, delim_whitespace=True, header=None, skiprows=skip_count,
                           nrows=(vel_count - skip_count))
    df_bond = pd.read_table(chain_file, delim_whitespace=True, header=None, skiprows=bond_count)

    return df_xyz, df_bond






def detect_fun_atoms(chain_file, type_choose=22, thresh=1.8):

    df_xyz, df_bond = read_bonds(chain_file=chain_file) #df_xyz -> coordinate system, df_bond -> topology of bonds
    header_list = ['id', 'molecule-id', 'type', 'x', 'y', 'z', 'nx', 'ny', 'nz']
    df_xyz.columns = header_list
    header_list = ['bond_id', 'bond_type', 'particle_1', 'particle_2']
    df_bond.columns = header_list


    df_cnt = df_xyz.loc[(df_xyz['type'] == type_choose)]
    df_other = df_xyz.loc[~(df_xyz['type'] == type_choose)]

    id_cnt = df_cnt.loc[:, ['id']].as_matrix().flatten()
    id_fun = df_other.loc[:, ['id']].as_matrix().flatten()
    type_fun = df_other.loc[:, ['type']].as_matrix().flatten()

    df_1 = df_cnt.loc[:, ['x', 'y', 'z']]
    df_2 = df_other.loc[:, ['x', 'y', 'z']]

    D, _, _  = cluster_01.compute_distance(df_1.as_matrix(), df_2.as_matrix())


    d_thr = D[D < thresh]


    if len(d_thr)>0:

        idx = np.where(D < thresh)
        id_fun = id_fun[idx[0]]
        id_cnt = id_cnt[idx[1]]
        type_choose = type_fun[idx[0]]

    else:

        id_fun = -1
        id_cnt = -1
        type_choose = -1



    ####now that we've selected the



    #this block to ensure that the extracted fun atoms is unique
    id_fun, unique_idx = np.unique(id_fun, return_index=True)


    if len(id_fun) > 1:
        type_choose = type_choose[unique_idx]
        id_cnt = id_cnt[unique_idx]


    #correct based on simulations

    if id_fun[0]!=-1:
        id_fun, id_cnt, type_choose = cluster_01.check_fun_atoms(fun_in=id_fun, cnt_in=id_cnt, df_xyz=df_xyz, type_in=type_choose)


    return d_thr, id_fun, id_cnt, type_choose, df_xyz, df_bond




def get_global_features(list_of_files, chain_files, N=5):

    global_mat = np.empty((0, 5)) #empty array with
    len_mat = np.zeros((5, ))

    for i in range(0, len(chain_files)):
        file_temp = sorted(glob.glob(chain_files[i]), key=key_func)

        for j in range(0, len(file_temp)):

            d_thr, id_fun, id_cnt, fun_type, df_xyz, df_bond = detect_fun_atoms(chain_file=file_temp[j])
            # get the center of the CNT circle
            center, R = cluster_01.get_circle_center(df=df_xyz)
            #print( "self.center: ", center )

            if len(id_fun) < 1:
                global_var = np.zeros((1, N))
            elif id_fun[0] != -1:
                global_var = cluster_01.global_features(fun_id=id_fun, cnt_id=id_cnt, center=center, df=df_xyz)
                global_var = global_var[None, :]
            else:
                global_var = np.zeros((1, N))

            #print( global_var )
            #print( global_var.shape )
            #print( global_mat.shape )



            global_mat = np.concatenate((global_mat, global_var), axis=0)

        len_mat[i] = len(global_mat)


        print( global_mat.shape )


    #xlink_mat = get_crosslink(list_of_files=list_of_files)
    #global_mat = np.concatenate((global_mat, xlink_mat), axis=1)



    return global_mat




def get_crosslink(list_of_files):

    xlink_mat = np.empty((0, 1))

    for i in range(0, len(list_of_files)):
        # print( list_of_files[i] )
        file_temp = sorted(glob.glob(list_of_files[i]), key=key_func)

        xlink_out = feature_extract_01.set_xlink_mat(file_name=list_of_files[i], total_length=int(len(file_temp)))

        xlink_mat = np.concatenate((xlink_mat, xlink_out), axis=0)

        print( xlink_out.shape )

    return xlink_mat






def key_func(x):
    nondigits= re.compile("\D")

    return int(nondigits.sub("", x))
