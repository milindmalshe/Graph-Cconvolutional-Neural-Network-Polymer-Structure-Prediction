import numpy as np

import hotspot_file_01
import multiprocessing
import pymp
import tensorflow as tf

import matplotlib.pyplot as plt

def compute_total_pe(df_list):

    pe = pymp.shared.array((len(df_list),), dtype='float64')


    with pymp.Parallel(multiprocessing.cpu_count()) as p:
        for t in p.range(0, len(df_list)):

            df = df_list[t]
            pe[t] = np.sum(np.asarray(df['c_poteng']), axis=0)


    T = np.arange(0, len(pe), 1, dtype=float)

    plt.plot(T, pe, 'k-')
    plt.show()


    return None


def compute_distance(cnt_array, cnt_other_array):

    dist_list = []
    points = cnt_array


    for j in range(0, len(cnt_other_array)):

        single_point = cnt_other_array[j, :]

        dist = np.sum(((points - single_point)**2), axis=1)
        dist = np.sqrt(dist)
        dist_list.append(dist)


    return np.asarray(dist_list)


def compute_local_pe(df_list, cnt_id, threshold=3.0):

    pe = pymp.shared.array((len(df_list),), dtype='float64')

    with pymp.Parallel(multiprocessing.cpu_count()) as p:
        for t in p.range(0, len(df_list)):

            df = df_list[t]

        #locate cnt atom
            df_select = df.loc[df['id']==cnt_id]
            mat_select = df_select.loc[:, ['x', 'y', 'z']].as_matrix()
            mat_other = df.loc[:, ['x', 'y', 'z']].as_matrix()
            pe_all = df.loc[:, ['c_poteng']].as_matrix()

            dist = compute_distance(mat_select, mat_other)

            idx = np.where(dist < threshold)

            pe_select = pe_all[idx[0]]

            pe[t] = np.sum(pe_select)

            #dist = compute_distance(mat_select, mat_other)

    T = np.arange(0, len(pe), 1, dtype=float)

    plt.plot(T, pe, 'k-')
    plt.show()

    return None




peratom_files = './cnt.pe.1126/dump.pe1epo.*'
param_list = {'channel_size': np.array([8.0, 8.0, 15.0]), 'delta': np.array([4.0, 4.0, 2.0]), 'tol': np.array([5.0, 5.0, 5.0]), 'Radius': 3.0, 'epsilon': 3.0}
#prop_lookup = ['c_ppa']
prop_lookup = ['c_poteng']
dprocess_1 = hotspot_file_01.dump_files(2, prop_lookup, 10, param_list, peratom_files)

compute_total_pe(df_list=dprocess_1.df_list)
compute_local_pe(df_list=dprocess_1.df_list, cnt_id=1978)