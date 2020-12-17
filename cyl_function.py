import numpy as np
import pandas as pd
import sys
import random

from scipy import special

file_to_read = str(sys.argv[1])

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


def compute_center(df_mat):

    x_min = np.min(df_mat[:, 0])
    x_max = np.max(df_mat[:, 0])
    y_min = np.min(df_mat[:, 1])
    y_max = np.max(df_mat[:, 1])

    center_cords = np.array([0.5*(x_min + x_max), 0.5*(y_min + y_max)])

    r1 = 0.5*(x_max- x_min)
    r2 = 0.5 * (y_max - y_min)

    r = 0.5*(r1 + r2)

    return center_cords, r



def compute_distance(cnt_array, cnt_other_array):

    dist_list = []
    points = cnt_array


    for j in range(0, len(cnt_other_array)):

        single_point = cnt_other_array[j, :]

        dist = np.sum(((points - single_point)**2), axis=1)
        dist = np.sqrt(dist)
        dist_list.append(dist)


    return np.asarray(dist_list)




def compute_cyl_function(file_to_read, r_min=0.0, r_max=25.0, r_trim=8.0, num_points=251, type_choose=22):


    df = read_datafile(file_to_read)
    df_cnt = df.loc[df['type']==type_choose]
    df_other = df.loc[df['type']!=type_choose]

    cnt_mat = df_cnt.loc[:, ['x', 'y']].as_matrix()
    center_cords, r = compute_center(cnt_mat)

    other_mat = df_other.loc[:, ['x', 'y']].as_matrix()
    #other_mat = df_other.loc[:, ['x', 'y']].as_matrix()
    dist = np.linalg.norm(other_mat - center_cords, axis=-1)

    dist_cnt = np.linalg.norm(cnt_mat - center_cords, axis=-1)



    #create histogram
    hist = np.histogram(dist, bins=np.linspace(r_min, r_max, num_points), density=True)
    hist_cnt = np.histogram(dist_cnt, bins=np.linspace(r_min, r_max, num_points), density=True)

    #trim off histogram from this block
    delta = (r_max - r_min)/(num_points - 1)
    idx_trim = int(r_trim/delta)
    hist_other = hist[0][0:idx_trim]
    hist_cnt = hist_cnt[0][0:idx_trim]

    return hist_other, hist_cnt



def truncate_hist(hist, delta, r_trim=5.0):
    idx_trim = int(r_trim / delta)
    hist_other = hist[0][0:idx_trim]

    return hist_other


def cyl_function02(file_to_read, r_min=0.0, r_max=50.0, num_points=251, type_choose=22):

    delta = (r_max - r_min) / (num_points - 1)

    df = read_datafile(file_to_read)
    df_cnt = df.loc[df['type'] == type_choose]
    cnt_mat = df_cnt.loc[:, ['x', 'y', 'z']].as_matrix()

    #array to compute KL


    df_C = df.loc[df['type'] == 12]
    #df_C = df.query('type == 12 | type == 2')
    C_mat = df_C.loc[:, ['x', 'y', 'z']].as_matrix()
    dist_C = compute_distance(cnt_mat, C_mat).flatten()
    hist_C = np.histogram(dist_C, bins=np.linspace(r_min, r_max, num_points), density=True)
    hist_C = truncate_hist(hist=hist_C, delta=delta)

    df_N = df.query('type == 9 | type == 15 | type == 23' )
    N_mat = df_N.loc[:, ['x', 'y', 'z']].as_matrix()
    dist_N = compute_distance(cnt_mat, N_mat).flatten()
    hist_N = np.histogram(dist_N, bins=np.linspace(r_min, r_max, num_points), density=True)
    hist_N = truncate_hist(hist=hist_N, delta=delta)



    df_O = df.loc[df['type'] == 18]
    O_mat = df_O.loc[:, ['x', 'y', 'z']].as_matrix()
    dist_O = compute_distance(cnt_mat, O_mat).flatten()
    hist_O = np.histogram(dist_O, bins=np.linspace(r_min, r_max, num_points), density=True)
    hist_O = truncate_hist(hist=hist_O, delta=delta)


    df_other = df.query('type != 12 & type !=9 & type !=15 & type !=23 & type !=18')
    other_mat = df_other.loc[:, ['x', 'y', 'z']].as_matrix()
    dist_other = compute_distance(cnt_mat, other_mat).flatten()
    hist_other = np.histogram(dist_other, bins=np.linspace(r_min, r_max, num_points), density=True)
    hist_other = truncate_hist(hist=hist_other, delta=delta)

    hist_mat = np.vstack((hist_C, hist_N, hist_O, hist_other))



    return hist_mat


def create_KL_mat(file_ref, file_in):

    hist_0 = cyl_function02(file_to_read=file_ref)
    hist_0[hist_0==0] = 1e-10

    hist_1 = cyl_function02(file_to_read=file_in)
    hist_1[hist_1 == 0] = 1e-10

    KL = np.sum(special.kl_div(hist_0, hist_1), axis=1)


    return KL

if __name__ == "__main__":

    hist_0, hist_0c = compute_cyl_function(file_to_read=file_to_read)
    hist_0[hist_0 == 0] = 1e-10
    hist_O1, hist_O1c = compute_cyl_function(file_to_read='data.O1epo')

    hist_C1, hist_C1c = compute_cyl_function(file_to_read='data.C1epo')
    hist_N1, hist_N1c = compute_cyl_function(file_to_read='data.N1epo')
    hist_Ns, hist_Nsc = compute_cyl_function(file_to_read='data.Ns1epo')
    #hist_C1[hist_C1 == 0] = 1e-10
    hist_ND, hist_NDc = compute_cyl_function(file_to_read='data.ND1epo')
    hist_CO, hist_COc = compute_cyl_function(file_to_read='data.CO1epo')
    hist_OO, hist_OOc = compute_cyl_function(file_to_read='data.OO1epo')


    hist_0c[hist_0c == 0] = hist_O1c[hist_O1c==0] = hist_C1c[hist_C1c==0] = hist_N1c[hist_N1c==0] = hist_Nsc[hist_Nsc==0] = 1e-10

    hist_NDc[hist_NDc == 0] = hist_COc[hist_COc == 0] =  hist_OOc[hist_OOc == 0] = 1e-10

    KL_0 = np.sum(special.kl_div(hist_0, hist_0))
    KL_O1 = np.sum(special.kl_div(hist_O1, hist_0))
    KL_C1 = np.sum(special.kl_div(hist_C1, hist_0))
    KL_N1 = np.sum(special.kl_div(hist_N1, hist_0))
    KL_Ns = np.sum(special.kl_div(hist_Ns, hist_0))
    KL_ND = np.sum(special.kl_div(hist_ND, hist_0))
    KL_CO = np.sum(special.kl_div(hist_CO, hist_0))
    e_OO = np.sum(special.kl_div(hist_OO, hist_0))

    #KL_O1 = np.sum(np.linalg.norm(hist_O1 - hist_0))
    #KL_C1 = np.sum(np.linalg.norm(hist_C1 - hist_0))
    #KL_ND = np.sum(np.linalg.norm(hist_ND - hist_0))
    #KL_CO = np.sum(np.linalg.norm(hist_CO - hist_0))
    #e_OO = np.sum(np.linalg.norm(hist_OO - hist_0))

    print KL_0
    print KL_O1
    print KL_C1
    print KL_N1
    print KL_Ns
    print KL_ND
    print KL_CO
    print e_OO

    KL_0c = np.sum(special.kl_div(hist_0, hist_0c))
    KL_O1c = np.sum(special.kl_div(hist_O1, hist_O1c))
    KL_C1c = np.sum(special.kl_div(hist_C1, hist_C1c))
    KL_N1c = np.sum(special.kl_div(hist_N1, hist_N1c))
    KL_Nsc = np.sum(special.kl_div(hist_Ns, hist_Nsc))
    KL_NDc = np.sum(special.kl_div(hist_ND, hist_NDc))
    KL_COc = np.sum(special.kl_div(hist_CO, hist_COc))
    e_OOc = np.sum(special.kl_div(hist_OO, hist_OOc))




    print "KL wrt the CNt distribution"
    print KL_0c
    print KL_O1c
    print KL_C1c
    print KL_N1c
    print KL_Nsc
    print KL_NDc
    print KL_COc
    print e_OOc

    KL_1 = create_KL_mat(file_ref='data.3rr1epo', file_in='data.O1epo')
    KL_2 = create_KL_mat(file_ref='data.3rr1epo', file_in='data.C1epo')
    KL_3 = create_KL_mat(file_ref='data.3rr1epo', file_in='data.N1epo')
    KL_4 = create_KL_mat(file_ref='data.3rr1epo', file_in='data.Ns1epo')
    KL_5 = create_KL_mat(file_ref='data.3rr1epo', file_in='data.ND1epo')

    KL_6 = create_KL_mat(file_ref='data.3rr1epo', file_in='data.CO1epo')
    KL_7 = create_KL_mat(file_ref='data.3rr1epo', file_in='data.OO1epo')

    print KL_1
    print KL_2
    print KL_3
    print KL_4
    print KL_5
    print KL_6
    print KL_7