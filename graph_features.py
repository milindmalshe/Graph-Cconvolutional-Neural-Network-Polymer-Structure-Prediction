###this function extracts the features necessary for GCN


import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance
import mpi4py
import pandas as pd
import re
import glob
import math


from enum import Enum

import cluster_01
import hotspot_file_01
import feature_extract_01
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import ML_graph01
from sklearn.model_selection import train_test_split

class graph_data():

    def __init__(self, MD_files, f_files, chain_files):


        self.MD_files = MD_files
        self. f_files = f_files
        self.chain_files = chain_files

        ####pick_a high number
        high_num = 100
        len_vec = np.zeros((len(MD_files),))

        #do a trail just to get the number of features
        trial_md = cluster_01.MD_model(MD_file=MD_files[2], f_value=f_files[2], chain_file=chain_files[2])
        trial_Z1 = trial_md.Z_1
        trial_Z2 = trial_md.Z_2
        trial_Z = trial_md.Z

        print( "len(MD files): ", len(MD_files))
        print( trial_Z1.shape)
        print( trial_Z2.shape)
        #now initialize the arrays
        Z_1 = np.zeros((len(MD_files), trial_Z1.shape[0], trial_Z1.shape[1]))
        Z_2 = np.zeros_like(Z_1)
        Z_0 = np.zeros((len(MD_files), high_num, trial_Z.shape[1], trial_Z.shape[2]))
        atomic_length = np.zeros((len(MD_files),))

        ###initialize array
        for idx in range(len(MD_files)):
            MD_test = cluster_01.MD_model(MD_file=MD_files[idx], f_value=f_files[idx], chain_file=chain_files[idx])
            Z_1[idx, :, :] = MD_test.Z_1
            Z_2[idx, :, :] = MD_test.Z_2
            len_vec[idx] = MD_test.Z.shape[0]
            Z_0[idx, 0:MD_test.Z.shape[0], :, :] = MD_test.Z
            atomic_length[idx] = MD_test.atomic_length

        max_len = np.amax(len_vec)

        self.Z_1 = Z_1
        self.Z_2 = Z_2
        self.Z_0 = Z_0
        self.atomic_length = atomic_length
        self.max_len = max_len






####global functions

def preprocessing_01(Z_1, Z_2, x_g, Y, alpha=0.25):


    max_z1 = np.amax(Z_1, axis=(0, 1))
    max_z2 = np.amax(Z_2, axis=(0, 1))

    z_1 = np.divide(Z_1, max_z1)
    z_2 = np.divide(Z_2, max_z2)

    x_1 = (1 - alpha)*z_1 + alpha*z_2

    ###block for maxpooling
    ##testing this out
    #x_1 = maxpool_graph(Z=x_1)

    x_gp = preprocessing.scale(x_g)

    Y_max = np.amax(Y)
    y = Y/Y_max


    return x_1, x_gp, y



def preprocessing_02(Z_1, Z_2):


    max_z1 = np.amax(Z_1, axis=(0, 1))
    max_z2 = np.amax(Z_2, axis=(0, 1))

    z_1 = np.divide(Z_1, max_z1)
    z_2 = np.divide(Z_2, max_z2)





    return z_1, z_2


def get_y(list_of_files, forcefile):

    targets = np.empty((0,))

    for i in range(0, len(forcefile)):
        # print list_of_files[i]
        file_temp = sorted(glob.glob(list_of_files[i]), key=key_func)

        raw_y = hotspot_file_01.get_raw_targets(forcefile[i])
        target_conf = feature_extract_01.configure_targets(file_list=file_temp, targets=raw_y)
        targets = np.append(targets, target_conf)


    return targets



def partition_data(X, Xg, Y, n, indices):

    #Xg is N


    N = int(n)

    X_t,  Xg_t, Y_t = X[:N, :, :],  Xg[:N, :], Y[:N]
    X_e, Xg_e, Y_e = X[N:, :, :],  Xg[N:, :], Y[N:]

    print( "indices: ", indices[N:])


    return X_t, Xg_t, Y_t, X_e, Xg_e, Y_e


def partition_data02(X, X2, Xg, Y, n, n2):

    #Xg is N


    N = int(n)
    N2 = int(n2)

    X_t,  X_t2, Xg_t, Y_t = X[:N, :, :],  X2[:N, :, :], Xg[:N, :], Y[:N]
    X_v, X_v2, Xg_v, Y_v = X[N:N2, :, :], X2[N:N2, :, :], Xg[N:N2, :], Y[N:N2]
    X_e, X_e2, Xg_e, Y_e = X[N2:, :, :],  X2[N2:, :, :], Xg[N2:, :], Y[N2:]


    return X_t, X_t2, Xg_t, Y_t, X_v, X_v2, Xg_v, Y_v, X_e, X_e2, Xg_e, Y_e


##--------------------------


def key_func(x):
    nondigits= re.compile("\D")

    return int(nondigits.sub("", x))



def shuffle_data(x_1, x_g, y):


    indices = np.random.permutation(x_1.shape[0])

    x_o, x_g, y_o = x_1[indices, :, :], x_g[indices, :], y[indices]


    return x_o, x_g, y_o, indices


def shuffle_data02(x_1, x_2, x_g, y):


    indices = np.random.permutation(x_1.shape[0])

    x_o, x_o2, x_g, y_o = x_1[indices, :, :], x_2[indices, :, :], x_g[indices, :], y[indices]


    return x_o, x_o2, x_g, y_o, indices


def split_data(x_1, x_g, y, n=200):

    y_e = np.zeros((20, ))

    while np.any(y_e < 0.025):

        x_1a, x_ga, y_a, idx = shuffle_data(x_1=x_1, x_g=x_g, y=y)
        x_t, x_gt, y_t, x_e, x_ge, y_e = partition_data(X=x_1a, Xg=x_ga, Y=y_a, n=n, indices=idx)



    return x_t, x_gt, y_t, x_e, x_ge, y_e


def split_data02(x_1, x_2, x_g, y, n=180, n2=200):

    y_e = np.zeros((len(y) - n2,))

    while np.any(y_e < 0.025):

        x_1, x_2, x_g, y, indices = shuffle_data02(x_1=x_1, x_2=x_2, x_g=x_g, y=y)
        x_t1, x_t2, x_gt, y_t, x_v, x_v2, x_gv, y_v, x_e, x_e2, x_ge, y_e = partition_data02(X=x_1, X2=x_2, Xg=x_g, Y=y, n=n, n2=n2)
        idx_t, idx_e = indices[:n], indices[n:n2]

        print( "idx_t: ", idx_t)
        print( "idx_e: ", idx_e)



    return x_t1, x_t2, x_gt, y_t, x_v, x_v2, x_gv, y_v, x_e, x_e2, x_ge, y_e


def maxpool_graph(Z, alpha=0.0, node_list=np.asarray([0, 1, 2, 3, 5, 10, 15, 20, 30, 40])): #50, 60, 70, 80, 90, 100

    Z_out = np.zeros((Z.shape[0], len(node_list)-1, Z.shape[2]))

    print( "Z shape: ", Z.shape)
    for i in range(0, Z_out.shape[1]):
        z_1 = np.mean((Z[:, node_list[i]:node_list[i+1], :]), axis=1)
        z_2 = np.amax((Z[:, node_list[i]:node_list[i+1], :]), axis=1)

        Z_out[:, i, :] = (1-alpha)*z_1 + alpha*z_2



    return Z_out





#####evaluation functions######

def evaluate_prob_graph(x_1, x_g, y, max_y, N_sample=100):


    #performance metrics


    x_t, x_gt, y_t, x_e, x_ge, y_e = split_data(x_1=x_1, x_g=x_g, y=y)

    yp_s = np.zeros((N_sample, len(y_e)))
    ye_s = np.zeros_like(yp_s)
    rel_err = np.zeros((N_sample,))
    abs_err = np.zeros_like(rel_err)
    r2_val = np.zeros_like(rel_err)

    RE_d = np.zeros_like(rel_err)
    AE_d = np.zeros_like(abs_err)
    R2_d = np.zeros_like(r2_val)

    for i in range(N_sample):
        x_t, x_gt, y_t, x_e, x_ge, y_e = split_data(x_1=x_1, x_g=x_g, y=y)
        model_gcn = ML_graph01.fit_model01B(X1=x_t, Xg=x_gt, Y=y_t)
        #model_gcn = ML_graph01.fit_model021(X1=x_t, Xg=x_gt, Y=y_t)
        y_p = model_gcn.predict([x_e, x_ge])

        ye_s[i, :] = max_y * y_e
        yp_s[i, :] = max_y * y_p.flatten()



        rel_err[i] = np.sqrt(((max_y * y_e - max_y * y_p.flatten()) ** 2).mean()) / np.sqrt(
        ((max_y * y_e) ** 2).mean())
        abs_err[i] = np.sqrt(((max_y * y_e - max_y * y_p.flatten()) ** 2).mean())
        r2_val[i] = r2_score(y_e, y_p.flatten())

        RE_d[i], AE_d[i], R2_d[i] = outlier_accuracy(y_e=y_e, y_p = y_p.flatten(), max_y=max_y)

        print( "rel_err", rel_err[i])
        print( "abs_err: ", abs_err[i])
        print( "R2_val: ", r2_val[i])
        print( "y_e: ", max_y * y_e)
        print( "y_p: ", max_y*y_p.flatten())

        print( "RE_d: ", RE_d[i])
        print( "AE_d: ", AE_d[i])
        print( "R2_d: ", R2_d[i])


    np.save('ye_s_012B', ye_s)
    np.save('yp_s_012B.npy', yp_s)
    np.save('rel_err_012B.npy', rel_err)
    #np.save('abs_err_B.npy', abs_err)
    #np.save('r2_s_B.npy', r2_val)

    #np.save('rel_err_B.npy', RE_d)
    #np.save('abs_err_B.npy', AE_d)
    #np.save('r2_s_B.npy', R2_d)

    print( "rel error: ", np.mean(rel_err), np.std(rel_err))
    print( "abs error: ", np.mean(abs_err), np.std(abs_err))
    print( "R2: ", np.mean(r2_val), np.std(r2_val))

    print( "***outlier removed: ")
    print( "rel error: ", np.mean(RE_d), np.std(RE_d))
    print( "abs error: ", np.mean(AE_d), np.std(AE_d))
    print( "R2: ", np.mean(R2_d), np.std(R2_d))

    return None



def outlier_accuracy(y_e, y_p, max_y):

    diff = np.abs(y_e - y_p)
    idx_to_del = np.argmax(diff)
    ye_d = np.delete(y_e, idx_to_del)
    yp_d = np.delete(y_p, idx_to_del)

    rel_err = np.sqrt(((max_y * ye_d - max_y * yp_d) ** 2).mean()) / np.sqrt(
        ((max_y * ye_d) ** 2).mean())
    abs_err = np.sqrt(((max_y * ye_d - max_y * yp_d) ** 2).mean())
    r2_val = r2_score(ye_d, yp_d)






    return rel_err, abs_err, r2_val
