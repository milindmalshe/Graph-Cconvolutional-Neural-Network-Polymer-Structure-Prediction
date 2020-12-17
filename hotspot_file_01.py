import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance
import mpi4py
import pandas as pd
#import atomman as am
from lammps import lammps
from itertools import islice
import os
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

from keras import backend as K

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

from math import sqrt

#from matplotlib.patches import Circle
#from matplotlib import cm

from sklearn.decomposition import PCA

import math
import re
import glob

#import pymp for parallel processing
import pymp

#import other files
import feature_extract_01
import ML_functions01
import cluster_01
import cluster_01B
import vis_modules


###-----------files to extract structures

class MD_models():

    def __init__(self, list_of_files, forcefile, str, cluster_file):

        choose = 1 #variable to select whether we want to save

        cov_var = 0

        if choose == 1:

            self.prop_str = str #The list of properties we want to read
            self.total_files = [] #instantiate an empty file to declare total files.
            self.targets = np.empty((0,)) #instantiate the targets
            #idx_choose = [2,4,6]

            #global_mat
            self.global_mat = np.empty((0, 1)) #instantiate global variables


            #need to read in the entries in each list
            size_mat = np.zeros((len(list_of_files), )) #need to keep track how many functionalized models for each neat file


            for i in range(0, len(list_of_files)):
            #for i in range(0, 1):
                #print list_of_files[i]
                file_temp = sorted(glob.glob(list_of_files[i]), key=key_func)  #sort the files chronologically
                self.total_files = self.total_files + file_temp #aggregating files


                #this block of code aggregates all the targets, i.e. the data from forcefile
                raw_y = get_raw_targets(forcefile[i]) #extracting the raw targets -> refer to the get_raw_targets
                target_conf = feature_extract_01.configure_targets(file_list=file_temp, targets=raw_y)  #check [4.2] in main doc
                self.targets = np.append(self.targets, target_conf) #append to build dataset



                ####

                size_mat[i] = len(file_temp) #get the total number of functionalized models for each neat model

                xlink_mat = feature_extract_01.set_xlink_mat(file_name=list_of_files[i], total_length=int(size_mat[i]))

                ## cov_var
                cov_mat = np.zeros((int(size_mat[i]), 1)) #cov_mat is a binary feature simply to indicate if a model is functionalized or not
                cov_mat[0, :] = 1


                #join mat is xlink mat + other features we might want to include
                join_mat = xlink_mat
                self.global_mat = np.concatenate((self.global_mat, join_mat), axis=0) #appending global features


                ##set xlink_mat



            var_mat = cluster_01B.get_global_features(list_of_files=list_of_files, chain_files=cluster_file)   #this file gets the global features

            #self.global_mat = np.concatenate((self.global_mat, var_mat, CL), axis=1)
            self.global_mat = np.concatenate((self.global_mat, var_mat), axis=1) #concatenating glob mat with respect to global features column-wise

            other_mat = preprocessing.scale(self.global_mat)  #normalize the global features

            #shuffle by file list
            self.total_files, self.targets, crosslink_mat, indices = feature_extract_01.shuffle_by_list02(file_list=self.total_files, Y=self.targets, other_mat=other_mat) #shuffles the input data for train/test split
            #np.save('file_indices.npy', indices)


            self.df_list, z_list, z_coords = read_MD_model(self.total_files) #read input model
            #crosslink_mat = np.zeros((len(self.df_list), 6))

            #NOTE: these commendted lines for troubleshooting

            #self.RBF, r_v = feature_extract_01.CNT_atoms(self.df_list[0], z_list[0])

            #for i in range(0, len(self.df_list)):
                #RBF_g = feature_extract_01.global_features(df=self.df_list[i], z_len=z_list[i])
                #print RBF_g[1:]
                #crosslink_mat[i, :] = RBF_g[1:]


            #normalize cross_linkmat


            #print self.RBF.shape
            #r_d, self.RBF_d = feature_extract_01.discretize_features(self.RBF)

            #for i in range(0, 7):
                #feature_extract_01.plot_features(r_v, self.RBF[100, :, :], i)



            #for i in range(0, 7):
                #feature_extract_01.plot_features02(r_v, self.RBF[256, :, :], r_d, self.RBF_d[256, :, :], i)

            ###block of code for the new RBF
            #self.RBF, self.R_v = feature_extract_01.RBF_Setter(df_list=self.df_list, z_list=z_list, total_files=self.total_files)
            self.RBF, self.R_v = feature_extract_01.RBF_Setter02(df_list=self.df_list, z_list=z_list) #RBF_setter


            CNT_data, RBF_data, y_out, center_list = feature_extract_01.transfor_rot(df_list=self.df_list, RBF_in=self.RBF, targets=self.targets, z_cords=z_coords)
            #CNT_data_f, RBF_data_f , _, _ = feature_extract_01.transfor_rot(df_list=self.df_list, RBF_in=self.RBF_f, targets=self.targets, z_cords=z_coords)


            #print len(RBF_data)
            #print np.where(RBF_data_f[0] > 0)

            #crosslink_mat = (np.repeat(crosslink_mat[:, :, None], repeats=int(len(y_out)/crosslink_mat.shape[0]), axis=2)).reshape(-1, crosslink_mat.shape[1])
            crosslink_mat = feature_extract_01.repeat_global_features(crosslink_mat)

            max_y = np.amax(y_out) #taking max out
            y_out = y_out/np.amax(y_out) #noralizing targets
            RBF_1, RBF_2, bound_mat = feature_extract_01.discretize_2D(cnt_array=CNT_data, RBF_array=RBF_data)
            #RBF_1f, RBF_2f, _ = feature_extract_01.discretize_2D(cnt_array=CNT_data, RBF_array=RBF_data_f, ts_bool=True)



            ###load model
            #AE block
            #model_AE = ML_functions01.load_NN('model_ae02.h5')

            x1 = RBF_1.copy()
            x2 = RBF_2.copy()
            #x3 = RBF_1f.copy()
            #x4 = RBF_2f.copy()
            # x2 =  RBF_2.copy()


            #this block saves data for testing the local state neural network model
            #np.save('df_list01.npy', self.df_list)
            #np.save('file_list01.npy', np.asarray(self.total_files))
            #np.savez('cnt_data01.npz', *CNT_data)
            #np.savez('atomic_rbf01.npz', *RBF_data)
            #np.savez('atomic_rbf01_f.npz', *RBF_data_f)
            #np.save('bound_mat01.npy', bound_mat)
            #np.save('rbf_coarse_101.npy', x1)
            #np.save('rbf_coarse_201.npy', x2)



            #RBF_1, RBF_2, cols_dropped = feature_extract_01.RBF_eliminator(RBF_1=RBF_1, RBF_2=RBF_2)
            #X_t1, X_t2, Y_t, X_e1, X_e2, Y_e = feature_extract_01.partition_data(X=RBF_1, X2=RBF_2, Y=y_out, f=0.5)

            #X_t1, X_t2, Y_t, X_e1, X_e2, Y_e = feature_extract_01.partition_data02(X=RBF_1, X2=RBF_2, Y=y_out, n=70)
            ##X_t1, X_t2, Y_t, X_e1, X_e2, Y_e, X_v1, X_v2, Y_v = feature_extract_01.partition_data03(X=RBF_1, X2=RBF_2, Y=y_out, n=48, n2=53) #replacement

        #model with global features

            ####save file so we can
            np.save('X1.npy', x1)
            np.save('X2.npy', x2)
            np.save('Xlink.npy', crosslink_mat)
            np.save('Y_out.npy', y_out)


            X_t1, X_t2, Xg_t, Y_t, X_e1, X_e2, Xg_e, Y_e = feature_extract_01.partition_data02B(X=x1, X2=x2, Xg=crosslink_mat, Y=y_out, n=200)
            #X_t1, X_t2, Y_t = feature_extract_01.shuffle_data(X_t1, X_t2, Y_t)

            #X_t1, X_t2 -> training data for discretized RDF
            #Xg_t -> traiing data -> global features
            #Y_t -> Training labels
            #Subscript '_e' -> corresponds to test data

            X_t1, X_t2, Xg_t, Y_t = feature_extract_01.shuffle_data02B(X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t)

            #X_t1, X_t2, Y_t, X_e1, X_e2, Y_e = feature_extract_01.shuffle_by_og(X1=RBF_1, X2=RBF_2, Y=y_out, n=15)
            #X_t1, X_t2, Y_t = feature_extract_01.shuffle_data(X_t1, X_t2, Y_t)


            #save R_v, col_droppped
            #np.save('col_dropped.npy', cols_dropped)
            np.save('R_v.npy', self.R_v)
            np.save('max_y.npy', max_y)
            np.save('x1_6.npy', x1)

            #training data
            np.save('RBF1_6.npy', RBF_1)
            np.save('X_t1_6.npy', X_t1)
            np.save('X_t2_6.npy', X_t2)
            np.save('Xg_t_6.npy', Xg_t)
            np.save('Y_t_6.npy', Y_t)

            #valdiation data
            #np.save('X_v1_5.npy', X_v1)
            #np.save('X_v2_5.npy', X_v2)
            #np.save('Xg_v_5.npy', Xg_v)
            #np.save('Y_v_5.npy', Y_v)

            #test error
            np.save('X_e1_6.npy', X_e1)
            np.save('X_e2_6.npy', X_e2)
            np.save('Xg_e_6.npy', Xg_e)
            np.save('Y_e_6.npy', Y_e)


            model_2 = ML_functions01.fit_model02B(RBF=x1, X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t)


            ###block to replace the past three lines of code
            #model_2 = ML_functions01.val_check01(x1=x1, x2=x2, crosslink_mat=crosslink_mat, y_out=y_out)
            #model_2 = ML_functions01.fit_model02C(RBF=RBF_1, X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t, X1_v=X_v1, X2_v=X_v2, Xg_v=Xg_v, Y_v=Y_v)

            #this block of code is to predict the model and
            Y2 = model_2.predict([X_e1, X_e2, Xg_e])

            e_val2 = sqrt(mean_squared_error(Y_e, Y2.flatten()))
            R2_val2 = r2_score(Y_e.flatten(), Y2)

            print( e_val2)
            print( e_val2 * max_y)
            print( R2_val2)
            print( Y2.flatten())
            print( Y_e)

            W = ML_functions01.get_model_weights(model_2)
            model_2.save_weights('W_2a.h5')

         #np.save('W0_b.npy', W[0])

            Y1, Y2 =feature_extract_01.pick_ogvalues(Y_e, Y2.flatten()) #this pikcs the og values from the predictions
            e_val3 = sqrt(mean_squared_error(Y1, Y2))
            R2_val3 = r2_score(Y1, Y2)

            print( e_val3)
            print( e_val3 * max_y)
            print( R2_val3)
            print( Y1*max_y)
            print( Y2*max_y)

        elif choose == 2:

            #this block of code
            ##partition data by loading variables
            x1 = np.load('X1.npy')
            x2 = np.load('X2.npy')
            crosslink_mat = np.load('Xlink.npy')
            y_out = np.load('Y_out.npy')


            X_t1, X_t2, Xg_t, Y_t, X_e1, X_e2, Xg_e, Y_e = feature_extract_01.split_reuse_data(x1=x1, x2=x2, xg=crosslink_mat, y=y_out, n1=200) #put this back in if it doesn't work


            X_t1, X_t2, Xg_t, Y_t = feature_extract_01.shuffle_data02B(X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t)
            model_2 = ML_functions01.fit_model02B(RBF=x1, X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t) #put this back in
            #model_2 = ML_functions01.fit_model02C(RBF=x1, X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t, X1_v=X_v1, X2_v=X_v2, Xg_v=Xg_v, Y_v=Y_v)

            Y2 = model_2.predict([X_e1, X_e2, Xg_e])

            e_val2 = sqrt(mean_squared_error(Y_e, Y2.flatten()))
            R2_val2 = r2_score(Y_e.flatten(), Y2)

            print( e_val2)
            print( e_val2 * 0.410)
            print( R2_val2)
            print( Y2.flatten())
            print( Y_e)

            Y1, Y2 = feature_extract_01.pick_ogvalues(Y_e, Y2.flatten())
            e_val3 = sqrt(mean_squared_error(Y1, Y2))
            R2_val3 = r2_score(Y1, Y2)
            rel_err = np.sqrt(((Y1 - Y2) ** 2).mean()) / np.sqrt(((Y1) ** 2).mean())

            mu = np.mean(Y1)
            sigma = np.std(Y1)

            norm_y1 = (Y1 - mu)/sigma
            norm_y2 = (Y2 - mu)/sigma

            R2_val4 = r2_score(norm_y1, norm_y2)

            print( e_val3)
            print( e_val3 * 0.410)
            print( R2_val3)
            print( rel_err)
            print( "R2 normalized: ", R2_val4)
            print( Y1 * 0.410)
            print( Y2 * 0.410)

            model_2.save('model_2.h5')
            np.save('X_13.npy', X_e1)
            np.save('X_23.npy', X_e2)
            np.save('X_g3.npy', Xg_e)
            grad_tensor = rdf_saliency(model_pick=model_2, x_e1=X_e1, x_e2=X_e2, x_ge=Xg_e)

            print( "grad_tensor: ")
            print( grad_tensor)
            evaluate_prob_cnn(x_1=x1, x_2=x2, x_g=crosslink_mat, y=y_out, max_y=0.410, n1=180, n2=200)


        else:

            max_y = np.load('max_y.npy')
            x1 = np.load('x1_6.npy')
            # training data
            RBF_1 = np.load('RBF1_6.npy')
            X_t1 = np.load('X_t1_6.npy')
            X_t2 = np.load('X_t2_6.npy')
            Xg_t = np.load('Xg_t_6.npy')
            Y_t = np.load('Y_t_6.npy')
            Y_t = np.load('Y_t_6.npy')

            # valdiation data
            #X_v1 = np.load('X_v1_6.npy')
            #X_v2 = np.load('X_v2_6.npy')
            #Xg_v = np.load('Xg_v_6.npy')
            #Y_v = np.load('Y_v_6.npy')

            # test error
            X_e1 = np.load('X_e1_6.npy')
            X_e2 = np.load('X_e2_6.npy')
            Xg_e = np.load('Xg_e_6.npy')
            Y_e = np.load('Y_e_6.npy')

            # model_2 = ML_functions01.fit_model02B(RBF=RBF_1, X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t)
            model_2 = ML_functions01.fit_model02B(RBF=x1, X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t)
            #model_2 = ML_functions01.fit_model02C(RBF=RBF_1, X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t, X1_v=X_v1, X2_v=X_v2,
            f = K.function([model_2.input[0], model_2.input[1], model_2.input[2], K.learning_phase()],
                           [model_2.layers[-1].output])

            #print "Model input shape: "
            #print model_2.input.shape
            #prediction = model_2.predict()[0]

            #                                      Xg_v=Xg_v, Y_v=Y_v)
            Y2 = model_2.predict([X_e1, X_e2, Xg_e])

            e_val2 = sqrt(mean_squared_error(Y_e, Y2.flatten()))
            R2_val2 = r2_score(Y_e.flatten(), Y2)

            print( e_val2)
            print( e_val2 * max_y)
            print( R2_val2)
            print( Y2.flatten())
            print( Y_e)



            W = ML_functions01.get_model_weights(model_2)
            model_2.save_weights('W_2a.h5')

            print( W[0][:, :, :, 0])
            print( W[0][:, :, :, 1])
            print( "second layer weights")
            print( W[2])

            np.save('W0_d.npy', W[0])
            np.save('W1_d.npy', W[1])
            np.save('W2_d.npy', W[2])



            Y1, Y2 = feature_extract_01.pick_ogvalues(Y_e, Y2.flatten())
            e_val3 = sqrt(mean_squared_error(Y1, Y2))
            R2_val3 = r2_score(Y1, Y2)

            print( e_val3)
            print( e_val3 * max_y)
            print( R2_val3)
            print( Y1 * max_y)
            print( Y2 * max_y)

            x_e1, x_e2, x_g = feature_extract_01.pick_og_features(X_e1, X_e2, Xg_e)
            mu_v = np.zeros((len(x_e1), ))
            sigma_v = np.zeros_like(mu_v)







            #np.save('Y_e2.npy', Y_e)


        #np.save('Y_p2.npy', Y.flatten())

        #print RBF_1.shape

        #need a block here to sort out




####----_Second files for dump processes

class dump_files():

    def __init__(self, id, str, total_hotspot, convolve_params, *args):
        self.process_id = id #ID of the dump files
        self.prop_str = str #The list of properties we want to read
        self.peratom_files = sorted(glob.glob(args[0]), key=key_func)

        self.unique_type = np.array([1, 2, 3, 4, 5, 9, 11, 12, 13, 15, 16, 18, 20])


        if len(args) > 1:
            self.local_files = sorted(glob.glob(args[1]), key=key_func)

        self.df_list, self.array_list, self.pe_hotspot_list, self.bound_list = select_by_property(file_list=self.peratom_files,
                                                                       prop_name=self.prop_str, N=total_hotspot, sort_choose=True)
        self.convolve_size = convolve_params['channel_size']
        self.stride = convolve_params['delta']
        self.box_tol = convolve_params['tol']
        self.Radius = convolve_params['Radius']
        self.epsilon = convolve_params['epsilon']
        #find hotspots by convolving



    def prop_end_timesteps(self):

        end_diff = self.property_list[len(self.property_list) - 1] - self.property_list[0]

        return end_diff





    def add_del_files(self, delepx_files, delcnt_files):
        self.delepx_files = sorted(glob.glob(delepx_files), key=key_func)
        self.delcnt_files = sorted(glob.glob(delcnt_files), key=key_func)

        self.df_delepx_list, self.delepx_array_list, _, _ = select_by_property(
            file_list=self.delepx_files,
            prop_name=self.prop_str, N=10, sort_choose=False)

        self.df_delcnt_list, self.delcnt_array_list, _, _ = select_by_property(
            file_list=self.delcnt_files,
            prop_name=self.prop_str, N=10, sort_choose=False)

        return None


    def add_group_files(self, alkane_file, fun_file, bondedCNT_file):

        self.alkane_files = sorted(glob.glob(alkane_file), key=key_func)
        self.fun_file = sorted(glob.glob(fun_file), key=key_func)
        self.bondedcnt_file = sorted(glob.glob(bondedCNT_file), key=key_func)


        self.df_alkane_list, self.alkane_array_list, _, _ = select_by_property(file_list=self.alkane_files, prop_name=[],
                                                                         N=10, sort_choose=False)

        self.df_fun_list, self.fun_array_list, _, _ = select_by_property(file_list=self.fun_file,
                                                                               prop_name=[],
                                                                               N=10, sort_choose=False)

        self.df_bondedcnt_list, self.bondedcnt_array_list, _, _ = select_by_property(file_list=self.bondedcnt_file,
                                                                         prop_name=[],
                                                                         N=10, sort_choose=False)
        del self.alkane_array_list[0:2]
        print( len(self.alkane_files))
        print( self.alkane_array_list)
        print( len(self.fun_file))
        print( len(self.bondedcnt_file))

        t_max = len(self.alkane_array_list)

        dist_alkaneC = np.zeros((t_max,))
        dist_cnt = np.zeros_like(dist_alkaneC)

        for t in range(0, t_max):
            pointC = self.alkane_array_list[t][0, 2:5]
            pointO = self.fun_array_list[t][0, 2:5]
            pointCNT = self.bondedcnt_array_list[t][0, 2:5]
            dist_alkaneC[t] = np.linalg.norm(pointC - pointO)
            dist_cnt[t] = np.linalg.norm(pointCNT - pointO)

        self.d_alkaneC = dist_alkaneC
        self.d_cnt = dist_cnt


        return None


    def extract_cnt_velocity(self, graph_dict):

        cnt_vz = pymp.shared.array((len(self.df_list), ), dtype='float64')
        other_vz = pymp.shared.array((len(self.df_list),), dtype='float64')
        cnt_disp = pymp.shared.array((len(self.df_list), ), dtype='float64')

        #Get data from dict

        f_min = graph_dict['f_min']
        f_max = graph_dict['f_max']
        t_max = graph_dict['t_max']
        res = graph_dict['res']  # res -> frequency of timesteps with which we have dump files


        #Extracting out indexes to draw graphs
        f_idx1 = graph_dict['f_idx1']
        f_idx2 = graph_dict['f_idx2']

        with pymp.Parallel(multiprocessing.cpu_count()) as p:
            for t in p.range(0, len(self.df_list)):
                df_tmp = self.df_list[t]
                df_cnt = df_tmp[df_tmp['type']==22]
                df_other =df_tmp[df_tmp['type']!=22]
                df_cnt_v = df_cnt['vz']
                df_other_v = df_other['vz']
                df_cnt_disp = df_cnt['c_dpa[4]']

                cnt_vz[t] = np.mean(np.asarray(df_cnt_v), axis=0)
                other_vz[t] = np.mean(np.asarray(df_other_v), axis=0)
                cnt_disp[t] = np.mean(np.asarray(df_cnt_disp), axis=0)


        self.cnt_mean_velocity = cnt_vz

        #stride = 20
        stride = 5
        dt = 0.1
        #v_diff = np.gradient(self.cnt_mean_velocity, 0.1*np.arange(0, len(self.cnt_mean_velocity)))
        self.cnt_mean_velocity = self.cnt_mean_velocity[:, None]
        self.cnt_mean_velocity = reduce_by_mean(self.cnt_mean_velocity, stride)

        #self.cnt_mean_acc = reduce_by_mean(v_diff[:, None], stride)
        self.cnt_mean_acc= np.gradient(self.cnt_mean_velocity.flatten())
        print( "mean", np.mean(self.cnt_mean_velocity))

        print( self.cnt_mean_acc)

        cnt_disp = cnt_disp[:, None]
        #cnt_disp = reduce_by_mean(cnt_disp, stride)


        f = f_min + ((f_max - f_min)/t_max)*(dt*stride*res)*np.arange(0, len(self.cnt_mean_velocity))

        #print "f: "
        #print f

        #extracting out indices for plots
        idx_1 = int(len(self.cnt_mean_velocity)*(f_idx1)/(f_max - f_min))
        idx_2 = int(len(self.cnt_mean_velocity) * (f_idx2) / (f_max - f_min))

        plt.plot(f[idx_1:idx_2], self.cnt_mean_velocity[idx_1:idx_2], 'ko')
        plt.xlabel('Pull-out force per atom (kcal/mol-$\AA$)', fontsize=14)
        plt.ylabel('Mean Velocity ($\AA$/fs)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        #plt.savefig('pullout_3rr5epo_00031.eps', bbox_inches="tight")
        #plt.savefig('fig_1A.pngf')
        plt.show()

        np.save('f_3rr4epo_00023.npy', f[idx_1:idx_2])
        np.save('v_3rr4epo_00023.npy', self.cnt_mean_velocity[idx_1:idx_2])
        #np.save('d_00004.npy', cnt_disp[idx_1:idx_2])

        plt.plot(f[idx_1:idx_2], cnt_disp[idx_1:idx_2])
        plt.xlabel('Pull-out force per atom (kcal/mol-A)')
        plt.ylabel('Cummulative displacement (A)')
        plt.savefig('fig_1B.eps')
        plt.show()

        #print f.shape
        #print self.cnt_mean_acc

        plt.plot(f[idx_1:idx_2], self.cnt_mean_acc[idx_1:idx_2])
        plt.xlabel('Pull-out force per atom (kcal/mol-A)')
        plt.ylabel('Acceleration (A/fs.squared)')
        #plt.savefig('fig_1B.eps')
        plt.show()

        if hasattr(self, 'd_alkaneC'):
            plt.plot(f[idx_1:idx_2], self.d_alkaneC[idx_1:idx_2])
            plt.xlabel('Pull-out force per atom (kcal/mol-$\AA$)')
            plt.ylabel('Distance between nitrogen and aromatic carbon')
            plt.savefig('fig_R-O.eps', format='eps', dpi=1000)
            plt.show()

        if hasattr(self, 'd_cnt'):
            plt.plot(f[idx_1:idx_2], self.d_cnt[idx_1:idx_2])
            plt.xlabel('Pull-out force per atom (kcal/mol-$\AA$)')
            plt.ylabel('Distance between CNT and oxygen ')
            plt.savefig('fig_CNT-O.svg')
            plt.show()

        #check bonds


        return None

    def plot_interaction_energy(self, graph_dict, filename):


        E2 = pymp.shared.array((len(self.df_list), ), dtype='float64')
        E3 = pymp.shared.array((len(self.df_list), ), dtype='float64')

         #graph_list = {'f_min': 0, 'f_max': 0.025, 't_max': 500000, 'res': 500}
        f_min = graph_dict['f_min']
        f_max = graph_dict['f_max']
        t_max = graph_dict['t_max']
        res = graph_dict['res'] #res -> frequency of timesteps with which we have dump files

        with pymp.Parallel(multiprocessing.cpu_count()) as p:
            for t in p.range(0, len(self.df_list)):
                df_1 = self.df_list[t]
                E1[t] = np.sum(np.asarray(df_1[self.prop_str]), axis=0)

                df_2 = self.df_delcnt_list[t]
                E2[t] = np.sum(np.asarray(df_2[self.prop_str]), axis=0)

                df_3 = self.df_delepx_list[t]
                E3[t] = np.sum(np.asarray(df_3[self.prop_str]), axis=0)


        I = E1 - E2 - E3
        I = I[:, None]
        stride = 2
        dt = 0.1
        I2 = reduce_by_mean(I, stride)
        plt.plot(f_min + ((f_max - f_min)/t_max)*(dt*stride*res)*np.arange(0, len(I2)), I2, 'ko')
        plt.xlabel('Pull-out force per atom (kcal/mol-A)')
        plt.ylabel('Interaction Energy (kcal/mol)')
        plt.savefig(filename)
        plt.show()


        return None




    def convolve_under_shear(self):

        stride = self.stride
        temp_channel_init = self.convolve_size.copy()
        max_n_points = 10
        n_surf = 4 #number of surfaces to convolve over

        max_t = len(self.array_list)
        bound_old = self.bound_list[0].copy()

        print( "Max_t: ", max_t)

        #time period to select
        t_p = 1
        def_temp = 0
        list_count = 0  # counter to compute diff_mat

        property_map_list = []
        loc_list = []
        idx_list_list = []
        self.cnt_bound_list = []


        #with pymp.Parallel(24) as p:
        for t in range(0, t_p):

            atom_new = self.array_list[t].copy()
            bound_new = self.bound_list[t].copy()

            #delatom import
            #delcnt_new = self.df_delcnt_list[t].copy()
            #delepx_new = self.df_delepx_list[t].copy()

            delta_x = (bound_new[0, 1] - bound_new[0, 0]) - (bound_old[0, 1] - bound_old[0, 0])
            delta_y = (bound_new[1, 1] - bound_new[1, 0]) - (bound_old[1, 1] - bound_old[1, 0])
            delta_z = (bound_new[2, 1] - bound_new[2, 0]) - (bound_old[2, 1] - bound_old[2, 0])

            temp_channel = temp_channel_init + np.asarray([delta_x, delta_y, delta_z])

            #cnt_bounds, box_bound = extract_cnt_coords(self.df_list[t], bound_new, self.box_tol, self.prop_str)
            #now let's start getting the cnt atoms
            self.cnt_bounds = extract_cnt_coords(df=self.df_list[t])
            self.compute_centroid_and_radius()

            print( "CNT Bounds: ", self.cnt_bounds)
            self.cnt_bound_list.append(self.cnt_bounds)
            print( "Bound list: ", self.bound_list[t])

            #define empty arrays for concatenating
            I_temp = np.empty((n_surf*max_n_points, )) #4 -> number of surfaces in
            box_min_temp = np.empty((n_surf*max_n_points, 3)) #define each convolution by the minimum

            #The surface id and the
            surface_temp = np.empty_like(I_temp)
            loc_temp = np.empty_like(box_min_temp)

            #final arrays
            loc_final = np.empty((max_n_points, 3))

            #numpy arrays to stack
            coords_array = np.empty((0, 3))


            #print "Bounds: ", bound_new

            #for each centroid we will create boxes

            #with pymp.Parallel(24) as p:
            #block to intend
            for i in range(0, 1):
                box_bound, atom_bound = self.find_box_atoms(df=self.df_list[t], cnt_point=self.centroids[i], R_vector=self.R_v[i, :], total_bounds=self.bound_list[t])
                property_map, x_c, y_c, z_c, count_mat, rho_mat, r = convolve_over_box(atom_array=atom_bound, sim_bounds=box_bound,
                                                                       channel_size=temp_channel, delta=stride, unique_type=self.unique_type)


                filter_coords = np.hstack((x_c.flatten()[:, None], y_c.flatten()[:, None], z_c.flatten()[:, None]))

                #print filter_coords.shape
                #print coords_array.shape
                coords_array = np.vstack((coords_array, filter_coords))



            #Redo find_location to find
            #I_final, idx_final = find_location_I(I=I_temp, max_n_points=max_n_points)





        return count_mat, rho_mat, r

    def plot_contour_3D(self, timestep, N):

        cnt_bound = self.cnt_bound_list[timestep]
        X = self.scatter_list[timestep]

        radius = 0.5*(cnt_bound[0, 1] - cnt_bound[0, 0])
        height = cnt_bound[2, 1] - cnt_bound[2, 0]
        elevation = cnt_bound[2, 0]
        resolution = 100
        color = 'b'
        x_center = 0.5*(cnt_bound[0, 1] + cnt_bound[0, 0])
        y_center = 0.5*(cnt_bound[1, 1] + cnt_bound[1, 1])


        #extract data for

        plot_3D_cylinder(X, radius, height, elevation=elevation, resolution=resolution, color=color, x_center=x_center,
                         y_center=y_center)

        plt.show()

        return None


    def convolve_shear_102(self):

        stride = self.stride
        temp_channel_init = self.convolve_size.copy()
        max_n_points = 10
        n_surf = 4 #number of surfaces to convolve over

        max_t = len(self.array_list)
        bound_old = self.bound_list[0].copy()


        property_map_list = []
        U_list = []
        idx_list_list = []
        cnt_bound_list = []

        #shared array to track:


        #with pymp.Parallel(24) as p:
        for t in range(0, 24):

            atom_new = self.array_list[t].copy()
            bound_new = self.bound_list[t].copy()
            delta_x = (bound_new[0, 1] - bound_new[0, 0]) - (bound_old[0, 1] - bound_old[0, 0])
            delta_y = (bound_new[1, 1] - bound_new[1, 0]) - (bound_old[1, 1] - bound_old[1, 0])
            delta_z = (bound_new[2, 1] - bound_new[2, 0]) - (bound_old[2, 1] - bound_old[2, 0])

            temp_channel = temp_channel_init + np.asarray([delta_x, delta_y, delta_z])

            #cnt_bounds, box_bound = extract_cnt_coords(self.df_list[t], bound_new, self.box_tol, self.prop_str)
            #now let's start getting the cnt atoms
            self.cnt_bounds = extract_cnt_coords(df=self.df_list[t])
            self.compute_centroid_and_radius()

            #append cnt bound list
            cnt_bound_list.append(self.cnt_bounds)

            #define empty arrays for concatenating
            I_temp = np.empty((n_surf*max_n_points, )) #4 -> number of surfaces in
            box_min_temp = np.empty((n_surf*max_n_points, 3)) #define each convolution by the minimum

            #The surface id and the
            surface_temp = np.empty_like(I_temp)
            loc_temp = np.empty_like(box_min_temp)

            #final arrays
            loc_final = np.empty((max_n_points, 3))


            #print "Bounds: ", bound_new

            #for each centroid we will create boxes

            #with pymp.Parallel(24) as p:
            #block to intend
            for i in range(0, 4):
                box_bound, atom_bound = self.find_box_atoms(df=self.df_list[t], cnt_point=self.centroids[i], R_vector=self.R_v[i, :], total_bounds=self.bound_list[t])
                property_map, xyz_array, idx_array = convolve_over_box(atom_array=atom_bound, sim_bounds=box_bound,
                                                                       channel_size=temp_channel, delta=stride)


                I = property_map

                print( I)
                I_sort, idx_max = find_location_I(I=I, max_n_points=max_n_points)

                print( "I: ", I)
                print( "I_sort", I_sort)
                I_temp[i*max_n_points:(i+1)*max_n_points] = I_sort
                surface_temp[i*max_n_points:(i+1)*max_n_points] = i

                loc_temp[i*max_n_points:(i+1)*max_n_points, :] = idx_max
                box_min_temp[i*max_n_points:(i+1)*max_n_points, :] = np.multiply(idx_max, self.stride) + box_bound[:, 0]
                print( "troubleshoot stride multiply")
                print( box_min_temp)




            #Redo find_location to find
            I_final, idx_final = find_location_I(I=I_temp, max_n_points=max_n_points)


            idx_final = idx_final.flatten()
            print( "idx_final: ", idx_final)
            box_final = box_min_temp[idx_final, :]
            #loc_final = loc_temp[idx_final]
            surface_final = surface_temp[idx_final]

            #print loc_final




        return None


    def find_box_atoms(self, df, cnt_point, R_vector, total_bounds):

        # df_cnt -> dataframe containing CNT properties
        # cnt_point -> the centroid of the rectangular box -> array of length 3
        # R_vector -> 3D array to add and substract to the cnt_point

        box_bounds = total_bounds.copy()

        #print "R_vector: ", R_vector

        for i in range(0, 3):
            if cnt_point[i] - R_vector[i] >= total_bounds[i, 0]:
                box_bounds[i, 0] = cnt_point[i] - R_vector[i]

            if cnt_point[i] + R_vector[i] <= total_bounds[i, 1]:
                box_bounds[i, 1] = cnt_point[i] + R_vector[i]

        # creating numpy arrays
        ID_mat = np.asarray(df.loc[:, 'id'])
        type_mat = np.asarray(df.loc[:, 'type'])
        xyz_mat = np.asarray(df.loc[:, ['x', 'y', 'z']])
        prop_mat = np.asarray(df.loc[:, self.prop_str])

        trans_array = shift_coords(xyz_mat, total_bounds)
        trans_bounds = np.transpose(shift_coords(np.transpose(box_bounds), total_bounds))

        idx = np.where((trans_array[:, 0] >= trans_bounds[0, 0]) & (trans_array[:, 0] <= trans_bounds[0, 1]) & (
            trans_array[:, 1] >= trans_bounds[1, 0])
                       & (trans_array[:, 1] <= trans_bounds[1, 1]) & (trans_array[:, 2] >= trans_bounds[2, 0]) & (
                           trans_array[:, 2] <= trans_bounds[2, 1]))

        # select from df which atoms go in

        ID_bound = ID_mat[idx[0]]
        ID_bound = ID_bound[:, None]
        xyz_atoms = xyz_mat[idx[0], :]

        type_bound = type_mat[idx[0]]
        type_bound = type_bound[:, None]
        prop_bound = prop_mat[idx[0], :]

        atom_bound = np.hstack((ID_bound, type_bound, xyz_atoms, prop_bound ))

        return box_bounds, atom_bound



    def compute_centroid_and_radius(self):

        cnt_bounds = self.cnt_bounds
        print( "cnt_bounds: ", cnt_bounds)
        #the cnt bounds will be stored as a numpy array
        #the rows in th np array correspond to [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]
        self.centroids = np.zeros((6, 3))
        self.centroids[0, :] = np.asarray([cnt_bounds[0, 0], 0.5*(cnt_bounds[1, 0] + cnt_bounds[1, 1]), 0.5*(cnt_bounds[2, 0] + cnt_bounds[2, 1])])
        self.centroids[1, :] = np.asarray([cnt_bounds[0, 1], 0.5 * (cnt_bounds[1, 0] + cnt_bounds[1, 1]),
                                           0.5 * (cnt_bounds[2, 0] + cnt_bounds[2, 1])])
        self.centroids[2, :] = np.asarray([0.5 * (cnt_bounds[0, 0] + cnt_bounds[0, 1]), cnt_bounds[1, 0], 0.5 * (cnt_bounds[2, 0] + cnt_bounds[2, 1])])
        self.centroids[3, :] = np.asarray([0.5 * (cnt_bounds[0, 0] + cnt_bounds[0, 1]), cnt_bounds[1, 1],
                                           0.5 * (cnt_bounds[2, 0] + cnt_bounds[2, 1])])
        self.centroids[4, :] = np.asarray([0.5 * (cnt_bounds[0, 0] + cnt_bounds[0, 1]), 0.5 * (cnt_bounds[1, 0] + cnt_bounds[1, 1]), cnt_bounds[2, 0]])
        self.centroids[5, :] = np.asarray([0.5 * (cnt_bounds[0, 0] + cnt_bounds[0, 1]), 0.5 * (cnt_bounds[1, 0] + cnt_bounds[1, 1]),
             cnt_bounds[2, 1]])

        R = self.Radius
        e = self.epsilon

        self.R_v = np.empty_like(self.centroids)
        self.R_v[0, :] = np.asarray([R, e + 0.5*(cnt_bounds[1, 1] - cnt_bounds[1, 0]), e + 0.5*(cnt_bounds[2, 1] - cnt_bounds[2, 0])])
        self.R_v[1, :] = np.asarray([R, e + 0.5 * (cnt_bounds[1, 1] - cnt_bounds[1, 0]), e + 0.5 * (cnt_bounds[2, 1] - cnt_bounds[2, 0])])
        self.R_v[2, :] = np.asarray([e + 0.5*(cnt_bounds[0, 1] - cnt_bounds[1, 1]), R, e + 0.5 * (cnt_bounds[2, 1] - cnt_bounds[2, 0])])
        self.R_v[3, :] = np.asarray([e + 0.5 * (cnt_bounds[0, 1] - cnt_bounds[0, 0]), R, e + 0.5 * (cnt_bounds[2, 1] - cnt_bounds[2, 0])])
        self.R_v[4, :] = np.asarray([e + 0.5*(cnt_bounds[0, 1] - cnt_bounds[0, 0]), e + 0.5*(cnt_bounds[1, 1] - cnt_bounds[1, 0]), R])
        self.R_v[5, :] = np.asarray([e + 0.5 * (cnt_bounds[0, 1] - cnt_bounds[0, 0]), e + 0.5 * (cnt_bounds[1, 1] - cnt_bounds[1, 0]), R])


        return None




class hotspot():
    def __init__(self, point_1, R_vector, file_name):

        if len(point_1) > 1:
            self.center = 0.5*(point_1[0] + point_1[1])
        else:
            self.center = point_1[0]

        self.radius = R_vector
        self.xyz_min = self.center - R_vector
        self.xyz_max = self.center + R_vector
        self.filename = file_name


        #the corner is xyz_min

    def get_atoms(self):
        #This function gets x y z co-ordinates and writes them to file
        df = pd.read_table(self.filename, delim_whitespace=True, header=None, skiprows=9)

        #read the file to get the header
        with open(self.filename) as f:
            text = f.readlines()

            for line in text:
                if line.startswith('ITEM: ATOMS'):
                    header_choose = line.strip()
                    header_choose = header_choose.split()
                    header_choose = header_choose[2:] #getting rid off the first two items, as it is ITEM: ATOMS

                    #hedaer_choose is the list of headers.




        df.columns = header_choose
        #print df.query('x > self.xyz_min[0]')
        #now let's look at the elements within the bound
        idx = [np.where((df['x'] > self.xyz_min[0]) & (df['x'] < self.xyz_max[0]) & (df['y'] > self.xyz_min[1]) & (df['y'] < self.xyz_max[1])
                       & (df['z'] > self.xyz_min[2]) & (df['z'] < self.xyz_max[2]))]
        #idx = [np.where(((df['x'] > self.xyz_min[0]) & (df['x'] < self.xyz_max[0])) | ((df['y'] > self.xyz_min[1]) & (
           # df['y'] < self.xyz_max[1])) | ((df['z'] > self.xyz_min[2]) & (df['z'] < self.xyz_max[2])))]
        idx = (np.asarray(idx)).ravel()
        self.df_bound = df.iloc[idx, :] #The boundary of the simulation box within acceptable limits


        return self



    ####function to create a box in hotspot
    def create_box(self):
        self.atype = np.asarray(self.df_bound['type'])
        self.pos = np.array(self.df_bound.loc[:, ['x', 'y', 'z']])

        self.min = np.amin(self.pos, axis=0)
        self.pos_transform = self.pos - self.min


        #create an Atoms Object
        self.atoms = am.Atoms(natoms=len(self.atype), prop={'atype': self.atype, 'pos': self.pos})
        self.box = am.Box(a=2*self.radius[0], b=2*self.radius[1], c=2*self.radius[2])
        self.system = am.System(atoms=self.atoms, box=self.box, pbc=(True, True, True), scale=False)
        sys_info = am.lammps.atom_data.dump(self.system, 'test1.in')

        return self


def read_MD_model(file_list):


    #this function is to read the input file, which is subsequently used to compute RDF -> step [7] in dmain document

    #first i will read the first file to get an idea of how many lines to skip
    filename = file_list[0]

    #fix header_list for the pandas table
    header_list = ['id', 'type', 'q', 'x', 'y', 'z', 'nx', 'ny', 'nz']


    #df_list = pymp.shared.list()
    #len_z_list = pymp.shared.list()
    #z_coords_list = pymp.shared.list()

    df_list = []
    len_z_list = []
    z_coords_list = []

    ###fixing params for pymp
    rnge = iter(range(len(file_list)))

    #with pymp.Parallel(multiprocessing.cpu_count()) as p:
    #for t in p.iterate(rnge):
    for t in range(0, len(file_list)):
        #print file_list[t]

        with open(file_list[t]) as f:

            df_bounds = pd.read_table(file_list[t], delim_whitespace=True, header=None, skiprows=5, nrows=3) #import the periodic cell boundary as a pandas dataframe
            df_bounds = df_bounds.iloc[:, 0:2].as_matrix()

            z_lo = df_bounds[-1, 0]
            z_hi = df_bounds[-1, 1]

            len_z = z_hi - z_lo #length of the periodic cell


            text = f.readlines()

            count = 0
            skip_count = 0
            end_count = 0

            for line in text:

                count = count + 1

                if line.startswith('Atoms'):
                    skip_count = count

                if line.startswith('Velocities'):
                    end_count = count - 4
                    break

        print( "######'")
        print( file_list)
        #note that skip_count and end_count indicates where

        Z = pd.read_table(file_list[t], delim_whitespace=True, header=None, skiprows=skip_count, nrows=(end_count - skip_count)) #read the coordinates and atom types

        Z.columns = header_list #attach headers mentioned above to the pandas dataframe

        df_list.append(Z) #eppending dataframe
        len_z_list.append(len_z)
        z_coords_list.append(np.asarray([z_lo, z_hi]))



    return df_list, len_z_list, z_coords_list



#Global functions
#functions to read
def select_by_property(file_list, prop_name, N, sort_choose=False):

    df_list = []
    array_list = []
    bound_list = []
    pe_hotspot_list = []

    for file_name in file_list:
        df = pd.read_table(file_name, delim_whitespace=True, header=None, skiprows=9)

        #read text to get headers
        # read the file to get the header
        with open(file_name) as f:
            text = f.readlines()

            for line in text:
                if line.startswith('ITEM: ATOMS'):
                    header_choose = line.strip()
                    header_choose = header_choose.split()
                    header_choose = header_choose[2:]  # getting rid off the first two items, as it is ITEM: ATOMS


            df_bounds = pd.read_table(file_name, delim_whitespace=True, header=None, skiprows=5, nrows=3)

            # bounds as array
            bound_array = df_bounds.as_matrix()


        df.columns = header_choose


        #now to index out the relevant columns
        if sort_choose == True:
            df = df.sort_values(by=prop_name, ascending=False)

        list_init = ['id', 'type', 'x', 'y', 'z']
        prop_new = list_init + prop_name

        array_out = np.asarray(df.loc[:, prop_new])
        array_out = np.asarray(df.loc[:, prop_new])
        hspot_array = array_out[:N, :]

        pe_hotspot_list.append(hspot_array)
        array_list.append(array_out)
        df_list.append(df)
        bound_list.append(bound_array)




    return df_list, array_list, pe_hotspot_list, bound_list



#keyfunction to sort filesa
def key_func(x):
    nondigits= re.compile("\D")

    return int(nondigits.sub("", x))



###########################

#Function to perform convolution of interest
def convolve_over_box(atom_array, sim_bounds, channel_size, delta, unique_type):

    #Nomenclature
    #atom_array -> simulation box containing all the atomic cords and property of interst
    #The first 3 columns column contain the co-ordinates of the atoms, the rest are the properties of interest
    #sim_bounds -> 3 x 2 array showing the limits of the box at that timestep:
    #channel size -> size of the "hotspot" box
    #delta -> increment of the filter
    #extract delta vector
    delta_x = delta[0]
    delta_y  =delta[1]
    delta_z = delta[2]

    #extract atomic co-ordinates
    x_lo = sim_bounds[0, 0]
    y_lo = sim_bounds[1, 0]
    z_lo = sim_bounds[2, 0]


    ###
    #extract co-ordinates from the atoms
    atom_bound = atom_array.copy()
    coords_array = atom_array[:, 2:5] #extract co-ordinates from atom _array
    prop_array = atom_array[:, 5:] #extract property array

    ##transofrm co-ordinates
    trans_array = coords_array.copy()
    trans_array[:, 0] = trans_array[:, 0] - x_lo
    trans_array[:, 1] = trans_array[:, 1] - y_lo
    trans_array[:, 2] = trans_array[:, 2] - z_lo
    atom_bound[:, 2:5] = trans_array

    #get the number of convolutions in each direction:
    #delta -> scalar indicating the stride
    num_x = math.ceil(((sim_bounds[0, 1] - sim_bounds[0, 0]) - channel_size[0])/(delta_x) + 1)
    num_x = int(num_x)
    num_y = int(math.ceil(((sim_bounds[1, 1] - sim_bounds[1, 0]) - channel_size[1])/(delta_y) + 1))
    num_z = int(math.ceil(((sim_bounds[2, 1] - sim_bounds[2, 0]) - channel_size[2]) / (delta_z) + 1))


    print( "Total nums: ", num_x, num_y, num_z)
    filter_bounds = np.array([[0, channel_size[0]], [0, channel_size[1]], [0, channel_size[2]]])
    filter_copy = filter_bounds.copy()


    #Now let's convolve the filter over our 3D box.
    #print trans_array[:5, :]

    #Initialize values to store
    store_prop = 100*np.ones(prop_array.shape[1], )
    # store_prop = np.zeros(prop_array.shape[1],)
    store_box = filter_bounds.copy()
    store_box = np.zeros((prop_array.shape[1], filter_bounds.shape[0], filter_bounds.shape[1]))

    #do a trial run to get the shape
    r, x_try = feature_extract_01.feature_equation_01(atom_bound=atom_bound[:10, :], unique_type=unique_type)
    l = x_try.shape[0]
    b = x_try.shape[1]

    #initalize numpy array
    property_map = np.zeros((num_x, num_y, num_z))
    x_center = np.zeros_like(property_map)
    y_center = np.zeros_like(property_map)
    z_center = np.zeros_like(property_map)


    #arrays to store the local densities
    count_mat = np.zeros((num_x, num_y, num_z))
    rho_mat = np.zeros((num_x*num_y*num_z, l, b))
    count = 0

    for z_loc in range(0, num_z):
        # Reset y
        filter_bounds[1, :] = filter_copy[1, :].copy()

        for y_loc in range(0, num_y):

            #Reset x
            filter_bounds[0, :] = filter_copy[0, :].copy()

            for x_loc in range(0, num_x):

                idx = np.where((trans_array[:, 0] > filter_bounds[0, 0]) & (trans_array[:, 0] < filter_bounds[0, 1]) & (trans_array[:, 1] > filter_bounds[1, 0])
                               & (trans_array[:, 1] < filter_bounds[1, 1]) & (trans_array[:, 2] > filter_bounds[2, 0]) & (trans_array[:, 2] < filter_bounds[2, 1]))


                #keeping track of count
                count_mat[x_loc, y_loc, z_loc] = count


                #this is the block to featurize
                atom_in_filter = atom_bound[idx[0], :]


                r, x = feature_extract_01.feature_equation_01(atom_bound=atom_in_filter, unique_type=unique_type)
                rho_mat[count, :] = x
                count += 1



                #Index out from the property table
                prop_box = prop_array[idx[0]]
                #computing mean of all atoms in the box:
                prop_mean = np.mean(prop_box, axis=0)
                property_map[x_loc, y_loc, z_loc] = prop_mean

                # updating co-ordinates of the filters:
                x_center[x_loc, y_loc, z_loc] = 0.5 * (filter_bounds[0, 0] + filter_bounds[0, 1]) + x_lo
                y_center[x_loc, y_loc, z_loc] = 0.5 * (filter_bounds[1, 0] + filter_bounds[1, 1]) + y_lo
                z_center[x_loc, y_loc, z_loc] = 0.5 * (filter_bounds[2, 0] + filter_bounds[2, 1]) + z_lo

                #finding the corresponding location of the box

                conv_box = trans_array[idx[0]]
                #u_vector = find_vectors(X=conv_box, r_th=4.0)
                #u_mat.append(u_vector)
                #idx_list.append(np.asarray([num_x, num_y, num_z]))


                #block to store property
                for i in range(0, len(prop_mean)):
                    if np.absolute(prop_mean[i]) < store_prop[i]:
                        store_prop[i] = np.absolute(prop_mean[i])
                        store_box[i, :, :] = filter_bounds.copy()





                #adjust channel box dimensions:


                filter_bounds[0, :] = np.array([filter_bounds[0, 0] + delta_x, filter_bounds[0, 1] + delta_x])




            filter_bounds[1, :] = np.array([filter_bounds[1, 0] + delta_y, filter_bounds[1, 1] + delta_y])

        filter_bounds[2, :] = np.array([filter_bounds[2, 0] + delta_z, filter_bounds[2, 1] + delta_z])






    return property_map, x_center, y_center, z_center, count_mat, rho_mat, r



#function to din
def stress_strain_curve(filename, strain_idx, sigma_idx):

    matrix_out = []
    line_count = 0

    #read file to get the stress-strain matrix
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line)>0 and line_count> 0:
                matrix_out.append(map(float, line.split()))

            line_count += 1

    matrix_out = np.asarray(matrix_out)
    matrix_out = reduce_by_mean(matrix_out, 5)
    strain_out = matrix_out[:, strain_idx]
    stress_out = matrix_out[:, sigma_idx]

    return strain_out, stress_out



#####read log-gile to get stress-strain curve


####################

def reduce_by_mean(mat_in, stride):

    row_max = int(math.ceil(float(len(mat_in))/float(stride)))
    col_max = int(mat_in.shape[1])

    mat_new = np.zeros((row_max, col_max))

    for i in range(0, row_max - 1):
        mat_new[i, :] = np.mean(mat_in[stride*i:stride*(i+1), :], axis=0)


    mat_new[row_max-1, :] = np.mean(mat_in[stride*(row_max-2):, :], axis=0)

    return mat_new




def read_log_lammps(filename, thermo_num):

    df = []
    for file in filename:
        with open(file, 'r') as f:
            text = f.readlines()

            line_count = 0
            line_choose = 0

            for line in text:
                word_list = line.strip().split()
                line_count += 1

                if len(word_list) == thermo_num:
                    line_choose = line_count
                    break
                #skiprow_list.append(len(word_list))


        df_temp = pd.read_table(file, delim_whitespace=True, skiprows=line_choose-1)

        #indexing out NaN values
        df_temp = remove_nan_dataframe(df_temp)
        df.append(df_temp)
        #print df

    df = pd.concat(df)



    stress = df['f_sxx_ave']
    stress = stress.as_matrix()
    stress_lores = reduce_by_mean(stress[:, None], 50)
    strain = df['Lx']
    L_mat = strain.as_matrix()
    L0 = L_mat[0]
    strain_hires = (L_mat - L0)/L0
    strain_lores = reduce_by_mean(strain_hires[:, None], 50)

    #potential energy
    PE = df['PotEng']
    PE = PE.as_matrix()
    pe_lores = reduce_by_mean(PE[:, None], 50)



    return stress_lores, strain_lores, strain_hires, pe_lores




#######
##function to find max ndarray
def max_nd_array(A, n):

    # A -> array to sort
    # n -> number of elements to pick
    A_c = A.copy()
    A_c = A_c.flatten()
    max_values = np.sort(A_c)[-n:]
    max_values = max_values[::-1]
    idx_max = np.argsort(A_c)[-n:]
    idx_max = idx_max[::-1]
    idx_max_unravel = np.unravel_index(idx_max, A.shape)


    return max_values, idx_max, idx_max_unravel



###function to remove nan in dataframes
def remove_nan_dataframe(df):

    data_columns = df.columns.values.tolist()

    numdf = (df.drop(data_columns, axis=1)
             .join(df[data_columns].apply(pd.to_numeric, errors='coerce')))

    numdf = numdf[numdf[data_columns].notnull().all(axis=1)]

    return numdf



def find_vectors(X, r_th):

    #x0 -> position of the atom in question
    u = np.empty((0, X.shape[1]))

    for i in range(0, len(X)):
        x0 = X[i, :]

        for j in range(i+1, len(X)):
            x1 = X[j, :]

            dist = LA.norm(x1 - x0)

            if dist < r_th and dist > 0:
                u_p = (x1 - x0)/dist
                u_p = np.reshape(u_p, (1,X.shape[1]))
                u = np.append(u, u_p, axis=0)



    return u



def compute_similarity(A, A_p):
    sim_mat = np.matmul(A, A_p.transpose())
    sim_val = LA.norm(sim_mat)

    return sim_val



def unpack_U_list(U_list):

    total_time = len(U_list)
    total_conv = len(U_list[0])

    sim_mat = np.zeros((total_conv, total_time))

    for c in range(0, total_conv):

        A0 = U_list[0][c]

        for t in range(0, total_time):

            A = U_list[t][c]
            sim_mat[c, t] = compute_similarity(A0, A)

            A0 = A.copy()




    return sim_mat



def find_max_sim(A):

    t_max = A.shape[1]
    diff_mat = np.zeros((len(A), t_max-1))
    max_vector = np.zeros((t_max,))
    argmax_vector = np.zeros((t_max,))

    for t in range(1, t_max):
        diff_mat[:, t-1] = A[:, t] - A[:, t-1]
        max_vector[t] = np.max(diff_mat[:, t-1])
        argmax_vector[t] = np.argmax(diff_mat[:, t-1])


    return max_vector, argmax_vector



def pca_fun(U_list, n):

    pca = PCA(n_components=n)

    total_time = len(U_list)
    total_conv = len(U_list[0])

    sim_mat = np.zeros((total_conv, total_time))
    u = np.zeros((n,))

    for c in range(0, total_conv):

        A0 = U_list[0][c]
        pca.fit(np.transpose(A0))
        v0 = pca.components_


        for t in range(0, total_time):

            A = U_list[t][c]
            pca.fit(np.transpose(A))
            v1 = pca.components_

            print( "v0: ", v0)
            print( "v1: ", v1)

            for i in range(0, n):
                u[i] = np.dot(v0[:, i], v1[:, i])

            print( "u:", u)

            sim_mat[c, t] = LA.norm(u)
            print( "sim_mat:", sim_mat[c,t])



    return sim_mat




def extract_cnt_coords(df, type_choose=22):

    df_cnt = df.loc[df['type'] == type_choose]
    cnt_bounds = np.zeros((3, 2))

    cnt_bounds[0, 0] = df_cnt['x'].min()
    cnt_bounds[0, 1] = df_cnt['x'].max()
    cnt_bounds[1, 0] = df_cnt['y'].min()
    cnt_bounds[1, 1] = df_cnt['y'].max()
    cnt_bounds[2, 0] = df_cnt['z'].min()
    cnt_bounds[2, 1] = df_cnt['z'].max()


    return cnt_bounds



def shift_coords(coords_array, sim_bounds):
    # extract atomic co-ordinates
    x_lo = sim_bounds[0, 0]
    y_lo = sim_bounds[1, 0]
    z_lo = sim_bounds[2, 0]


    ##transofrm co-ordinates
    trans_array = coords_array.copy()
    trans_array[:, 0] = trans_array[:, 0] - x_lo
    trans_array[:, 1] = trans_array[:, 1] - y_lo
    trans_array[:, 2] = trans_array[:, 2] - z_lo

    return trans_array



def find_location_I(I, max_n_points):

    I_f = I.flatten()
    I_sort = np.sort(I_f)[::-1][:max_n_points]
    argmax_val = I_f.argsort()[::-1][:max_n_points]
    multi_idx = np.unravel_index(np.asarray(argmax_val), I.shape)
    multi_idx = np.transpose(np.array(multi_idx))



    return I_sort, multi_idx


#plot cylinder
def plot_3D_cylinder(X2, radius, height, elevation=0, resolution=100, color='b', x_center = 0, y_center = 0):
    fig=plt.figure()
    ax = Axes3D(fig, azim=30, elev=30)

    x = np.linspace(x_center-radius, x_center+radius, resolution)
    z = np.linspace(elevation, elevation+height, resolution)
    X, Z = np.meshgrid(x, z)

    Y = np.sqrt(radius**2 - (X - x_center)**2) + y_center # Pythagorean theorem

    ax.plot_surface(X, Y, Z, linewidth=0, color=color, alpha=0.2, rstride=20, cstride=10)
    ax.plot_surface(X, (2*y_center-Y), Z, linewidth=0, color=color, alpha=0.2, rstride=20, cstride=10)

    floor = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

    ceiling = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation+height, zdir="z")

    ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], marker='o')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')


    return None



#####
def get_raw_targets(filename):

    df = pd.read_csv(filename, sep=',', header=None)
    y_out = df.as_matrix()[:, 1]

    return y_out


def state_action_pair(S1, S2, A):

    choose_load = True

    if choose_load == True:

        S1 = np.load('RBF_1_all.npy')
        S2 = np.load('RBF_2_all.npy')

        S = np.vstack(S1, S2)





    return None


###------functions to perform multiple simulations------------
def evaluate_prob_cnn(x_1, x_2, x_g, y, max_y, n1, n2, N_sample=50, f=400):

    #N_sample -> correspods to the total number of instantiations
    #performance metrics
    X_t1, X_t2, Xg_t, Y_t, X_e1, X_e2, Xg_e, Y_e = feature_extract_01.split_reuse_data(x1=x_1, x2=x_2, xg=x_g,
                                                                                       y=y, n1=200)


    #initializing empty arrays ->

    yp_s = np.zeros((N_sample, int(len(Y_e)/f)))
    ye_s = np.zeros_like(yp_s)
    rel_err = np.zeros((N_sample,))
    abs_err = np.zeros_like(rel_err)
    r2_val = np.zeros_like(rel_err)

    for i in range(N_sample):


        #X_t1, X_t2, Xg_t, Y_t, X_v1, X_v2, Xg_v, Y_v, X_e1, X_e2, Xg_e, Y_e = feature_extract_01.split_val_data(x1=x_1,
                                                                                                              #  x2=x_2,
                                                                                                               # xg=x_g,
                                                                                                              #  y=y,
                                                                                                               # n1=n1,
                                                                                                              #  n2=n2)

        X_t1, X_t2, Xg_t, Y_t, X_e1, X_e2, Xg_e, Y_e = feature_extract_01.split_reuse_data(x1=x_1, x2=x_2, xg=x_g,
                                                                                           y=y, n1=200)
        X_t1, X_t2, Xg_t, Y_t = feature_extract_01.shuffle_data02B(X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t)
        model_2 = ML_functions01.fit_model02B(RBF=x_1, X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t) #put this back in
        #model_2 = ML_functions01.fit_model02C(RBF=x_1, X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t, X1_v=X_v1, X2_v=X_v2, Xg_v=Xg_v,
                                              #Y_v=Y_v)

        Y_p = model_2.predict([X_e1, X_e2, Xg_e])
        Y_e, Y_p = feature_extract_01.pick_ogvalues(Y_e, Y_p.flatten())

        ye_s[i, :] = max_y * Y_e
        yp_s[i, :] = max_y * Y_p.flatten()



        rel_err[i] = np.sqrt(((max_y * Y_e - max_y * Y_p.flatten()) ** 2).mean()) / np.sqrt(
        ((max_y * Y_e) ** 2).mean())
        abs_err[i] = np.sqrt(((max_y * Y_e - max_y * Y_p.flatten()) ** 2).mean())
        r2_val[i] = r2_score(Y_e, Y_p.flatten())

        print( "rel_err", rel_err[i])
        print( "abs_err: ", abs_err[i])
        print( "R2_val: ", r2_val[i])
        print( "y_e: ", max_y * Y_e)
        print( "y_p: ", max_y*Y_p.flatten())

    ###we save the results for post-processing

    np.save('ye_RDF_50C.npy', ye_s)
    np.save('yp_RDF_50C.npy', yp_s)
    np.save('RE_RDF_50C.npy', rel_err)
    np.save('AE_RDF_50C.npy', abs_err)
    np.save('R2_RDF_50C.npy', r2_val)

    print( "rel error: ", np.mean(rel_err), np.std(rel_err))
    print( "abs error: ", np.mean(abs_err), np.std(abs_err))
    print( "R2: ", np.mean(r2_val), np.std(r2_val))



    return None



def rdf_saliency(model_pick, x_e1, x_e2, x_ge, idx=0):

    x_ec1 = x_e1[idx, :, :, :]
    x_ec2 = x_e2[idx, :, :, :]
    x_gec = x_ge[idx, :]
    grad_tensor = vis_modules.saliency_01(model=model_pick, X=[x_ec1[np.newaxis, :, :, :], x_ec2[np.newaxis, :, :, :], x_gec[np.newaxis, :]])


    return grad_tensor
