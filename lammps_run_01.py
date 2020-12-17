import numpy as np
from numpy import linalg as LA
import atomman as am
import pandas as pd
import os
from lammps import PyLammps, lammps

import matplotlib.pyplot as plt

import glob
import cPickle as pickle
import hotspot_file_01
import time
import feature_extract_01
import make_datafile

import tensorflow as tf

#peratom_files = './cnt.N2.1123/dump.sld1epo.*'
peratom_files = './3rr5epo.10/dump.sld1epo.*'
param_list = {'channel_size': np.array([8.0, 8.0, 15.0]), 'delta': np.array([4.0, 4.0, 2.0]), 'tol': np.array([5.0, 5.0, 5.0]), 'Radius': 3.0, 'epsilon': 3.0}
#prop_lookup = ['c_ppa']
prop_lookup = ['vz']
dprocess_1 = hotspot_file_01.dump_files(2, prop_lookup, 10, param_list, peratom_files)


#print dprocess_1.df_delepx_list[0]
#adding delatom files
#alkaneC_files = './cnt.N2.1123/dump.alkaneC1epo.*'
#bondedO_files = './cnt.N2.1123/dump.grp3A.*'
#bondedCNT_files = './cnt.N2.1123/dump.grp3B.*'

#dprocess_1.convolve_other()
#dprocess_1.add_group_files(alkane_file=alkaneC_files, fun_file=bondedO_files, bondedCNT_file=bondedCNT_files)


#dprocess_1.convolve_under_shear()

#Graph dictionary
#graph_list = {'f_min': 0, 'f_max': 0.05, 't_max': 200000, 'res': 500, 'f_idx1': 0.0, 'f_idx2': 0.05}
graph_list = {'f_min': 0.0, 'f_max': (0.200), 't_max': 200000, 'res': 500, 'f_idx1': 0, 'f_idx2': (0.200)}
dprocess_1.extract_cnt_velocity(graph_list)
#dprocess_1.plot_interaction_energy(graph_dict=graph_list, filename='fig_1C.eps')


## print dprocess_1.df_delcnt_list[0]

#count_mat, rho_mat, R = dprocess_1.convolve_under_shear()
#feature_extract_01.plot_features(R, rho_mat[5, :, :], 4)

#rho_mat = feature_extract_01.feature_cleanup(rho_mat)
#test_mat = np.arange(61*13).reshape((61, 13))

#print np.tensordot(rho_mat, test_mat, 2)/10000

#dprocess_1.plot_contour_3D(timestep=0, N=100)
#print dprocess_1.cnt_bounds



# trans_bounds, trans_array, xyz_atoms, type_bound, prop_bound = hotspot_file_01.extract_cnt_coords(dprocess_1.df_list[0],
#                                                                                                  dprocess_1.bound_list[0], param_list['tol'], prop_lookup)

#print cnt_bounds
#print box_bounds

#read

#plt.plot(np.arange(0, len(dprocess_1)), stress_lores, 'k-')
#plt.xlabel('$\gamma$')
#plt.ylabel('$\sigma$ (MPa)')
#plt.savefig('fig_stress_strain.eps')
#plt.show()

#plt.plot(strain_lores, pe_lores, 'k-')
#plt.xlabel('$\gamma$')
#plt.ylabel('PE (kcal/mole)')
#plt.savefig('fig_2.eps')
#plt.show()

#choose value
#choose_val = True

#if choose_val:
#    dump_process_1 = pickle.load(open('dump_file2.p', 'rb'))
#else:
#    dump_process_1 = hotspot_file_01.dump_files(1, prop_lookup, 10, param_list, peratom_files)
#    dump_process_1.convolve_under_strain(strain_mat=strain_hires, max_strain=0.04, t_strain=0.04)
#    pickle.dump(dump_process_1, open('dump_file2.p', 'wb'))

#A_mat = hotspot_file_01.unpack_U_list(dump_process_1.U_list)
#a1, a2 = hotspot_file_01.find_max_sim(A_mat)


#print a2


#trans_array = hotspot_file_01.convolve_over_box(atom_array=process_1.pe_array_list[100], sim_bounds=process_1.bound_list[100], channel_size=channel_size, delta=delta)
#print trans_array
#Now to find the xyz co-ordinates where the interatomic distance is high
#N = 10 #I want to get 10 hotspot locations
#T = 5#Only the final snapshot

#print len(process_1.coords_list)
#process_1.find_atoms_by_dist(process_1.array_list, process_1.coords_list,  N=N)

#print process_1.xyz_list


#choosing hotspot
#hotspot_id = 0 #select timestep
#point = np.vstack((process_1.xyz_1[hotspot_id], process_1.xyz_2[hotspot_id]))
#pe_list = hotspot_file_01.select_by_property(pe_files, 'c_2B')


#R_vector = [5, 5, 5]
#file_to_read = './deform/dump.defo.10000'
#hotspot_all = hotspot_file_01.hotspot(point, R_vector=R_vector, file_name=file_to_read)
#hotspot_all = hotspot_all.get_atoms()
#hotspot_all = hotspot_all.create_box()

#calc_dist = LA.norm(process_1.xyz_2-process_1.xyz_1, axis=1)


#The goal now is to valdiate that the deformed material within the hotspot has the same interatomic distance
#file_troubleshoot = './mini_defo/mini.dump.10000'
#df = pd.read_table(file_troubleshoot, delim_whitespace=True, header=None, skiprows=9)
#df = df.sort_values(by=3, ascending=False) # 2 because y

#file_troubleshoot = './mini_defo/dump.mini.10000'
#df2 = pd.read_table(file_troubleshoot, delim_whitespace=True, header=None, skiprows=9)
#df2= df2.sort_values(by=0, ascending=True)  # 2 because y

#print df
#print (df2.iloc[9, :] + df2.iloc[4, :])/2



