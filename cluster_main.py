import numpy as np
import pandas as pd
import cluster_01
import matplotlib.pyplot as plt
import glob

import cluster_01B
import graph_features
import ML_graph01
import vis_modules
import hotspot_file_01

from sklearn.metrics import mean_squared_error, r2_score
from vis.visualization import visualize_saliency

file_list = ['./MDfiles/select03/cluster_analysis/c1epo.*', './MDfiles/select03/cluster_analysis/c2epo.*',
             './MDfiles/select03/cluster_analysis/c5epo.*', './MDfiles/select03/cluster_analysis/c3epo.*', './MDfiles/select03/cluster_analysis/c4epo.*']

MD_pattern = ['./MDfiles/select03/3rr1epo.*', './MDfiles/select03/3rr2epo.*', './MDfiles/select03/3rr5epo.*',
             './MDfiles/select03/3rr3epo.*', './MDfiles/select03/3rr4epo.*']

force_pattern = ['./MDfiles/select03/f_3rr1epo.csv', './MDfiles/select03/f_3rr2epo.csv', './MDfiles/select03/f_3rr5epo.csv',
             './MDfiles/select03/f_3rr3epo.csv', './MDfiles/select03/f_3rr4epo.csv']


chain_list = ['./MDfiles/select03/chain_analysis/b1epo.*', './MDfiles/select03/chain_analysis/b2epo.*',
             './MDfiles/select03/chain_analysis/b5epo.*', './MDfiles/select03/chain_analysis/b3epo.*', './MDfiles/select03/chain_analysis/b4epo.*']




y = np.load('targets.npy')

#printing lengths

BFDGE = cluster_01.BFDGE()
print( "BFDGE" )
print( BFDGE )
idx = 137
load_bool = True
MD_files,f_files, chain_files = cluster_01.read_MD_files(list_of_files=MD_pattern, forcefile=force_pattern, chainfile=chain_list)

print( "****************************" )
print( len(MD_files) )
print( len(f_files) )
print( len(chain_files) )

MD_test = cluster_01.MD_model(MD_file=MD_files[idx], f_value=f_files[idx], chain_file=chain_files[idx])
print( MD_files[idx], f_files[idx] )

#vis_modules.get_interface_atoms(df_xyz=MD_test.df_xyz)

#MD_data = graph_features.graph_data(MD_files=MD_files, f_files=f_files, chain_files=chain_files)

if load_bool == True:
    #Z_1 = np.load('Z_1_100.npy')
    #Z_2 = np.load('Z_2_100.npy')
    Z_1 = np.load('Z_1_40.npy')
    Z_2 = np.load('Z_2_40.npy')
    Z_0 = np.load('Z_0_40.npy')
    max_len = np.load('max_len_10.npy')
    print( "max_len: ", max_len )
else:
    MD_data = graph_features.graph_data(MD_files=MD_files, f_files=f_files, chain_files=chain_files)
    #np.save('Z_1_5.npy', MD_data.Z_1)
    #np.save('Z_2_5.npy', MD_data.Z_2)
    np.save('Z_0_40.npy', MD_data.Z_0)
    np.save('max_len_10.npy', MD_data.max_len)
    np.save('CL_20.npy', MD_data.atomic_length)


x_g = cluster_01B.get_global_features(list_of_files=MD_pattern, chain_files=chain_list)
Y = graph_features.get_y(list_of_files=MD_pattern, forcefile=force_pattern)

print( Y.shape )
x_1, x_g, y = graph_features.preprocessing_01(Z_1=Z_1, Z_2=Z_2, x_g=x_g, Y=Y)
x_t, x_gt, y_t, x_e, x_ge, y_e = graph_features.split_data(x_1=x_1, x_g=x_g, y=y)

idx = 0
#print( "*******Training data****" )
#print( "X_t[idx] :", x_t[idx, :, :] )
#print( "Y_t: ", y_t[idx] )
#print( "x_gt: ", x_gt[idx, :] )


#print( "*******Test data****" )
#print( "X_E[idx] :", x_e[idx, :, :] )
#print( "Y_e: ", y_e[idx] )
#print( "x_ge: ", x_ge[idx, :] )

print( "********************Model************" )
model_gcn = ML_graph01.fit_model01(X1=x_t, Xg=x_gt, Y=y_t)
#model_gcn = ML_graph01.fit_model01B(X1=x_t, Xg=x_gt, Y=y_t)
y_p = model_gcn.predict([x_e, x_ge])
#grad_top_1 = visualize_saliency(model=model_gcn, layer_idx=0, filter_indices=1, seed_input=[x_e[0, :, :], x_ge[0, :]])
idx=0
x_ec = x_e[idx, :, :]
x_gec = x_ge[idx, :]
#grad_tensor = vis_modules.saliency_01(model=model_gcn, X=[x_ec[np.newaxis, :, :], x_gec[np.newaxis, :]])
#print( "grad_tensor: ", grad_tensor )
print( "y_e: ", np.amax(Y)*y_e )
print( "y_p: ", np.amax(Y)*y_p.flatten() )
#vis_modules.plot_saliency_maps(grad_tensor[0][0][0, :, :])
#print( "grad_tensor [1] : ", grad_tensor[1] )

#w_1, b_1 = model_gcn.layers[-1].get_weights()
#w_2, b_2 = model_gcn.layers[-2].get_weights()






idx = 2
#print( "w_1: ", w_1 )
#print( "w_2: ", w_2.shape )
#print( "w_2: ", w_2 )
#cluster_01B.get_global_features(chain_files=chain_list)
#cluster_files = cluster_01.cluster_files(list_of_files=file_list)

print( "error: " )
print(  np.sqrt(((np.amax(Y)*y_e - np.amax(Y)*y_p.flatten())**2).mean())/np.sqrt(((np.amax(Y)*y_e)**2).mean()) )
print(  np.sqrt(((np.amax(Y)*y_e - np.amax(Y)*y_p.flatten())**2).mean()) )
print( "R2: ", r2_score(y_e, y_p) )
graph_features.evaluate_prob_graph(x_1=x_1, x_g= x_g, y=y, max_y=np.amax(Y))



print( "***************************************" )
print( "Alt model" )
model_2 = ML_graph01.fit_model01B(X1=x_t, Xg=x_gt, Y=y_t)
y_p = model_2.predict([x_e, x_ge])
print( "y_e: ", np.amax(Y)*y_e )
print( "y_p: ", np.amax(Y)*y_p.flatten() )
print( "error: " )
print(  np.sqrt(((np.amax(Y)*y_e - np.amax(Y)*y_p.flatten())**2).mean())/np.sqrt(((np.amax(Y)*y_e)**2).mean()) )
print(  np.sqrt(((np.amax(Y)*y_e - np.amax(Y)*y_p.flatten())**2).mean()) )
print( "R2: ", r2_score(y_e, y_p) )


#Z_1, Z_2 = graph_features.preprocessing_02(Z_1, Z_2)
#print( "Whats going in? " )
#print( "Z_1: ", Z_1[0:10, :, :] )
#print( "Z_2: ", Z_2[0:10, :, :] )
#print( "x_g: ", x_g[0:10, :] )
######this block of code is to try out the new model with modified weight matrix
#ML_graph01.gcn_012(Z_1=Z_1, Z_2=Z_2, x_g=x_g, y_e=y)
#x = cluster_files.rho_list

#print( "length(x): ", len(x) )
#print( "length(y): ", len(y) )
#plt.scatter(x, y, marker='o', color='k')
#plt.plot(y_14A, y_14A, 'k-')
#plt.scatter(y_5A, y_5B, marker='o', color='k')
#plt.plot(y_5A, y_5A, 'k-')
#plt.xlabel('Pull-out force from MD simulation, $y_a$ (kcal/mol-$\AA$)')
#plt.ylabel('Pull-out force predicted by CNN, $y_p$ (kcal/mol-$\AA$)')
#plt.savefig('fig_14A.svg')
#plt.show()

#print( "R^2 score", r2_score(x, y) )

#MD_test = cluster_01.MD_model(MD_file=MD_files[idx], f_value=f_files[idx], chain_file=chain_files[idx])
#print( MD_files[idx], f_files[idx]

print( "**********************************************shared model************************" )
print( "shared model" )
print( Z_0.shape )
x_t, x_gt, y_t, x_e, x_ge, y_e = graph_features.split_data(x_1=Z_0, x_g=x_g, y=y)
model_2 = ML_graph01.fit_model021(X1=x_t, Xg=x_gt, Y=y_t)
y_p = model_2.predict([x_e, x_ge])
print( "y_e: ", np.amax(Y)*y_e )
print( "y_p: ", np.amax(Y)*y_p.flatten() )
print( "error: " )
print(  np.sqrt(((np.amax(Y)*y_e - np.amax(Y)*y_p.flatten())**2).mean())/np.sqrt(((np.amax(Y)*y_e)**2).mean()) )
print(  np.sqrt(((np.amax(Y)*y_e - np.amax(Y)*y_p.flatten())**2).mean()) )
print( "R2: ", r2_score(y_e, y_p) )

print( "**********************************************shared model 2************************" )
#x_t, x_gt, y_t, x_e, x_ge, y_e = graph_features.split_data(x_1=Z_0, x_g=x_g, y=y)
model_2 = ML_graph01.fit_model022(X1=x_t, Xg=x_gt, Y=y_t)
y_p = model_2.predict([x_e, x_ge])
print( "y_e: ", np.amax(Y)*y_e )
print( "y_p: ", np.amax(Y)*y_p.flatten() )
print( "error: " )
print(  np.sqrt(((np.amax(Y)*y_e - np.amax(Y)*y_p.flatten())**2).mean())/np.sqrt(((np.amax(Y)*y_e)**2).mean()) )
print(  np.sqrt(((np.amax(Y)*y_e - np.amax(Y)*y_p.flatten())**2).mean()) )
print( "R2: ", r2_score(y_e, y_p) )

