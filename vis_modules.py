import keras.backend as K
from vis.visualization import visualize_saliency

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import feature_extract_01

def saliency_01(model, X):


    output_layer = model.output
    idx_t = len(X) #total number of inputs
    grad_tensor_list = []

    for idx in range(idx_t):
        grad_tensor = K.gradients(output_layer, model.inputs)[idx] #idx to indicate its the first entry in an array
        derivate_fn = K.function(model.inputs, [grad_tensor])
        out_grad = derivate_fn(X)
        grad_tensor_list.append(out_grad) #append gradients

    ##
    #out_grad = derivate_fn(X)



    return grad_tensor_list


###--------------------------------------
def occlusion_rdf01(model, X1, X2, Xg):

    X1, X2, Xg = X1[np.newaxis, :, :, :], X2[np.newaxis, :, :, :], Xg[np.newaxis, :]


    Y_p = model.predict([X1, X2, Xg])

    delta_x1 = np.zeros((X1.shape[1], X1.shape[2], X1.shape[3]))
    delta_x2 = np.zeros_like(delta_x1)
    delta_xg = np.zeros((Xg.shape[1], ))

    min_xg = np.load('min_xg.npy')
    ####block out
    for i in range(0, delta_x1.shape[0]):
        for j in range(0, delta_x1.shape[1]):
            for k in range(0, delta_x1.shape[2]):
                X_temp = X1.copy()
                X_temp[0, i, j, k] = 0

                Y_temp = model.predict([X_temp, X2, Xg])

                #print "Y_p: ", Y_p
                #print "Y_temp: ", Y_temp
                delta_x1[i, j, k] = Y_p - Y_temp

                #if X1[0, i, j, k] == 0 and delta_x1[i, j, k] != 0:
                    #print "X1[0, i, j, k]: ", X1[0, i, j, k]
                    #print "X_temp[0, i, j, k]: ", X_temp[0, i, j, k]
                    #print "delta_x1[0, i, j, k]: ", delta_x1[i, j, k]

    for i in range(0, delta_x2.shape[0]):
        for j in range(0, delta_x2.shape[1]):
            for k in range(0, delta_x2.shape[2]):
                X_temp = X2.copy()
                X_temp[0, i, j, k] = 0

                Y_temp = model.predict([X1, X_temp, Xg])

                # print "Y_p: ", Y_p
                # print "Y_temp: ", Y_temp
                delta_x2[i, j, k] = Y_p - Y_temp

                #if X2[0, i, j, k] == 0 and delta_x2[i, j, k] != 0:
                    #print "X2[0, i, j, k]: ", X2[0, i, j, k]
                    #print "X_temp[0, i, j, k]: ", X_temp[0, i, j, k]
                    #print "delta_x1[0, i, j, k]: ", delta_x2[i, j, k]

    for i in range(0, delta_xg.shape[0]):
        X_temp = Xg.copy()
        X_temp[0, i] = 0
        Y_temp = model.predict([X1, X2, X_temp])
        delta_xg[i] = Y_p - Y_temp

    #input dimensions [z, x, property]
    #print X1[X1 !=0]
    #print "Delta_x1:", np.amax(np.abs(delta_x1))
    #print "Delta_X2:", np.amax(np.abs(delta_x2))
    #print "Delta xg: ", delta_xg

    return (delta_x1), (delta_x2), (delta_xg)


def plot_saliency_maps(X):

    #X is m x n
    #X[X < 0] = 0
    fig, ax = plt.subplots()
    cax = ax.matshow(X)
    fig.colorbar(cax)
    plt.show()
    plt.savefig('fig_heatmap02.png', bbox_inches="tight")

    labels = ['Alkane C', 'Benzene ring', 'Primary N', 'Epoxide C', 'Seconary N', 'Tertiary N', 'Hydroxyl O', 'Periodicity in x/y', 'Periodicity in Z', 'Node Proximity to CNT']

    ax = sns.heatmap(X, xticklabels=labels, square=False)
    plt.show(ax)
    ax.set_xlabel('Node features')
    ax.set_ylabel('Node #')
    figure = ax.get_figure()
    #figure.savefig('fig_heatmap02B.png', dpi=400, bbox_inches="tight")
    #plt.savefig('fig_heatmap.png', bbox_inches="tight")
    #plt.xticks(X[], labels, rotation='vertical')



    return None



def get_interface_atoms(df_xyz, cnt_type=22, select_type=[2, 4, 9, 12, 15, 18, 16], r_thres=5.0, r_min = 3.5, r_max = 5.0):



    select_type = [9]

    df_cnt = df_xyz.loc[df_xyz['type'] == cnt_type]
    df_other = df_xyz.loc[df_xyz['type'].isin(select_type)]



    cnt_xyz = df_cnt.loc[:, ['x', 'y', 'z']].as_matrix()
    fun_id = df_other.loc[:, ['id']].as_matrix()
    fun_xyz =  df_other.loc[:, ['x', 'y', 'z']].as_matrix()

    D, _, _ = feature_extract_01.compute_distance(cnt_array=cnt_xyz, cnt_other_array=fun_xyz)
    idx_thr = np.where(D < r_thres)[0]

    print "idx_thr ", len(idx_thr)
    id_select = (fun_id[idx_thr].squeeze())

    #print "D_mat: ", len(D[D < r_thres])
    #print D.shape
    print "id_select ", len(id_select)


    idx_intv = np.where((D < r_max) & (D > r_min))[0]
    id_range = fun_id[idx_intv].squeeze()

    print "id_range: ", np.unique(id_range)


    command_out = 'ParticleType == 22'

    for j in np.unique(id_range).tolist():
        command_out = command_out + ' || ' + ' ParticleIdentifier == ' + str(j)

    print command_out
    find_distance(df_xyz=df_xyz, particle_1=4673)

    return id_select




def find_distance(df_xyz, particle_1, cnt_type=22):

    df_cnt = df_xyz.loc[df_xyz['type'] == cnt_type]
    cnt_xyz = df_cnt.loc[:, ['x', 'y', 'z']].as_matrix()

    df_fun = df_xyz.loc[df_xyz['id'] == particle_1]
    fun_xyz = df_fun.loc[:, ['x', 'y', 'z']].as_matrix()

    D, _, _ = feature_extract_01.compute_distance(cnt_array=cnt_xyz, cnt_other_array=fun_xyz)
    d = np.amin(D).squeeze()

    print "d: ", d

    return None