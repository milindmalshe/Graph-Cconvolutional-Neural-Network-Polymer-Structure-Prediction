import keras.backend as K
from vis.visualization import visualize_saliency

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
import vis_modules
import hotspot_file_01
import feature_extract_01

from scipy.interpolate import Rbf

def gen_plot01(model_cnn, X_1, X_2, X_g, tot_atom=7):

    #grad_out = hotspot_file_01.rdf_saliency(model_pick=model_cnn, x_e1=X_1, x_e2=X_2, x_ge=X_g)
    #computing the grad tensor
    grad_out = vis_modules.saliency_01(model=model_cnn, X=[X_1[np.newaxis, :, :, :], X_2[np.newaxis, :, :, :],
                                                               X_g[np.newaxis, :]])

    df_1 = grad_out[0][0]
    df_2 = grad_out[1][0]
    dg = grad_out[2][0]


    f_1 = reduce_dim_max(df_1=df_1, df_2=df_2)


    f_1 = f_1.reshape(int(len(f_1)/tot_atom), tot_atom)

    plot_1d(f_1)




    return df_1, df_2, dg


def occlusion_plot01(model_cnn, X_1, X_2, X_g, tot_atom=7):
    # grad_out = hotspot_file_01.rdf_saliency(model_pick=model_cnn, x_e1=X_1, x_e2=X_2, x_ge=X_g)
    # computing the grad tensor
    O1, O2, Og = vis_modules.occlusion_rdf01(model_cnn, X_1, X_2, X_g) #pass on X_1, X_2 and X_g. -> the later global features



    O_max = reduce_dim_max(O1[np.newaxis, :, :, :], O2[np.newaxis, :, :, :]) #need to compare O_1 and O_2 to extract the spatial max

    #O_max = O_max.reshape(tot_atom, int(len(O_max) / tot_atom))
    O_max = O_max.reshape(int(O_max.shape[0] / tot_atom), tot_atom) #sep

    plot_1d(O_max)

    return O_max, Og


def reduce_dim_max(df_1, df_2):

    df_1, df_2 = df_1[0, :, :, :], df_2[0, :, :, :]


    f_1 = np.amax(df_1, axis=(0, 1))

    print "******investigate f************"
    #investigate_f(f_1)



    f_2 = np.amax(df_2, axis=(0, 1)) #taking max along z-x domain, not the property domain
    f_t = np.concatenate((f_1[None, :], f_2[None, :]), axis=0)
    f_max = np.amax(f_t, axis=0)

    #investigate_f(f_2)



    return f_max


def investigate_X(X, tot_atom=7):



    X_p = X.reshape(X.shape[0], X.shape[1], int(X.shape[2]/tot_atom), tot_atom)

    for i in range(0, tot_atom):
        idx_to_inv = np.where(X_p[:, :, 0:5, i] > 0)
        print "i: ", i
        print "idx_to_inv: ", idx_to_inv



    return None


def investigate_f(f, tot_atoms=7):

    f_p = f.reshape(int(f.shape[0] / tot_atoms), tot_atoms)


    for i in range(0, tot_atoms):
        idx_to_inv = np.where(f_p[0:5, i] > 0)
        print "i: ", i
        print "idx_to_inv: ", idx_to_inv



    return None



#def


def plot_1d(Z):

    #R_v = np.asarray([0, 0.5, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    R_v = np.load('R_v.npy')

    #R_v = np.asarray([0, 0.5, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    R_v = R_v[0, :]

    print "R_v: ", R_v

    r_l = np.linspace(0, 6.0, 60)

    color_mat = ['k', 'k', 'b', 'k', 'b', 'r', 'b']
    marker_mat = ['o', 's', 'o', '+', 's', 'o', '+']
    label_mat = ['Alkane C', 'Aromatic C', 'N in primary amine', 'C in epoxide', 'N in secondary Amine',
                 'O  in Epixide', 'N in tertiary amine']
    plt.figure()

    for j in range(Z.shape[1]):

        rbfi = Rbf(R_v[:-1], Z[:-1, j], epsilon=0.25)  # radial basis function interpolator instance
        di = rbfi(r_l)
        di[di <= 0] = 0
        plt.scatter(R_v, Z[:, j], s=100, marker=marker_mat[j], color=color_mat[j], label=label_mat[j])
        plt.plot(r_l, di, color=color_mat[j])


    plt.xlabel('Radial distance from CNT atom ($\AA$)', fontsize=18)
    plt.ylabel('$\epsilon$ (kcal/mol-$\AA$)', fontsize=18)
    plt.ylim([0, 0.05])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper right')
    plt.savefig('occ_3rr5epo_00031.svg', bbox_inches="tight")
    #plt.savefig('occ_3rr2epo_00010.png')
    #plt.savefig('fig_120_3rr5epo_00020.eps')
    plt.show()

    return None




def bar_plot(a_vec):

    N = np.arange(len(a_vec))

    plt.bar(N, a_vec)
    plt.xticks(N, ('$\phi$', '$a_{g,1}$', '$a_{g,2}$', '$a_{g,3}$', '$a_{g,4}$', '$a_{g,5}$'))
    plt.ylabel('$\epsilon$')
    #plt.savefig('bar_3rr2epo_00010.eps')
    plt.savefig('bar_3rr2epo_00010.png')
    plt.show()

    return None



def contor_plot(X, tot_atom=7, idx_atom=5, idx_r=1):

    X = X.reshape(X.shape[0], X.shape[1], int(X.shape[2] / tot_atom), tot_atom)

    ax = sns.heatmap(X[:, :, idx_r, idx_atom], square=False)
    plt.show(ax)
    #ax.set_xlabel('Node features')
    #ax.set_ylabel('Node #')

    for i in range(0, 14):
        ax = sns.heatmap(X[:, :, i, idx_atom])
        plt.show(ax)


    return None




if __name__=="__main__":
    X_1 = np.load('X_13.npy')   #corresponds to RDF-1
    X_2 = np.load('X_23.npy')   #corresponds to RDF-2
    X_g = np.load('X_g3.npy')   #corresponds to global feaures
    model_cnn = load_model('model_2.h5')    #import RDF-CNN model
    og_indices = np.load('file_indices.npy')    #import the indices of the files after they're shuffled

    min_xg = np.min(X_g, axis=0) #minimum value of the
    np.save('min_xg.npy', min_xg)

    X_1, X_2, X_g = feature_extract_01.pick_og_features(X1=X_1, X2=X_2, Xg=X_g) #pick "unique", i.e. non-augmented features.

    #A = [0.04315454, 0.02538007, 0.02762747, 0.05325091, 0.02449626, 0.14951509]
    #A = np.asarray(A)

    #you can loop over multiple models to perform occlusion. as it is there are 25 test models

    for i in range(20, 21):
        #df_1, df_2, dg = gen_plot01(model_cnn=model_cnn, X_1=X_1[i, :, :, :], X_2=X_2[i, :, :, :], X_g=X_g[i, :])
        O_max, Og = occlusion_plot01(model_cnn=model_cnn, X_1=X_1[i, :, :, :], X_2=X_2[i, :, :, :], X_g=X_g[i, :])
        #print "dg: ", dg
        #print "O_max: ", O_max
        bar_plot(Og[:-1])

        #investigate X:
        #investigate_X(X=X_1[i, :, :, :])
        #investigate_X(X=X_2[i, :, :, :])
        #X_c = reduce_dim_max(X_1, X_2).reshape(14, 7)

        #print "X_c: ", X_c
        print "Og: ", Og
        #print "idx_to_go: ", np.where(X_c[0:5, :])
        #print og_indices[86]

    ###read MD model
    #file_list = ['./MDfiles/select03/3rr4epo.00023']
    ##this block of codes
    file_list = ['./MDfiles/select03/3rr2epo.00013']
    df_list, z_list, _ =  hotspot_file_01.read_MD_model(file_list=file_list)
    df = df_list[0]
    z_len = z_list[0]
    vis_modules.get_interface_atoms(df_xyz=df)
    rbf, R = feature_extract_01.CNT_atoms(df=df, z_len=z_len)

    r_d, RBF_d = feature_extract_01.discretize_features(rbf)

    #print r_d, RBF_d[100, :, 0]


    ##print "rbf shape: ", rbf.shape
    #for i in range(0, 1):
        #feature_extract_01.plot_features(R, rbf[100, :, :], i)



    r_d = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75])
    RBF_d = np.insert(RBF_d, [0], np.zeros((RBF_d.shape[0], 4, RBF_d.shape[2])), axis=1)[:, :-1, :]
    print r_d.shape
    print RBF_d.shape

    #for i in range(0, 1):
        #feature_extract_01.plot_features(r_d, RBF_d[100, :, :], i)
