import numpy as np
import pandas as pd
import hotspot_file_01
import matplotlib.pyplot as plt

import pymp
import math
import multiprocessing

from scipy import optimize

from sklearn import preprocessing

def detect_surface_cnt(df, type_choose=22, **kwargs):

    if ('cnt_bounds' in kwargs):
        cnt_bounds = kwargs['cnt_bounds']
    else:
        cnt_bounds = hotspot_file_01.extract_cnt_coords(df)

    df_other = df.loc[~(df['type'] == type_choose)]
    df_cnt = df.loc[df['type'] == type_choose]


    df_1 = df_cnt.loc[:, ['x', 'y', 'z']]
    df_2 = df_other.loc[:, ['x', 'y', 'z']]
    Z = compute_distance(df_1.as_matrix(), df_2.as_matrix())


    print "Z: ", Z

    z = np.amin(Z)
    idx_val = (np.argmin(Z))
    idx_val = np.unravel_index(idx_val, Z.shape)

    #print 0.5*(df_1.as_matrix()[idx_val[1], :] + df_2.as_matrix()[idx_val[0], :])


    return None





def compute_distance(cnt_array, cnt_other_array):

    dist_list = []
    sort_dist_list = []
    idx_list = []
    N = 5
    points = cnt_array

    for j in range(0, len(cnt_other_array)):

        single_point = cnt_other_array[j, :]

        dist = np.sum(((points - single_point)**2), axis=1)
        dist = np.sqrt(dist)

        sort_dist = np.sort(dist)
        sort_dist = sort_dist[:N]

        idx_sort = np.argsort(dist)
        idx_sort = idx_sort[:N]


        dist_list.append(dist)
        sort_dist_list.append(sort_dist)
        idx_list.append(idx_sort)



    return np.asarray(dist_list), sort_dist_list, idx_list



#start reading the line
def track_fun_bond(df_t, fun_type_list):

    bond_list = []

    for t in range(0, len(df_t)):
        df = df_t[t]
        z_array = check_bond(df=df, fun_type_list=fun_type_list)
        bond_list.append(z_array)




    return np.asarray(bond_list)



def check_bond(df, fun_type_list, cnt_type=22):

    #this function is to determine if the bonding between the CNT and the functional group is retained
    #fun_type -> functional group type
    df_cnt = df.loc[df['type'] == cnt_type]
    df_1 = df_cnt.loc[:, ['x', 'y', 'z']]

    z_list = []
    #the Z-list constains the mean distance of functional group with its nearest neighbors

    #we do a loop as fun_type is a list, there may be more than one atom i'm interested in
    for i in range(0, len(fun_type_list)):
        fun_type = fun_type_list[i]
        df_fun = df.loc[df['type'] == fun_type]

        df_fun_loc = df_fun.loc[:, ['x', 'y', 'z']]


        #fun_loc -> multiple locations where the functional groups are located
        _, Z2, _ = compute_distance(df_1.as_matrix(), df_fun_loc.as_matrix())
        z_mean = np.mean(Z2)
        z_list.append(z_mean)



    return np.asarray(z_list)






def local_features(df_cnt, df_other):

    #This function provides the local features for each convololution box
    cnt_loc = df_cnt.loc[:, ['x', 'y', 'z']]

    #other atom
    other_loc = df_other.loc[:, 'mass', 'x', 'y', 'z']
    cnt_mat = cnt_loc.as_matrix()
    other_mat = other_loc.as_matrix()

    FR = feature_equation_01(cnt_mat=cnt_mat, other_mat=other_mat)



    return FR




def feature_equation_01(atom_bound, unique_type, bounds=np.asarray([2.0, 5.0, 10.0]), intervals=np.asarray([0.1, 0.1, 0.5])):

    #Z is a function of r
    #here insert the list of
    type_choose = 22

    #the first thing to do is detect the atoms

    #The idea is that for each unique type we will create an array
    b = len(unique_type)
    r1 = np.linspace(0, bounds[0], bounds[0]/intervals[0], endpoint=False)
    r2 = np.linspace(bounds[0], bounds[1], (bounds[1] - bounds[0])/intervals[1], endpoint=False)
    r3 = np.linspace(bounds[1], (bounds[2]), (bounds[2]-bounds[1])/intervals[2] + 1)


    r = np.concatenate((r1, r2, r3), axis=0)


    x = np.zeros((len(r), b))

    for k in range(0, b):
        type = unique_type[k]
        sigma_r = 1

        #Seggregate by type
        #df_cnt = df.loc[df['type'] == type_choose]
        #df_atom = df.loc[df['type'] == type]

        idx_cnt = np.where(atom_bound[:, 1] == type_choose)
        idx_fun = np.where(atom_bound[:, 1] == type)

        cnt_mat = atom_bound[idx_cnt[0]]
        fun_mat = atom_bound[idx_fun[0]]


        #indexing out co-ordinates only
        cnt_mat = cnt_mat[:, 2:5]
        fun_mat = fun_mat[:, 2:5]


        #cnt_mat = df_cnt.loc[:, ['x', 'y', 'z']].as_matrix()
        #atom_mat = df_atom.loc[:, ['x', 'y', 'z']].as_matrix()
        dist_mat = compute_distance(cnt_mat, fun_mat)

        d = dist_mat[0].flatten()

        for count, r_val in enumerate(r):
            x[count, k] = np.sum(np.divide((np.exp((-(r_val - d)**2)/(sigma_r**2))), d))




    return r, x

#-----------------------------------------------------------------------------
def feature_equation_02(r, cnt_mat, other_mat, other_type, unique_type=np.asarray([2, 4, 9, 12, 15, 18, 16]), type_choose=22):

    b = len(unique_type)
    cnt_mat = np.transpose(cnt_mat)


    x = np.zeros((len(r), b))

    for k in range(0, b):
        type = unique_type[k]
        #sigma_r = .05
        sigma_r = 0.05
        #Seggregate by type
        #df_cnt = df.loc[df['type'] == type_choose]
        #df_atom = df.loc[df['type'] == type]


        #fun mat

        idx_fun = np.where((other_type.flatten()) == type)[0]
        fun_mat = other_mat[idx_fun]


        dist_mat = compute_distance(cnt_mat, fun_mat)


        #the correct one is in fun mat

        d = dist_mat[0].flatten()

        d = d[d>0.001]
        d = d[d < 10.0]


        #print d[d<2.0]


        #print d.shape
        #print np.amax(d)
        #print np.amin(d)

        for count, r_val in enumerate(r):
            x[count, k] = np.sum(np.divide((np.exp((-(r_val - d)**2)/(sigma_r**2))), d))






    return x



####-------------------------------------------------------------------------------
def feature_troubleshoot(df, type_choose=22):

    # this is the block to featurize
    df_other = df.loc[~(df['type'] == type_choose)]
    df_cnt = df.loc[df['type'] == type_choose]
    FR = local_features(df_cnt=df_cnt, df_other=df_other)


    return None


def detect_locations(df, threshold=4.0, type_choose=22, **kwargs):


    if 'fun_type' in kwargs:
        fun_type = kwargs['fun_type']
        df_fun = df.loc[df['type'] == fun_type]

    elif 'fun_id' in kwargs:
        fun_id = kwargs['fun_id']
        df_fun = df.loc[df['id'] == fun_id]


    df_cnt = df.loc[df['type'] == type_choose]

    index_fun = df_fun.loc[:, ['id']].as_matrix()
    index_cnt = df_cnt.loc[:, ['id']].as_matrix()

    df_1 = df_cnt.loc[:, ['x', 'y', 'z']]
    df_2 = df_fun.loc[:, ['x', 'y', 'z']]
    z1, _, _ = compute_distance(df_1.as_matrix(), df_2.as_matrix())



    idx_choose = np.where((np.asarray(z1) < threshold))

    id_fun = np.unique(index_fun[idx_choose[0]])
    id_cnt = np.unique(index_cnt[idx_choose[1]])

    df_fun_out = df_fun[df_fun.id.isin(id_fun)]
    df_cnt_out = df_cnt[df_cnt.id.isin(id_cnt)]


    z1 = np.asarray(z1)


    return df_fun_out, df_cnt_out




def plot_features(R, input_array, idx):

    plt.plot(R, input_array[:, idx])
    plt.xlabel('r (A)')
    plt.ylabel('Z-value')
    plt.savefig('fig_Zvalue.eps')
    plt.show()

    return None




def plot_features02(R, input_array, R2, input_array2, idx):

    plt.plot(R, input_array[:, idx], R2, input_array2[:, idx], 'ro')
    plt.xlabel('r (A)')
    plt.ylabel('Z-value')
    plt.savefig('fig_Zvalue.eps')
    plt.show()

    return None

#def coarsen_features


def feature_cleanup(rho_mat):

    rho_out = rho_mat.copy()

    rho_mat[rho_mat < 1.0e-3] = 0

    mean_val = np.mean(rho_mat, axis=0)
    std_val = np.std(rho_mat, axis=0)

    for k in range(0, rho_mat.shape[0]):
        rho_out[k, :, :] = rho_mat[k, :, :] - mean_val
        rho_out[k, :, :] = np.divide(rho_out[k, :, :], std_val)


    rho_out[np.isnan(rho_out)] = 0

    return rho_out


def CNT_atoms(df, z_len, bound_default=np.asarray([1.0, 2.0, 5.0]), type_choose=22):

    #the goal of this function is to flatten the CNT
    #and for each atoms track surrounding atoms

    df_cnt = df.loc[df['type']==type_choose]
    df_other = df
    #df_other = df.loc[df['type']!=type_choose]

    y_min =  np.amin(df_cnt.loc[:, 'y'].as_matrix())
    y_max = np.amax(df_cnt.loc[:, 'y'].as_matrix())

    y_mid = 0.5*(y_min + y_max)

    df_cnt1 = df_cnt.loc[df['y'] > y_mid]

    mat_xy = df_cnt.loc[:, ['x', 'y']].as_matrix()



    #RBF_array, r_v = compute_RDF(df_cnt, df_other, z_len)
    RBF_array, r_v = compute_RDF(df_cnt=df_cnt, df_other=df_other, z_len=z_len, bounds=np.asarray([1.0, 2.0, 5.0]), intervals=[0.01, 0.01, 0.01])



    return RBF_array, r_v


def CNT_atoms02(df, z_len, bound_default=np.asarray([1.0, 2.0, 5.0]), fun_type = [12, 9, 15, 18], type_choose=22):

    #the goal of this function is to flatten the CNT
    #and for each atoms track surrounding atoms

    df_cnt = df.loc[df['type']==type_choose]
    df_other = df




    y_min =  np.amin(df_cnt.loc[:, 'y'].as_matrix())
    y_max = np.amax(df_cnt.loc[:, 'y'].as_matrix())

    y_mid = 0.5*(y_min + y_max)

    df_cnt1 = df_cnt.loc[df['y'] > y_mid]

    cnt_mat = df_cnt.loc[:, ['x', 'y', 'z']].as_matrix()
    cnt_id = df_cnt.loc[:, ['id']].as_matrix()



    ###----------------------
    # choose dataframe with a different type
    df_fun = df.loc[df['type'].isin(fun_type)]
    fun_mat = df_fun.loc[:, ['x', 'y', 'z']].as_matrix()

    fun_type = df_fun.loc[:, ['type']].as_matrix()


    ###need to keep track of both the
    D, _, _ = compute_distance(cnt_array=cnt_mat, cnt_other_array=fun_mat)

    #RBF_array, r_v = compute_RDF(df_cnt, df_other, z_len)
    RBF_array, r_v = compute_RDF(df_cnt=df_cnt, df_other=df_other, z_len=z_len, bounds=np.asarray([1.0, 2.0, 5.0]), intervals=[0.01, 0.01, 0.01])

    ###define an fun array
    #RBF_fun = np.zeros_like(RBF_array)
    R_fun = create_bounds()
    RBF_fun = np.zeros((RBF_array.shape[0], len(R_fun), RBF_array.shape[2]))
    ##2.0 A is the threshold

    if np.any(D < 2.0):

        #find where D is less than 2.0
        idx = np.where(D < 2.0)

        #index out cnt id where functionalization has taken place
        idx_cnt = idx[1]
        idx_fun = idx[0]

        print "idx_cnt: ", idx_cnt
        print "idx_fun: ", idx_fun

        type_fun = (fun_type[idx_fun]).flatten()
        print "type fun: ", type_fun

        #first check fun_type
        opt_mat = create_fun_vector(cnt_mat=cnt_mat, idx_mat=idx_cnt, type_mat=type_fun)

        print "opt_mat: "
        print opt_mat



        #print np.where(fun_mat > 0)

        count = 0
        #we showed here
        for n in idx_cnt:

            fun_choose = fun_mat[idx_fun[count], :]
            fun_choose = fun_choose[:, None]
            type_n = type_fun[count]

            print "fun choose: ", fun_choose
            print "type_n: ", type_n

            df_other = df.loc[df['type']!=type_n]
            other_mat = df_other.loc[:, ['x', 'y', 'z']].as_matrix()
            other_type = df_other.loc[:, ['type']].as_matrix()

            other_image = check_periodic(fun_choose[2], other_mat, z_len)
            ##print other_image[2311, :]

            #troubleshoot


            RBF_fun[n, :, :] = feature_equation_02(r=R_fun, cnt_mat=fun_choose, other_mat=other_image, other_type=other_type, unique_type=np.asarray([2, 4, 9, 14, 15, 18, 16]))

            count += 1


    ##read the rbf_array




    return RBF_fun, R_fun


def compute_RDF(df_cnt, df_other, z_len, bounds=np.asarray([1.0, 2.0, 5.0]), intervals=np.asarray([0.01, 0.01, 0.01])):


    #parameters to compute the RDF
    r1 = np.linspace(0, bounds[0], bounds[0] / intervals[0], endpoint=False)
    r2 = np.linspace(bounds[0], bounds[1], (bounds[1] - bounds[0]) / intervals[1], endpoint=False)
    r3 = np.linspace(bounds[1], (bounds[2]), (bounds[2] - bounds[1]) / intervals[2] + 1)

    r = np.concatenate((r1, r2, r3), axis=0)

    unique_type = np.asarray([2, 4, 9, 12, 15, 18, 23])
    #unique_type = np.asarray([2, 4, 9, 12, 15, 18, 3]) #change this if things don't work

    #for each CNT atom I want to compute an RDF
    cnt_id = df_cnt.loc[:, ['id']].as_matrix()

    #print "fun-CNT id: ", np.where(cnt_id == 3395)

    other_id = df_other.loc[:, ['id']].as_matrix()

    other_type = df_other.loc[:, ['type']].as_matrix()

    #print "fun id: ", np.where(other_id == 3413)

    cnt_mat = df_cnt.loc[:, ['x', 'y', 'z']].as_matrix()
    other_mat = df_other.loc[:, ['x', 'y', 'z']].as_matrix()

    #print "Troubleshoot: "
    #print cnt_mat[254, :]
    #print other_mat[3381, :]
    #print other_type[3381]

    #print cnt_mat.shape

    # shared array
    RBF_array = pymp.shared.array((len(cnt_mat), len(r), len(unique_type)), dtype='float64')

    with pymp.Parallel(multiprocessing.cpu_count()) as p:
        for i in p.range(len(cnt_mat)):
        #for i in range(1):
            cnt_choose = cnt_mat[i, :]
            #print other_mat[2311, :]
            cnt_choose = cnt_choose[:, None]

            other_image = check_periodic(cnt_choose[2], other_mat, z_len)
            ##print other_image[2311, :]

            #troubleshoot


            RBF_array[i, :, :] = feature_equation_02(r=r, cnt_mat=cnt_choose, other_mat=other_image, other_type=other_type)




    ###this is an extra block to troubleshoot where
    other_test = df_other.loc[df_other['type']!=22]
    test_mat = other_test.loc[:, ['x', 'y', 'z']].as_matrix()
    D, _, _ = compute_distance(cnt_array=cnt_mat, cnt_other_array=test_mat)

    #print D[D < 2.0]
    #print np.where([D < 2.0])




    return RBF_array, r




def check_periodic(cnt_z, other_mat, z_len):

    other_z = other_mat[:, 2]  #z-coordinate

    z_min = np.amin(other_mat[:, 2])
    z_max = np.amax(other_mat[:, 2])
    z_mid = 0.5*(z_min + z_max)



    if cnt_z < z_mid:
        other_z[other_z > z_mid] = other_z[other_z> z_mid] - z_len
    else:
        other_z[other_z < z_mid] = other_z[other_z < z_mid] + z_len


    other_mat[:, 2] = other_z


    return other_mat



def discretize_features(RBF, new_intv=np.asarray([0.5, 0.25, 0.5]), old_intv=np.asarray([0.01, 0.01, 0.01]), bounds=np.asarray([1.0, 2.0, 5.0])):

    factor_v = (np.divide(new_intv, old_intv)).astype(int)

    #idx_new = (np.divide(bounds, new_intv)).astype(int)
    #idx_old = (np.divide(bounds, old_intv)).astype(int)
    idx_new = np.zeros((len(new_intv), ))
    idx_old = np.zeros_like(idx_new)

    temp_val = 0
    bound_old = 0
    temp2 = 0


    for idx in range(0, len(new_intv)):
        idx_new[idx] = temp_val + int((bounds[idx] - bound_old)/new_intv[idx])
        idx_old[idx] = temp2 + int((bounds[idx] - bound_old) / old_intv[idx])

        temp_val = int(idx_new[idx])
        bound_old = int(bounds[idx])
        temp2 = int(idx_old[idx])



    idx_new = np.insert(idx_new, 0, 0)
    idx_old = np.insert(idx_old, 0, 0)
    idx_new = idx_new.astype(int)
    idx_old = idx_old.astype(int)

   # print idx_new
    #print idx_old

    #computing the size of the new array
    r1 = np.linspace(0, bounds[0], bounds[0] / new_intv[0], endpoint=False) + 0.5*new_intv[0]
    r2 = np.linspace(bounds[0], bounds[1], (bounds[1] - bounds[0]) / new_intv[1], endpoint=False) + 0.5*new_intv[1]
    r3 = np.linspace(bounds[1], (bounds[2]), (bounds[2] - bounds[1]) / new_intv[2] + 1) + 0.5*new_intv[2]

    r = np.concatenate((r1, r2, r3), axis=0)

    RBF_out = np.zeros((RBF.shape[0], len(r), RBF.shape[2]))

    #print "r: "
    #print r

    for k in range(0, RBF.shape[0]):

        count = 0
        #with pymp.Parallel(multiprocessing.cpu_count()) as p:
        for i in range(len(new_intv)):

            first_idx = idx_new[i]
            last_idx = idx_new[i+1]

            #print first_idx, last_idx
            #print idx_new, idx_old

            #print "i :", i
            #print "first_idx: ", first_idx
            #print "last_idx: ", last_idx
            #print "factor_f: ", factor_v[i]

            Z = RBF[k, :, :]
            #print Z


            for j in range(0, last_idx - first_idx):

                #print j
                #print count
                #print idx_old[i] + j*factor_v[i], idx_old[i] + (j+1)*factor_v[i]
                #print Z[idx_old[i] + j*factor_v[i]:idx_old[i] + (j+1)*factor_v[i], :]

                try:
                    max_out = np.amax(Z[(idx_old[i] + j*factor_v[i]):(idx_old[i] + (j+1)*factor_v[i]), :], axis=0)
                except:
                    max_out = np.zeros(RBF.shape[2], )

                #replacing using a sum
                #print "Max out"
                #print max_out
                #max_out = np.sum(Z[(idx_old[i] + j * factor_v[i]):(idx_old[i] + (j + 1) * factor_v[i]), :], axis=0)

                RBF_out[k, count, :] = max_out
                count += 1

                #print "count: ", count



    #print "r"
    #print r

    #cleaning up RBF
    RBF_out[RBF_out < 0.00001] = 0

    return r, RBF_out





###The following 3 functions are used to fit the CNT coordinates to a a circle.
##first to extract the x-y coordinates for each df in list

def fit_circle(df_list, type_choose=22):

    center_list = []
    R_list = []

    #with pymp.Parallel(multiprocessing.cpu_count()) as p:
    for i in range(len(df_list)):


        df = df_list[i]
        df_cnt = df.loc[df['type'] == type_choose]

        xy_mat = df_cnt.loc[:, ['x', 'y']].as_matrix()

        x_m = np.mean(xy_mat[:, 0])
        y_m = np.mean(xy_mat[:, 1])
        center_estimate = x_m, y_m
        center, ier = optimize.leastsq(f, center_estimate, args=(xy_mat[:, 0], xy_mat[:, 1]))
        xc, yc = center
        Ri = calc_R(xy_mat[:, 0], xy_mat[:, 1], *center)
        R = Ri.mean()
        residu = np.sum((Ri - R) ** 2)

        center_list.append(center)
        R_list.append(R)
        #plt.plot(xy_mat[:, 0], xy_mat[:, 1], 'ro')
        #plt.show()
        #plot_data_circle(x=xy_mat[:, 0], y=xy_mat[:, 1], xc=xc, yc=yc, R=R)


    print center_list
    print R_list

    return center_list, R_list



def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)

    return Ri - Ri.mean()


def plot_data_circle(x,y, xc, yc, R):
    f = plt.figure( facecolor='white')  #figsize=(7, 5.4), dpi=72,
    plt.axis('equal')

    theta_fit = np.linspace(-math.pi, math.pi, 180)

    x_fit = xc + R*np.cos(theta_fit)
    y_fit = yc + R*np.sin(theta_fit)
    plt.plot(x_fit, y_fit, 'b-' , label="fitted circle", lw=2)
    plt.plot([xc], [yc], 'bD', mec='y', mew=1)
    plt.xlabel('x')
    plt.ylabel('y')
    # plot data
    plt.plot(x, y, 'ro', label='data', mew=1)

    plt.legend(loc='best',labelspacing=0.1 )
    plt.grid()
    plt.show()
    plt.title('Least Squares Circle')


    return None


####with the above three functions, we found the best fit circle,

def transfor_rot(df_list, RBF_in, targets, z_cords, type_choose=22, N_alpha = 20):

    del_alpha = np.linspace(0, 2*(math.pi), N_alpha, endpoint=False)


    #This function tranforms based on the rotational symmetry
    #for each df in df_list, extract xyz, rotate it by alpha
    center_list, R_list = fit_circle(df_list)
    center_out = []

    CNT_out = []
    RBF_out = []

    for i in range(0, len(df_list)):

        df = df_list[i]
        df_cnt = df.loc[df['type']==type_choose]

        #need to extract RBF
        RBF_mat = RBF_in[i]

        #extract box bounds
        z_bounds = z_cords[i]
        z_min = z_bounds[0]
        z_max = z_bounds[1]

        cnt_id = df_cnt.loc[:, ['id']].as_matrix()
        xyz_mat = df_cnt.loc[:, ['x', 'y', 'z']].as_matrix()

        xyz_trans = xyz_mat.copy()
        xyz_trans[:, 0:2] = xyz_mat[:, 0:2] - center_list[i]
        Z_val = xyz_mat[:, 2]

        R_p = np.linalg.norm(xyz_trans[:, 0:2], axis=1)


        theta_0 = np.arctan2(xyz_trans[:, 1], xyz_trans[:, 0])


        #initialize a random array
        cnt_aug = np.zeros((len(del_alpha), xyz_mat.shape[0], xyz_mat.shape[1]))
        cnt_aug[:, :, 2] = Z_val



        for j in range(len(del_alpha)):

            theta = theta_0 + del_alpha[j]
            cnt_aug[j, :, 0] = np.multiply(R_p, np.cos(theta))
            cnt_aug[j, :, 1] = np.multiply(R_p, np.sin(theta))


        cnt_aug[:, :, 0:2] = cnt_aug[:, :, 0:2] + center_list[i]
        cnt_final = transform_trans(cnt_rot=cnt_aug, z_min=z_min, z_max=z_max)

        #RBF_temp repeat
        RBF_t = np.repeat(RBF_mat[None, :, :], len(cnt_final), axis=0)


        #convert numpy array to list:
        cnt_list = array_to_list(cnt_final)
        RBF_list = array_to_list(RBF_t)

        if len(CNT_out) > 0:
            CNT_out = CNT_out + cnt_list
            RBF_out = RBF_out + RBF_list
        else:
            CNT_out = cnt_list
            RBF_out = RBF_list




    y_out = (np.repeat(targets[:, None], repeats=len(cnt_final), axis=1)).flatten()


        ##append array here

    #plt.plot(cnt_aug[0, :, 0], cnt_aug[0, :, 1], 'ko')
    #plt.plot(cnt_aug[5, :, 0], cnt_aug[5, :, 1], 'ro')
    #plt.plot(cnt_aug[8, :, 0], cnt_aug[8, :, 1], 'bo')
    #plt.show()

    #print "z_lo, z_hi: ", z_min, z_max
    #print "cnt_aug: "
   # print cnt_aug[0,:10, :]
    #print "cnt_final: "
    #print cnt_final[0, :10, :]
    #print cnt_final[2, :10, :]
    #print cnt_final[10, :10, :]

    #print cnt_final.shape

    #plt.plot(cnt_final[0, :, 1], cnt_final[0, :, 2], 'go')
    #plt.plot(cnt_final[11, :, 1], cnt_final[21, :, 2], 'ro')
    #plt.show()


    #print cnt_final.shape
    #idx_1 = np.where(cnt_final < 30.0)
    #bool_val = cnt_final < 30.0


    #cnt_choose = cnt_final[bool_val[:, :, 2]]

    #cnt_1 = cnt_final[0, :, :]
    #bool2 = cnt_1 < 30.0
    #cnt_c2 = cnt_1[bool2[:, 1]]



        ## we can compute the angle f





    #cnt_aug = transform_trans(cnt_rot=cnt_aug, z_min=)
    #CNT_out = np.concatenate(CNT_out, axis=0)

    print "Troubleshoot: "
    print CNT_out[0].shape
    print (CNT_out[0])[:, 2]
    print RBF_out[0].shape
    print len(CNT_out)
    print len(RBF_out)

    #Troubleshoot
    Z = CNT_out[0]
    X = Z.reshape(CNT_out[0].shape[1], CNT_out[0].shape[2])
    X2 = Z.reshape(CNT_out[1].shape[1], CNT_out[1].shape[2])

    #plt.plot(X[:, 1], X[:, 2], 'go')
    #plt.plot(X2[:, 1], X2[:, 2], 'go')
    #plt.plot(cnt_final[21, :, 1], cnt_final[21, :, 2], 'ro')
    #plt.show()


    #RBF_out = np.asarray(RBF_out)


    #CNT_f = CNT_out.reshape(-1, CNT_out.shape[2], CNT_out.shape[3])
    #RBF_f = RBF_out.reshape(-1, RBF_out.shape[2], RBF_out.shape[3])

    return CNT_out, RBF_out, y_out, center_list



def linearize_cicle(cnt_list, center_list):

    theta_list = []

    for k in range(0, len(cnt_list)): #k is th enumbner of models

        cnt_xy = cnt_list[k][:, 0:2]
        theta = np.arctan(np.divide((cnt_xy[:, 1] - center_list[k][1]), (cnt_xy[:, 0] - center_list[k][0])))
        theta_list.append(theta)


    return theta_list


def transform_trans(cnt_rot, z_min, z_max, N_t = 20):

    #cnt_rot -> 3-D array [#rot copies, cnt_atoms, coords]
    #N_t -> number of translational copies


    delta = np.linspace(0, (z_max - z_min), N_t, endpoint=False)
    Z_trans = np.zeros((cnt_rot.shape[0]*(N_t), cnt_rot.shape[1]))
    cnt_trans = np.zeros((cnt_rot.shape[0]*(N_t), cnt_rot.shape[1], cnt_rot.shape[2]))

    for k in range(0, cnt_rot.shape[0]):
        Z_pick = cnt_rot[k, :, 2]


        for i in range(0, cnt_rot.shape[1]):
            Z_trans[k*(N_t):(k+1)*(N_t), i] = Z_pick[i] + delta
            cnt_trans[k*N_t:(k+1)*N_t, i, 0:2] = cnt_rot[k, i, 0:2]

    Z_trans[Z_trans > z_max] = Z_trans[Z_trans > z_max] - (z_max - z_min)
    cnt_trans[:, :, 2] = Z_trans
    #create a new array
    #for each rotational copy, create a translational copy as well:


    return cnt_trans




def discretize_2D(cnt_array, RBF_array, ts_bool = False, center_array= np.asarray([30.0, 30.0]), grid = np.asarray([4.0, 2.0]), x_min=25.5, x_max = 35.5, z_min = 8.0, z_max = 52.0):

    #cnt_array -> (k x i x j) k-> data # i -> cnt atom # j -> coords array -> k lists of (i * j) data
    #RBF_array -> k lists

    #specify 2D grid parameters
    num_z = (np.ceil((z_max - z_min)/grid[0])).astype(int)
    num_x = (np.ceil((x_max - x_min)/grid[1])).astype(int)

    z_lin = np.linspace(z_min, z_max, num_z + 1)
    x_lin = np.linspace(x_min, x_max, num_x + 1)

    id_mat1 = np.zeros((num_z, num_x)) #stores corresponding id matrix

    #read in the first entry in the list to get the shape

    RBF_1 =  np.zeros((len(RBF_array), num_z, num_x, (RBF_array[0]).shape[2]))
    RBF_2 =  np.zeros_like(RBF_1)
    bound_mat = np.zeros((len(RBF_array), num_z, num_x, 4))

    sum_val = 0

    ##---------------output---------------------------

    for k in range(0, len(RBF_array)):

        cnt_mat = np.squeeze(cnt_array[k], axis=0)
        rbf_mat = np.squeeze(RBF_array[k], axis=0)


        #print "cnt_mat troubleshoot"
        #print len(cnt_mat)
        #print cnt_mat.shape
        #print rbf_mat.shape
        yc = center_array[1]

        bool_val = cnt_mat < yc

        mat_1 = cnt_mat[bool_val[:, 1]]
        mat_2 = cnt_mat[~bool_val[:, 1]]
        mat_1 = np.delete(mat_1, 1, 1)
        mat_2 = np.delete(mat_2, 1, 1)

        count = 0
        sum_val = 0
        sum_val2 = 0

        idx_m1 = np.where(np.logical_and(np.abs(mat_1[:, 0] - 31.69) < 0.01, np.abs(mat_1[:, 1] - 45.43) < 0.01))
        idx_m2 = np.where(np.logical_and(np.abs(mat_2[:, 0] - 31.69) < 0.01, np.abs(mat_2[:, 1] - 45.43) < 0.01))

        #if len(idx_m1[0]) > 0 or len(idx_m2[0]) > 0:
            #print "M indices"
            #print idx_m1
            #print idx_m2

        #find location


        ##let's process both arrays:

        for i in range(0, num_z):
            for j in range(0, num_x):

                #see location
                z_min = z_lin[i]
                z_max = z_lin[i+1]
                z_c = 0.5*(z_min + z_max)


                x_min = z_lin[j]
                x_max = z_lin[j + 1]
                x_c = 0.5 * (x_min + x_max)

                min_bound = np.asarray([x_lin[j], z_lin[i]])
                max_bound = np.asarray([x_lin[j+1], z_lin[i+1]])
                bound_array = np.concatenate((min_bound, max_bound), axis=None)
                bound_mat[k, i, j, :] = bound_array

                mat_11 = mat_1 - min_bound
                mat_12 = max_bound - mat_1

                bool_1 = np.logical_and(mat_11[:, 0] > 0, mat_11[:, 1] > 0)
                bool_2 = np.logical_and(mat_12[:, 0] > 0, mat_12[:, 1] > 0)
                bool_f = np.logical_and(bool_1, bool_2)
                idx_1 = (np.where(bool_f==True)[0])

                #if ts_bool == True and (np.isin(393, idx_1)):
                 #   print "idx_1: "
                 #   print idx_1

                #select from RBF_mat

                if len(idx_1) > 0:
                    RBF_1[k, i, j, :] = np.amax(rbf_mat[idx_1, :], axis=0)
                else:
                    RBF_1[k, i, j, :] = 0


                id_mat1[i, j] = count

                ####Do the same for the lower half
                mat_21 = mat_2 - min_bound
                mat_22 = max_bound - mat_2

                bool_1 = np.logical_and(mat_21[:, 0] > 0, mat_21[:, 1] > 0)
                bool_2 = np.logical_and(mat_22[:, 0] > 0, mat_22[:, 1] > 0)
                bool_g = np.logical_and(bool_1, bool_2)
                idx_2 = (np.where(bool_g == True)[0])

                #if ts_bool == True:
                #    print "idx_2: "
                  #  print idx_2

                if np.isin(193, idx_2):
                    print idx_2
                    print "rbf_mat(idx_2): "
                    print rbf_mat[idx_2, :]



                if len(idx_2) > 0:
                    RBF_2[k, i, j, :] = np.amax(rbf_mat[idx_2, :], axis=0)
                else:
                    RBF_2[k, i, j, :] = 0

                #count += 1



    #center_array -> 3D array [k, i, j] k-> number of
    #The idea is to divide the 2D box into two halves and then merge them together

    #print RBF_1.shape
    #print RBF_2.shape

    return RBF_1, RBF_2, bound_mat



def RBF_reshape(RBF_array):

    #RBF -> (i, type, r)
    #i -> number of atoms

    RBF_out = RBF_array.reshape(RBF_array.shape[0], (RBF_array.shape[1])*(RBF_array.shape[2]))

    return RBF_out



###This function sets the RBF so that they can be processed using a CNN

def RBF_Setter(df_list, z_list):


    RBF_list = []
    r_list = []

    for i in range(len(df_list)):

        RBF, r_v = CNT_atoms(df_list[i], z_list[i])
        r_d, RBF_d = discretize_features(RBF)

        RBF_d = RBF_reshape(RBF_d)
        RBF_list.append(RBF_d)
        r_list.append(r_d)

        #print total_files[i]
        print RBF_d.shape




    RBF_out = RBF_list

    R_d = np.array(r_list)


    return RBF_out, R_d


def RBF_Setter02(df_list, z_list, total_files):


    RBF_list = []
    r_list = []

    for i in range(len(df_list)):

        RBF, r_v = CNT_atoms(df_list[i], z_list[i])
        r_d, RBF_d = discretize_features(RBF=RBF, new_intv=[0.5, 0.2, 0.5], bounds=np.asarray([1.0, 2.0, 5.0]))

        RBF_d = RBF_reshape(RBF_d)
        RBF_list.append(RBF_d)
        r_list.append(r_d)

        #print total_files[i]
        print RBF_d.shape




    RBF_out = RBF_list

    R_d = np.array(r_list)


    return RBF_out, R_d


def RBF_Setter03(df_list, z_list, unique_type=np.asarray([2, 4, 9, 12, 15, 18, 16]), action_type=np.asarray([9, 12, 15, 18])):


    RBF_list = []
    RBF_f_list = []
    r_list = []
    rf_list = []

    #this block of code indicates where unique type and action type
    #idx = np.where(np.in1d(unique_type, action_type))[0]

    for i in range(len(df_list)):

        RBF, r_v = CNT_atoms(df_list[i], z_list[i])
        RBF_fun , r_f = CNT_atoms02(df_list[i], z_list[i])

        r_d, RBF_d = discretize_features(RBF=RBF, new_intv=[0.5, 0.2, 0.5], bounds=np.asarray([1.0, 2.0, 5.0]))

        #setting bounds and intervals for the "other matrix"
        r_f, RBF_f = discretize_features(RBF=RBF_fun, new_intv=[0.5, 0.2, 0.25], bounds=np.asarray([1.0, 2.0, 8.0]))

        #extracting out the action RBF


        RBF_d = RBF_reshape(RBF_d)
        RBF_f = RBF_reshape(RBF_f)
        RBF_list.append(RBF_d)
        RBF_f_list.append(RBF_f)
        r_list.append(r_d)
        rf_list.append(r_f)
        #print total_files[i]
        print RBF_d.shape




    RBF_out = RBF_list

    R_d = np.array(r_list)
    R_f = np.array(rf_list)


    return RBF_out, RBF_f_list, R_d, R_f



def array_to_list(X):

    #The issue
    split_array = np.split(X, X.shape[0], axis=0)
    list_out = [split_array[i] for i in range(X.shape[0])]



    return list_out


def RBF_eliminator(RBF_1, RBF_2):

    #RBF_out1 = RBF_1[np.all(RBF_1 == 0, axis=(0, 1, 2))]
    #RBF_out2 = RBF_2[np.all(RBF_2 == 0, axis=(0, 1, 2))]


    print "Troubleshoot"

    idx_1 = np.where(np.all(RBF_1 == 0, axis=(0, 1, 2)))
    idx_2 = np.where(np.all(RBF_2 == 0, axis=(0, 1, 2)))

    col_to_drop = np.intersect1d(idx_1[0], idx_2[0])

    RBF_out1 = np.delete(RBF_1, col_to_drop, axis=3)
    RBF_out2 = np.delete(RBF_2, col_to_drop, axis=3)

    col_to_drop = col_to_drop/7
    #return RBF_out1, RBF_out2
    return RBF_out1, RBF_out2, col_to_drop


def configure_targets(file_list, targets):

    #need to tally the file_list with the targets
    target_final = np.zeros((len(file_list), ))
    id_f = np.zeros_like(target_final)

    for i in range(0, len(file_list)):

        file_num = (file_list[i]).rsplit('.')[-1]
        id_f[i] = int(file_num)

    print "Troubelshoot: "
#    print "id_f: ", id_f[32]


    id_f = id_f.astype(int)
    target_final = targets[id_f]

#    print "targets: ", target_final[32]

    return target_final


def normalize_RBF(RBF):


    RBF_mean = RBF.mean(axis=(0,1,2))
    RBF_std = RBF.std(axis=(0,1,2))

    RBF_out = np.divide((RBF - RBF_mean), RBF_std)

    print RBF
    print RBF_out
    print RBF.shape

    return RBF_mean, RBF_std


def partition_data(X, X2, Y, f):

    n = int(f*X.shape[0])

    indices = np.random.permutation(X.shape[0])
    train_idx, test_idx = indices[:n], indices[n:]


    X_t1, X_t2, Y_t = X[train_idx, :, :, :], X2[train_idx, :, :, :], Y[train_idx]
    X_e1, X_e2, Y_e = X[test_idx, :, :, :], X2[test_idx, :, :, :], Y[test_idx]


    return X_t1, X_t2, Y_t, X_e1, X_e2, Y_e



def partition_data02(X, X2, Y, n, f=400):



    N = int(f*n)

    X_t1, X_t2, Y_t = X[:N, :, :, :], X2[:N, :, :, :], Y[:N]
    X_e1, X_e2, Y_e = X[N:, :, :, :], X2[N:, :, :, :], Y[N:]


    return X_t1, X_t2, Y_t, X_e1, X_e2, Y_e


def partition_data02B(X, X2, Xg, Y, n, f=400):

    #Xg is N


    N = int(f*n)

    X_t1, X_t2, Xg_t, Y_t = X[:N, :, :, :], X2[:N, :, :, :], Xg[:N, :], Y[:N]
    X_e1, X_e2, Xg_e, Y_e = X[N:, :, :, :], X2[N:, :, :, :], Xg[N:, :], Y[N:]


    return X_t1, X_t2, Xg_t, Y_t, X_e1, X_e2, Xg_e, Y_e


def partition_data03(X, X2, Xg, Y, n, n2, f=400):

    ##partition data based on validation data
    #n2>n1


    N = int(f*n)
    N2 = int(f*n2)


    X_t1, X_t2, Xg_t, Y_t = X[:N, :, :, :], X2[:N, :, :, :], Xg[:N, :], Y[:N]
    X_v1, X_v2, Xg_v, Y_v = X[N:N2, :, :, :], X2[N:N2, :, :, :], Xg[N:N2], Y[N:N2]
    X_e1, X_e2, Xg_e, Y_e = X[N2:, :, :, :], X2[N2:, :, :, :], Xg[N2:, :], Y[N2:]

    #pick og validation
    print X_v1.shape
    print X_v2.shape
    print Xg_v.shape
    print Y_v
    X_v1, X_v2, Xg_v, Y_v = pick_og_val(X1=X_v1, X2=X_v2, X_g=Xg_v, Y=Y_v)


    return X_t1, X_t2, Xg_t, Y_t, X_v1, X_v2, Xg_v, Y_v, X_e1, X_e2, Xg_e, Y_e


def shuffle_data(X1, X2, Y):


    indices = np.random.permutation(X1.shape[0])

    X_o1, X_o2, Y_o = X1[indices, :, :, :], X2[indices, :, :, :], Y[indices]


    return X_o1, X_o2, Y_o


def shuffle_data02B(X1, X2, Xg, Y):


    indices = np.random.permutation(X1.shape[0])

    X_o1, X_o2, Xg_o, Y_o = X1[indices, :, :, :], X2[indices, :, :, :], Xg[indices, :], Y[indices]


    return X_o1, X_o2, Xg_o, Y_o


def shuffle_by_list(file_list, Y):

    file_array = np.asarray(file_list)
    indices = np.random.permutation(int(len(file_list)))



    file_out = file_array[indices]
    Y_out = Y[indices]



    return file_out, Y_out


def shuffle_by_list02(file_list, Y, other_mat):

    file_array = np.asarray(file_list)
    indices = np.random.permutation(int(len(file_list)))



    file_out = file_array[indices]
    Y_out = Y[indices]
    mat_out = other_mat[indices, :]



    return file_out, Y_out, mat_out



def pick_ogvalues(Y_e, Y_p, f=400):

    num = int(len(Y_e)/f)
    Y1 = np.zeros(num,)
    Y2 = np.zeros_like(Y1)


    for i in range(0, num):

        Y1[i] = Y_e[i*f]
        Y2[i] = Y_p[i*f]


    return Y1, Y2



###OG for validation
def pick_og_val(X1, X2, X_g, Y, f=400):

    num = int(len(Y)/f)
    x1 = np.zeros((num, X1.shape[1], X1.shape[2], X1.shape[3]))
    x2 = np.zeros((num, X2.shape[1], X2.shape[2], X1.shape[3]))
    xg = np.zeros((num, X_g.shape[1]))
    y = np.zeros((num, ))

    for i in range(0, num):

        x1[i, :, :, :] = X1[i*f, :, :, :]
        x2[i, :, :, :] = X2[i*f, :, :, :]
        xg[i, :] = X_g[i*f, :]
        y[i] = Y[i*f]



    return x1, x2, xg, y



def shuffle_by_og(X1, X2, Y, n=10, f=400):



    idx_r = np.random.permutation(int(X1.shape[0]/f))
    idx_e = idx_r[0:n]

    X1_t = np.zeros(((X1.shape[0] - n)*f, X1.shape[1], X1.shape[2], X1.shape[3]))
    X2_t = np.zeros_like(X1_t)
    Y_t = np.zeros(((X1.shape[0] - n)*f, ))

    X1_e = np.zeros((n * f, X1.shape[1], X1.shape[2], X1.shape[3]))
    X2_e = np.zeros_like(X1_e)
    Y_e = np.zeros((n*f, ))

    count_e = 0
    count_t = 0

    for j in range(0, int(X1.shape[0]/f)):

        if j in idx_e:
            X1_e[count_e*f:(count_e + 1)*f, :, :, :] = X1[j*f:(j+1)*f, :, :, :]
            X2_e[count_e * f:(count_e + 1) * f, :, :, :] = X2[j * f:(j + 1) * f, :, :, :]
            Y_e[count_e*f:(count_e + 1)] = Y[j*f:(j+1)*f]

            count_e += 1

        else:
            X1_t[count_t * f:(count_t + 1) * f, :, :, :] = X1[j * f:(j + 1) * f, :, :, :]
            X2_t[count_e * f:(count_t + 1) * f, :, :, :] = X2[j * f:(j + 1) * f, :, :, :]
            Y_t[count_t * f:(count_t + 1)] = Y[j * f:(j + 1) * f]





    return X1_t, X2_t, Y_t, X1_e, X2_e, Y_e




def vis_main01(R_v, W, col_dropped):


    #color scheme
    #R_v -> dimensions of R


    #zero_col = np.zeros(len(R_v), )

    W_o = np.insert(arr=W, obj=col_dropped, values=0, axis=0)







    return W_o



#def multiple_file_read(main_vector, force_vector):


def split_validation_set(X1, X2, Y, n_p = 0.05, f=400):


    n = int(n_p*X1.shape[0])
    N = len(X)

    X_t = X[0:f*n]
    #prepare a validation set
    #X_t



    return None



def CNT_mean(CNT_array):

    CNT_sum = np.mean(CNT_array, axis=0)

    return CNT_sum


def global_features(df, z_len, bound_default=np.asarray([1.0, 2.0, 5.0])):

    RBF_g, _ = CNT_atoms(df=df, z_len=z_len, bound_default=np.asarray([1.0, 2.0, 5.0]))

    RBF_sum = CNT_mean(RBF_g)
    RBF_1 = np.sum(RBF_sum, axis=0)



    return RBF_1



def repeat_global_features(M, f=400):

    M_out = np.zeros((f*M.shape[0], M.shape[1]))

    for i in range(0, M.shape[0]):
        M_out[i*f:(i+1)*f, :] = M[i, :]



    return M_out





####define action space

def create_fun_vector(cnt_mat, idx_mat, type_mat, unique_type=np.asarray([12, 9, 15, 18])):

    #cnt_mat -> matrix of Cnt array
    #fun_mat has the size of len(cnt_mat) and unique_type

    fun_mat = np.zeros((len(cnt_mat), len(unique_type)))
    type_count = 0

    for idx in idx_mat:

        #first let's deal with fun_type
        type_choose = type_mat[type_count]
        idx_type = np.where(unique_type==type_choose)
        fun_mat[idx, idx_type] = 1

        type_count += 1
    #fun type is a list of types





    return fun_mat




def create_bounds(bounds=np.asarray([1.0, 2.0, 8.0]), intervals=np.asarray([0.01, 0.01, 0.01])):

    # parameters to compute the RDF
    r1 = np.linspace(0, bounds[0], bounds[0] / intervals[0], endpoint=False)
    r2 = np.linspace(bounds[0], bounds[1], (bounds[1] - bounds[0]) / intervals[1], endpoint=False)
    r3 = np.linspace(bounds[1], (bounds[2]), (bounds[2] - bounds[1]) / intervals[2] + 1)

    r = np.concatenate((r1, r2, r3), axis=0)

    return r


