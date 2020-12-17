import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance
import mpi4py
import pandas as pd
import re
import glob
import math


from enum import Enum
from scipy import optimize

##-----create an enum

class resin_sequence(Enum):
    # define
    O_C = 1  # O-C bod
    pN_B = 2 #primary nitrogen-benzene bond
    sN_B = 3 #secondary nitrogen-benzene bond
    eC_sN = 4 #epoxide carbon -reacted carbon bond
    eC_tN = 5 #epoxide secondary amine

    resin_key = 6
    resin_rev = 7 #reverse of resin key

    N_N = 8

    C_N = 11
    C_rC = 12
    #C_tN = 12
    B_N = 13
    #B_tN = 14

    sN_eC = 15
    tN_eC = 16



    # BFDGE_OC = [18, 12] #CNT-O-epoxide
    # BFDGE_CsN = [12, 14, 15]#CNT-
    # BFDGE_CtN = [12, 14, 16]

    # BFDGE_sNpN = [15, 4, 4, 4, 9]

class BFDGE():

    def __init__(self):

        self.R = [12, 2, 2, 3, 4, 4, 4, 4,  2, 4, 4, 4, 4, 3, 2, 2, 12] #end to end sequence of atoms

        #notes:
        #overlap between end atom at the end and the atom at the beginning (because sequece[1:])

        ##define all pathways between twoo nodes, i.e. epoxide C and benzene atom
        R1 = self.R #functionalized resin molecule not bonded to any
        R2 = self.R[:-1] + [14, 15, 4]  #functionalized epoxy to primary nitrogen
        R3 = self.R[:-1] + [14, 16, 4]

        #assume that the connecting benezene is the connecting node
        R4 = [15, 14] + self.R[1:]
        R5 = [15, 14] + self.R[1:-1] + [14, 15, 4]
        R6 = [15, 14] + self.R[1:-1] + [14, 16, 4]
        #

        R7 = [16, 14] + self.R[1:]
        R8 = [16, 14] + self.R[1:-1] + [14, 15, 4]
        R9 = [16, 14] + self.R[1:-1] + [14, 16, 4]
        #tertiary connection

        #The "S" resins are the R resins in reverse
        R10 = self.R[1:]
        R11 = self.R[1:-1] + [14, 15]
        R12 = self.R[1:-1] + [14, 16]

        self.sequences = {}
        self.sequences[resin_sequence.O_C] = [[18, 2]]
        self.sequences[resin_sequence.pN_B] = [[9, 4]]
        self.sequences[resin_sequence.sN_B] = [[15, 4]]


        self.sequences[resin_sequence.eC_sN] = [(self.R[:-1]) + [14, 15], (self.R[1:-1]) + [14, 15]]
        self.sequences[resin_sequence.eC_tN] = [(self.R[:-1]) + [14, 16], (self.R[1:-1]) + [14, 15]]
        self.sequences[resin_sequence.resin_key] = [R1, R2, R3, R4, R5, R6, R7, R8, R9]

        ###second degree chains
        self.sequences[resin_sequence.C_N] = [[12, 14, 15], [12, 14, 16]]
        self.sequences[resin_sequence.C_rC] = [R10, R11, R12]
        #self.sequences[resin_sequence.C_tN] = [[12, 14, 16]]
        self.sequences[resin_sequence.B_N] = [[4, 4, 4, 9], [4, 4, 4, 15], [4, 4, 4, 16]]
        self.sequences[resin_sequence.N_N] = [[9, 4, 4, 4, 9], [15, 4, 4, 4, 9], [16, 4, 4, 4, 9],
                                              [9, 4, 4, 4, 15], [15, 4, 4, 4, 15], [16, 4, 4, 4, 15],
                                              [9, 4, 4, 4, 16], [15, 4, 4, 4, 16], [16, 4, 4, 4, 16]]
        #self.sequences[resin_sequence.B_N] = [[4, 4, 4, 9], [4, 4, 4, 15], [4, 4, 4, 16], [4, 4, 4, 4, 9], [4, 4, 4, 4, 15], [4, 4, 4, 4, 16]]
        #self.sequences[resin_sequence.B_tN] = [[4, 4, 4, 16]]

        #define the reverse N-epoxy chain
        self.sequences[resin_sequence.sN_eC] = ((self.R[:-1]) + [14, 15])[::-1]
        self.sequences[resin_sequence.tN_eC] = ((self.R[:-1]) + [14, 15])[::-1]







class cluster_files():
    def __init__(self, list_of_files):
        self.list = list_of_files
        self.rho_list = []


        for i in range(0, len(list_of_files)):
            file_temp = sorted(glob.glob(list_of_files[i]), key=key_func)

            for j in range(0, len(file_temp)):

                rho_cl = self.read_files(file_temp[j])
                self.rho_list.append(rho_cl)


        print (self.rho_list)





    def read_files(self, filename, cnt_type=22):

        df = pd.read_table(filename, delim_whitespace=True, header=None, skiprows=3)

        type = ((df.iloc[:, 3].str.split("_", expand=True)).as_matrix()[:, -1]).astype(int)
        particle_id = (df.iloc[:, 4]).as_matrix()
        cluster_id = df.iloc[:, -1].as_matrix()

        #index out CNT type
        idx_cnt= np.where(type==cnt_type)[0]
        unique_cnt_cluster = np.unique(cluster_id[idx_cnt]) #find the unique clusters in

        if len(unique_cnt_cluster) == 1:

            cluster_choose = unique_cnt_cluster[0]
            particle_idx = np.where(cluster_id==cluster_choose)[0]
            particle_pick = particle_id[particle_idx]
            rho_cl = float(len(particle_pick))/float(len(particle_id))

        else:

            rho_cl = 0.0




        return rho_cl




#---------
class MD_model():

    def __init__(self, MD_file, f_value, chain_file, classname=BFDGE, extract_feature_bool=True):

        #extract_feature_bool -> indicates if we want to extract feautures
        self.feature_bool = extract_feature_bool
        ##MD file -> MD file sin question
        #f_value -> force value
        self.classname = classname
        self.file_name = MD_file
        self.f_value = f_value
        self.chain_file = chain_file
        self.gamma_bool = True
        self.max_degree = 40
        self.num_features = 10
        self.max_path = 40
        #find center:

        #this class identifies each MD model, and for each model, identify all the clusters

        #seq_to_pick = self.load_resin(self, class_name=BFDGE, seq_type=O_C)

        #this code to get the block bounds

        self.box_bounds, self.L = self.get_bounds()

        #print "Box Lengths: ", self.L

        d_thr, self.id_fun, id_cnt, self.fun_type, self.df_xyz, self.df_bond = self.detect_fun_atoms()

        #print "troubleshoot: ", d_thr, self.id_fun, id_cnt, self.fun_type
        #print "id_cnt: ", id_cnt

        #get the center of the CNT circle
        self.center, self.R = get_circle_center(df=self.df_xyz)
        #print "self.center: ", self.center
        #print "self.id_fun: ", self.id_fun



        if len(self.id_fun) > 0:
            if self.id_fun[0] != -1:
                global_var = global_features(fun_id=self.id_fun, cnt_id=id_cnt, center=self.center, df=self.df_xyz)

        #find_chain_particles(sequence=[12, 2, 2, 3, 4, 4, 4, 4,  2, 4, 4, 4, 4, 3, 2, 2, 12], init_id=3770, df_xyz=self.df_xyz, df_bond=self.df_bond)

        ###this block loads the chains
            if self.id_fun[0] != -1:
               self.execute_chain()
            else:
               self.Z_1 = np.zeros((self.max_degree + 1, self.num_features))
               self.Z_2 = np.zeros((self.max_degree + 1, self.num_features))
               self.Z = np.zeros((self.max_path, self.max_degree + 1, self.num_features))

               self.atomic_length = 0

        else:

            self.Z_1 = np.zeros((self.max_degree + 1, self.num_features))
            self.Z_2 = np.zeros((self.max_degree + 1, self.num_features))
            self.Z = np.zeros((self.max_path, self.max_degree + 1, self.num_features))
            self.atomic_length = 0

        #if ('force_wpattern' in kwargs):

           #force_wpattern = kwargs['force_wpattern']
           #self.MD_file_list, self.f_file_list = read_MD_files(list_of_files=MD_wpattern, forcefile=force_wpattern)


        #else:

            #self.MD_file_list, _ = read_MD_files(list_of_files=MD_wpattern)



    def execute_chain(self, max_degree=40, fun_max=5):

        #this function takes in the inital atom id as input and generates a staggered list containing the atom ids
        #this should be done for each new funtionalized atom
        #max_degree -> maximum degree of removal from the cnt atom
        #fun_max = maximum number of
        #initialize lists

        next_atoms = []
        next_atom_types = []

        stag_list = []

        #start initializing a hierarchy of lists
        #level_0 initializaation
        X0_list = [] #high level node list
        A0_list = [] #high level Adjacency matrix list

        self.fun_node = self.id_fun

        #total dimensions -> [C, N, O]
        #Adjacency matrix
        if self.feature_bool:
            self.fun_mat = np.zeros((fun_max, 3))

        #initialize
        thread_list = []
        thread_node = []
        adjacency_list = []
        bool_list = []



        for idx, fun_id in enumerate(self.id_fun):

            #D = benzene_density_count(df=self.df_xyz, node_id=self.id_fun[idx])
            print( idx, self.id_fun[idx])

            self.set_fun_feature(idx=idx)

            print( "---new fun----------------")
            #level 1 list initialization
            X1_list = []  # high level node list
            A1_list = []  # high level Adjacency matrix list

            #thread_list = []

            node_list = []
            node_type_list = []
            node_seq = []
            node_seq_type = []

            for n in range(0, max_degree):

                print( "-------------------------------------New degree of removal---------------------")
                # level 2 list initialization

                X2_list = []  # high level node list
                A2_list = []  # high level Adjacency matrix list

                if n == 0:

                    init_id = [fun_id]
                    next_atoms = []
                    next_atom_types = []
                else:
                    init_id = next_atoms
                    init_types = next_atom_types

                    #reset next_atoms to get rid of old next_atoms
                    next_atoms = []
                    next_atom_types = []

                print( "init_id; ", init_id)

                for i in range(0, len(init_id)):

                    print( "!!!!!!!!!New atom id !!!!!!!!!!!!!!")
                    #print init_id
                    print( "init id: ", init_id[i])

                    atom_seq_1 = atom_seq_2 = type_seq_1 = type_seq_2 = -1


                    # first load the sequece that you'd like to investigate
                    # need to define
                    init_pick = init_id[i] #select which among the init id

                    if n==0:
                        fun_bool = True
                        fun_val = self.fun_type[idx]   ##the type of atom in question
                        end_type = -1
                    else:
                        fun_bool = False
                        fun_val =-1
                        end_type = init_types[i]



                    #now load the resin molecule and traverse through the chain
                    #print "n, end_type: ", n, end_type
                    #print "fun bool, fun_val: ", fun_bool, fun_val

                    orig_atom, epoxy_bool = trace_origin(init_atom=init_pick, node_list=node_list,
                                                        node_type_list=node_seq_type)

                    #print "orig atom: ", orig_atom
                    #print "epoxy bool: ", epoxy_bool

                    seq_1, seq_2, num_pathway = self.load_resin_molecule(fun_bool=fun_bool,
                                                                                    fun_type=fun_val, end_type=end_type, epoxy_bool=epoxy_bool)

                    #print "Seq 1: ", seq_1
                    #print "Seq 2: ", seq_2, fun_bool, fun_val
                    #print seq_1, seq_2, num_pathway
                    if seq_1 == -1:
                        continue


                    atom_seq_1, type_seq_1, other_id, chain_length, periodic_bool = self.track_atoms(sequence_atoms=seq_1, init_atom=init_pick)

                    #print "atom seq 1 (before): ", atom_seq_1

                    if not atom_seq_1:
                        atom_seq_1, type_seq_1, _, chain_length, periodic_bool = self.track_atoms(sequence_atoms=seq_1, init_atom=init_pick,
                                                                     other_id=other_id)

                    #print "atom seq 1 (after): ", atom_seq_1

                    if atom_seq_1 == -1 or len(atom_seq_1) < 1:

                        break

                    end_type_old = end_type
                    end_atom, end_type = atom_seq_1[-1], type_seq_1[-1]

                    ###add this line to keep track of paths
                    thread_list, og_list, thread_node, og_node, adjacency_list, og_adjacency, bool_thread, og_bool = thread_check(new_T=atom_seq_1, T_list=thread_list,
                                                                                                   node_list=thread_node, adjacency_list=adjacency_list, chain_length=chain_length, bool_thread=bool_list,
                                                                                                                                  bool_array=periodic_bool)


                    #thread_list, og_list = thread_check02(new_T=atom_seq_1, T_list=thread_list)

                    #print "thread list 1: ", thread_list
                    print( "thread node 1: ", thread_node)
                    print( "chain length 1: ", chain_length)
                    print( "peridic bool: ", periodic_bool)
                    #print "adjacency list 1: ", adjacency_list
                    #print "og adjacency list 1: ", og_adjacency
                    #print "bool thread 1 ", bool_list
                    #print "og bool 1", og_bool

                    #og_list -> the orignal lst before concatenation

                    #appending for X2_list
                    node_list.append([init_pick, end_atom])
                    node_type_list.append([end_type_old, end_type])
                    node_seq.append(atom_seq_1)
                    node_seq_type.append(type_seq_1)

                    if len(node_list) and node_list not in X2_list> 0:
                        X2_list.append(node_list)
                    ##check if the list is empty. if it is empty, it indicates that the

                    #print "next atoms 1: ", next_atoms


                    if len(next_atoms) > 0:
                        next_atoms = next_atoms + [end_atom]
                        next_atom_types = next_atom_types + [next_atom_types]

                    else:
                        next_atoms = [end_atom]
                        next_atom_types = [end_type]

                    #print "nxt atoms 2: ", next_atoms
                    #print "next atom types 2: ", next_atom_types

                    print( "atom seq 1: ", atom_seq_1)

                    if num_pathway == 2:
                        #if we're branching off due to a tertiary amine, let's fetch off the taken_C
                        #taken_C = atom_seq_1[1]
                        type_check = type_seq_1[1]



                        if type_check == 14:
                            taken_C = atom_seq_1[1]
                            atom_seq_2, type_seq_2, _, chain_length, periodic_bool = self.track_atoms(sequence_atoms=seq_2,
                                                                                       init_atom=init_pick, taken_C=taken_C)
                        else:
                            taken_C = find_taken_C(init_atom=init_pick, node_seq_list=node_seq, type_seq_1=type_seq_1)

                            #print "taken C: ", taken_C

                            if taken_C == -1:
                                atom_seq_2, type_seq_2, _, chain_length, periodic_bool = self.track_atoms(sequence_atoms=seq_2,
                                                                                      init_atom=init_pick)
                            else:
                                atom_seq_2, type_seq_2, _, chain_length, periodic_bool = self.track_atoms(sequence_atoms=seq_2,
                                                                                           init_atom=init_pick, taken_C=taken_C)

                        #print "fun atom: ", fun_id
                        thread_list, _, thread_node, _, adjacency_list, og_adjacency, bool_thread, og_bool = thread_check(new_T=atom_seq_2, T_list=thread_list, og_list=og_list,
                                                      create_thread=True, n_value=n, fun_atom=fun_id, node_list=thread_node,
                                                            adjacency_list=adjacency_list, chain_length=chain_length,
                                                                            og_adjacency=og_adjacency, og_node=og_node, bool_thread=bool_list, bool_array=periodic_bool, og_bool=og_bool)


                        #thread_list, _, _ = thread_check(new_T=atom_seq_2, T_list=thread_list, og_list=og_list)


                        end_atom_2, end_type_2 = atom_seq_2[-1], type_seq_2[-1]
                        next_atoms = next_atoms + [end_atom_2]
                        next_atom_types = next_atom_types + [end_type_2]
                        print( "atom_Seq_2: ", atom_seq_2)


                        #print "thread list 2: ", thread_list
                        print( "thread node 2: ", thread_node)
                        print( "chain length 2: ", chain_length)
                        #print "adjacency list 2: ", adjacency_list
                        #print "og adjacency list 2: ", og_adjacency
                        #print "bool thread 2 ", bool_list
                        #print "og bool 2", og_bool

                        node_list.append([init_pick, end_atom_2])
                        node_type_list.append([end_type_old, end_type_2])
                        node_seq.append(atom_seq_2)
                        node_seq_type.append(type_seq_2)

                        #if len(node_list_2) and node_list_2 not in X2_list> 0:
                            #X2_list.append(node_list_2)

                    #print "nxt atoms 3: ", next_atoms
                    #print "next atom types 3: ", next_atom_types
                    #print "node seq: ", node_seq
                    #print "node seq type: ", node_seq_type


                #load_resin_molecule(self, fun_bool, fun_type, end_type)
                    stag_list.append(atom_seq_1)

                    ###append to all_list:
                    try:
                        all_list = all_list + atom_seq_1
                        all_type = all_type + type_seq_1
                    except:
                        all_list = atom_seq_1
                        all_type = type_seq_1

                    if num_pathway == 2:
                        all_list = all_list + atom_seq_2
                        all_type = all_type + type_seq_2

                    #print "atom seq 1: ", atom_seq_1
                    #print "type seq 1: ", type_seq_1
                    #print "atom seq 2: ", atom_seq_2
                    #print "type seq 2: ", type_seq_2
                    #print "next atoms: ", next_atoms
                    #print "next atom types: ", next_atom_types
                    #print "fun id: ", fun_id
                    #print "node list: ", node_list
                    #print "node type list: ", node_type_list
                    #print "node seq: ", node_seq
                    #print "node seq type: ", node_seq_type

                    #print "node list: ", thread_node
                    #print "adjacency list: ", adjacency_list


                if len(X2_list) > 0: X1_list.append(X2_list)

            X0_list.append(X1_list)

        #print "next_atoms: ", next_atoms
        #print "next_atom_types: ", next_atom_types'
        #print "thread list f: ", thread_list
        idx_c = 0
        #print "thread node f: ", thread_node
        #print "thread node list chosen: ", thread_list[idx_c]
        #print "adjacency list f: ", adjacency_list
        #print "bool_list: ", bool_list
        print( "all_list: ", np.unique(all_list).tolist())

        command_out = 'ParticleType == 22'

        for j in np.unique(all_list).tolist():

            command_out = command_out + ' || ' + ' ParticleIdentifier == ' + str(j)

        print( command_out)
        #print "all type: ", all_type
        print( "all length ", len(np.unique(all_list).tolist()))
        #print "fun id: ", self.id_fun
        #print "X0 list: ", X0_list

        ##troubleshoot here
        #x_1 = get_type_mat(thread_node_list=thread_node[3], df_xyz=self.df_xyz)
        #print bool_list[-4]
        x_2 = feature_periodic(thread_periodic=bool_list[idx_c])
        #x_2p = feature_periodic(x_2)
        #x_3 = get_CNT_distance(thread_node_list=thread_node[3], df_xyz=self.df_xyz)
        #print "x_1: ", x_1
        #print "x_2: ", x_2
        #print "x_2 abs: ", np.absolute(bool_list[-4])

        #print "x_3: ", x_3

        X = extract_graph_features(node_list=thread_node, df_xyz=self.df_xyz, bool_list=bool_list, max_degree=max_degree+1)
        A = get_adjacency_matrix(a_list=adjacency_list, max_degree=max_degree+1)
        D = get_diag_mat(A)
        Z = get_Z_mat(D=D, A=A, X=X)



        #print "X: ", X[idx_c, 0:15, :]
        #print "A: ", A[idx_c, 0:15, :]
        #print "D: ", D[idx_c, 0:15, :]
        #print "Z: ", Z[idx_c, 0:15, :]
        #print "Z: ", Z[idx_c+1, 0:15, :]

        Z_1, Z_2, Z_u = get_unique_Z(Z=Z)
        print( "Z1: ", Z_1.shape)
        print( "Z2: ", Z_2.shape)
        print( "Z_u: ", Z_u.shape)

        self.Z_1 = Z_1
        self.Z_2 = Z_2
        self.Z = Z
        self.atomic_length = len(np.unique(all_list).tolist())

        return None




    def detect_fun_atoms(self, type_choose=22, thresh=1.8, d_node=10.0):

        df_xyz, df_bond = self.read_bonds() #df_xyz -> coordinate system, df_bond -> topology of bonds
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

        D, _, _  = compute_distance(df_1.as_matrix(), df_2.as_matrix())


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

        d_node = D[D < d_node]

        #print id_fun

        #this block to ensure that the extracted fun atoms is unique
        id_fun, unique_idx = np.unique(id_fun, return_index=True)

        print( id_fun)

        if len(id_fun) > 1:
            type_choose = type_choose[unique_idx]
            id_cnt = id_cnt[unique_idx]


        #correct based on simulations

        if id_fun[0]!=-1:
            id_fun, id_cnt, type_choose = check_fun_atoms(fun_in=id_fun, cnt_in=id_cnt, df_xyz=df_xyz, type_in=type_choose)


        return d_thr, id_fun, id_cnt, type_choose, df_xyz, df_bond


    def read_bonds(self):


        with open(self.chain_file) as f:
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

        print( "troubleshoot", skip_count, vel_count, bond_count)

        df_xyz = pd.read_table(self.chain_file, delim_whitespace=True, header=None, skiprows=skip_count,  nrows=(vel_count - skip_count))
        df_bond = pd.read_table(self.chain_file, delim_whitespace=True, header=None, skiprows=bond_count)



        return df_xyz, df_bond


    def get_bounds(self):

        #this function gets the box bounds
        with open(self.chain_file, 'r') as f:
            df_bounds = pd.read_table(self.chain_file, delim_whitespace=True, header=None, skiprows=5, nrows=3)

            # bounds as array
            bound_array = df_bounds.as_matrix()
            bound_array = bound_array[:, :2]

        L_x = bound_array[0, 1] - bound_array[0, 0]
        L_y = bound_array[1, 1] - bound_array[1, 0]
        L_z = bound_array[2, 1] - bound_array[2, 0]

        L = np.asarray([L_x, L_y, L_z])


        return bound_array, L



    def load_resin_molecule(self, fun_bool, fun_type, end_type, **kwargs):

        ##extract out kwargs
        if 'epoxy_bool' in kwargs:
            epoxy_bool = kwargs['epoxy_bool']

        if 'p_amine_bool' in kwargs:
            amine_bool = kwargs['p_amine_bool']

        #first unpack opt_list
        resin = self.classname()

        #fun_bool is a boolean to indicate whether the node is a functionalized atom
        #fun type -> [0, 1, 2, 3] depending upon the functioanlized atom
        #if fun_bool is False and the node is internal, then we need to find the end atom to start the next chain

        num_pathway = 1 ##num_pathway -> the pathways that a can follow a given end-point
        seq_type = -1
        seq_type_2 = -1

        if fun_bool == True:

            num_pathway = 1

            if fun_type == 12:
                seq_type = resin_sequence.resin_key
            elif fun_type == 9:
                seq_type = resin_sequence.pN_B
            elif fun_type == 15:
                num_pathway = 2
                seq_type = resin_sequence.sN_B
                seq_type_2 = resin_sequence.resin_key
            elif fun_type == 18:
                seq_type = resin_sequence.O_C
            else:
                seq_type = -1

        else:

            #based on the end type we need to select a chain molecule
            if end_type == 4: #end with a benzene molecule
                num_pathway = 1
                seq_type = resin_sequence.B_N   #link benzene to nitrogen
            elif end_type == 15:
                num_pathway = 1

                if epoxy_bool == False:
                ##check if the starting atom is benzene or not
                    seq_type = resin_sequence.resin_key
                else:
                    seq_type = resin_sequence.N_N

            elif end_type == 16:
                num_pathway = 2

                if epoxy_bool == False:
                    seq_type = resin_sequence.resin_key
                    seq_type_2 = resin_sequence.resin_key
                else:
                    seq_type = resin_sequence.N_N
                    seq_type_2 = resin_sequence.resin_key

            elif end_type == 2: #in case of two
                num_pathway =2
                seq_type = resin_sequence.C_N
                seq_type_2 = resin_sequence.C_rC
            else:
                seq_type = -1
                seq_type_2 = -1


        #print "seq_type: ", seq_type
        #print "seq_type 2: ", seq_type_2

        if seq_type != -1:
            seq_to_pick = resin.sequences[seq_type]
        else:
            seq_to_pick = -1

        if seq_type_2 != -1:

            seq_to_pick2 = resin.sequences[seq_type_2]
        else:
            seq_to_pick2 = -1


        ##this function takes in the class name and outputs a sequence
        #seq_type -> O_C,



        return seq_to_pick, seq_to_pick2, num_pathway



    def track_atoms(self, init_atom, sequence_atoms, **kwargs):



        ###define a function to track along the resin molecule
        #track the sequence of atoms

        seq_out = type_out = []


        #track if there are multiple options:

        for j in range(len(sequence_atoms)):

            seq_to_track = sequence_atoms[j]

            if ('other_id' in kwargs):
                other_id_in = kwargs['other_id']
                seq_tmp, type_tmp, other_id, chain_length, periodic_bool, bool_check = find_chain_particles(sequence=seq_to_track, init_id=init_atom,
                                                     df_xyz=self.df_xyz, df_bond=self.df_bond, other_id=other_id_in, L=self.L) #change this
            elif ('taken_C' in kwargs):
                taken_C = kwargs['taken_C']
                seq_tmp, type_tmp, other_id, chain_length, periodic_bool, bool_check = find_chain_particles(sequence=seq_to_track,
                                                                                 init_id=init_atom,
                                                                                 df_xyz=self.df_xyz,
                                                                                 df_bond=self.df_bond, taken_C=taken_C, L=self.L)  # change this

            else:
                seq_tmp, type_tmp, other_id, chain_length, periodic_bool, bool_check = find_chain_particles(sequence=seq_to_track, init_id=init_atom,
                                                                   df_xyz=self.df_xyz,
                                                                   df_bond=self.df_bond, L= self.L)  # change this

            #if the final sequence ends with -1, then we know we have the right sequence

            ##this block of code finds the bol_array




            if seq_tmp[-1] != -1:

                seq_out = seq_tmp
                type_out = type_tmp

                break

        #def traverse_chain(fun_array, resin_type==1):

        #print "chain length: ", chain_length
        #print "other id: ", other_id
        print( "periodic bool tracker: ", periodic_bool)
        print( "bool check: ", bool_check)
        print( "seq_out: ", seq_out)

        #extracting the bool_array
        if periodic_bool and len(seq_out)>0:
            bool_array, x_1, x_2, x_2p = periodic_fun(atom_seq=seq_out, L_array=self.L, df_xyz=self.df_xyz, bool_check=bool_check)
            x_1 = x_1.reshape(1, 3)
            x_2p = x_2.reshape(1, 3)

            #print "x_1: ", x_1
            #print "x_2: ", x_2
            #print "x_2p: ", x_2p
            d_1, _, _ = compute_distance(x_1, x_2p)
            d_1 = d_1.flatten()[-1]
        elif len(seq_out) > 0:
            bool_array = np.zeros((3,))
            d_1 = compute_gamma_d(atom_seq=seq_out, df_xyz=self.df_xyz)
        else:
            d_1 = 0
            bool_array = np.zeros((3, ))

        #if self.gamma_bool and len(seq_out)>0:
            #gamma = compute_gamma(atom_seq=seq_out, df_xyz=self.df_xyz, L=self.L, chain_length=chain_length)
        #else:
            #gamma = 0

        #print "d_1: ", d_1
        #print "chain length: ", chain_length
        #print "gamma: ", d_1/chain_length
        if self.gamma_bool:
            if chain_length != 0:
                c1 = d_1/chain_length
                chain_length = c1

            else:
                chain_length = 0

        if not hasattr(chain_length, "__len__"):
            fix_arr = chain_length*np.ones((1, 1))
            chain_length = fix_arr





        #print "d_1: ", d_1
        #print "chain length: ", chain_length
        ##print "gamma: ", d_1/chain_length


        bool_array = bool_array[None, :]


        return seq_out, type_out, other_id, chain_length, bool_array



    def set_fun_feature(self, idx):

        if self.feature_bool:

            if self.fun_type[idx] == 12:
                self.fun_mat[idx, 0] = 1
            elif self.fun_type[idx] == 9 or self.fun_type[idx] == 15:
                self.fun_mat[idx, 1] = 1
            elif self.fun_type[idx] == 18:
                self.fun_mat[idx, 2] = 1
            else:
                self.fun_mat[idx, :] = 0




        return None


    def fun_adjacency(self, max_fun=5, theta_bool=True, z_bool=True):

        num_fun = len(self.id_fun) #total number of functionalized atoms

        #need the coordinates of all the
        if z_bool == True:
            A_z = np.zeros((5, 5))




        return None



       ##------

#globalfiles

def read_MD_files(list_of_files, **kwargs):

    #list of files -> [XXXX.00*, YYY.00*]
    #need to convert wildcard pattern to a single list

    targets = np.empty((0,)) #targets only needed if forcefile is in **kwargs
    total_files = []
    chain_files = []

    for i in range(0, len(list_of_files)):
        # print list_of_files[i]
        file_temp = sorted(glob.glob(list_of_files[i]), key=key_func)
        total_files = total_files + file_temp


        ###if the forcefile is in kwargs then tie the pullout force with the given model

        if ('forcefile' in kwargs):
            ##forcefile has the same
            forcefile = kwargs['forcefile']


            raw_y = get_raw_targets(forcefile[i])
            target_conf = configure_targets(file_list=file_temp, targets=raw_y)
            targets = np.append(targets, target_conf)

        if ('chainfile' in kwargs):

            chainfile = kwargs['chainfile']

            file_new = sorted(glob.glob(chainfile[i]), key=key_func)
            chain_files = chain_files + file_new


    return total_files, targets, chain_files

##-----utility fun--------


def key_func(x):
    nondigits= re.compile("\D")

    return int(nondigits.sub("", x))



#def split
def get_raw_targets(filename):

    df = pd.read_csv(filename, sep=',', header=None)
    y_out = df.as_matrix()[:, 1]

    return y_out


###--------------
def configure_targets(file_list, targets):

    #need to tally the file_list with the targets
    target_final = np.zeros((len(file_list), ))
    id_f = np.zeros_like(target_final)

    #print "file list: ", len(file_list)
    for i in range(0, len(file_list)):

        file_num = (file_list[i]).rsplit('.')[-1]
        id_f[i] = int(file_num)


    #print "idf: ", len(id_f), len(targets)
    id_f = id_f.astype(int)
    target_final = targets[id_f]

#    print "targets: ", target_final[32]

    return target_final


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



def find_chain_particles(sequence, init_id, df_xyz, df_bond, symmetry_bool = False, **kwargs):

    if 'L' in kwargs:
        L = kwargs['L']
    else:
        L = np.zeros((3, ))

    id_mat = df_xyz.loc[:, ['id']].as_matrix()
    type_mat = df_xyz.loc[:, ['type']].as_matrix()
    bond_mat = df_bond.loc[:, ['particle_1', 'particle_2'] ].as_matrix()

    #This function takes in two argiments: sequence -> list of types for the sequence
    #and init_id -> the particle id of the first particle
    #Symmetry_bool -> indicates wether

    chain_id = [init_id]
    chain_type = [sequence[0]]
    count = 1
    other_id = -1
    #Init troubleshoot
    #idx_p = np.where(id_mat == 4131)
    #type_p = type_mat[idx_p[0]].flatten()

    #print "type p: ", type_p
    chain_length = 0

    periodic_bool = False
    bool_check = np.zeros((1, 3))
    #print "troubleshoot: ", init_id


    for type_choose in sequence[1:]:

        #type -> the type of atom
        #find where the atom sequence matches
        idx_all = np.where(bond_mat == init_id) #find all bonding where init_id is involved
        #bond_choose
        #extract out the rows
        bond_select =  bond_mat[idx_all[0]]
        other_particle = np.sort(bond_select[bond_select!=init_id]) #find the other bonded particle

        # print ""
        #print "bond_select: ", bond_select
        #print "other_particle: ", other_particle
        #print "id_mat: ", id_mat

        #Indexing out the "other particle"
        idx_tmp = index_mat(id_mat.flatten(), other_particle.flatten())
        #idx_tmp = np.where(id_mat == other_particle)

        #print "idx_temp troubleshoot: ", idx_tmp
        #type_bond = type_mat[(idx_tmp[0])].flatten() ####this is the confusing part
        type_bond = type_mat[idx_tmp].flatten()

        #print "type_bond: ", type_bond
        #print "type bond 2: ", type_bond_second
        idx_tmp = np.where(type_bond == type_choose)



        #print "idx_temp: ", idx_tmp
        #print "count: ", count
        ###this block of code ensures that when tracking benezene atoms

        if len(idx_tmp[0]) > 1 and count < len(sequence) and type_choose == 4:
            #print "benzene block: "
            particle_tmp = other_particle[idx_tmp]
            #print "particle tmp: ", particle_tmp
            idx_to_del =  np.where(np.isin(particle_tmp, np.asarray(chain_id)))
            particle_tmp = np.delete(particle_tmp, idx_to_del)
            #print "particle tmp: ", particle_tmp
            #new_atom_id = particle_tmp[0] ### cehck this for benzene

            ##testing out the code for benzene
            #if
            if symmetry_bool == False and len(particle_tmp) > 1:
                new_atom_id, other_id = detect_particle_benzene(df_xyz=df_xyz, particle_1=particle_tmp[0],
                                                      particle_2=particle_tmp[1])

                ###conisdering the case where we know that our previous work hasn;t worked and we need to try
                if ('other_id' in kwargs):
                    other_id_in = kwargs['other_id']
                    #print other_id_in, other_id

                    if other_id == other_id_in:
                        #print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                        new_atom_id = other_id_in
                        #print "other block: "
                        #print other_id_in
                        #print new_atom_id
            else:
                new_atom_id = particle_tmp[0]




            #once the previous a

        elif len(idx_tmp[0]) > 0:

            new_atom_id = other_particle[idx_tmp][0]

            if type_choose == 14 and 'taken_C' in kwargs:
                taken_C = kwargs['taken_C']
                #print "other_particle: ", other_particle[idx_tmp]
                #print "taken_C: ", taken_C
                chosen_particle = other_particle[idx_tmp]
                new_atom_id = chosen_particle[chosen_particle != taken_C][0]

                #print "new atom id: ", new_atom_id

        else:

            new_atom_id = -1


        #renew init_id
        chain_id.append(new_atom_id)
        chain_type.append(type_choose)

        if new_atom_id == -1:
            break
        else:
            ###find the bond distance
            #print "computing bond distance: "
            df_1 = df_xyz.loc[df_xyz['id']==init_id]
            f_1 = df_1.loc[:, ['x', 'y', 'z']].as_matrix()

            df_2 = df_xyz.loc[df_xyz['id'] == new_atom_id]
            f_2 = df_2.loc[:, ['x', 'y', 'z']].as_matrix()

            d_1, _, _ = compute_distance(f_1, f_2)

            if d_1 > 2.0:
                d_1, bool_check = get_adjusted_d(p_1=init_id, p_2 = new_atom_id, df_xyz=df_xyz, L_array=L, bool_check=bool_check)
                periodic_bool = not periodic_bool

            chain_length += d_1



            #print "d_1: ", d_1





        init_id = new_atom_id
        count += 1
        #indx out from the main id matrix where new_atom id
        idx_tmp2 = np.where(id_mat == new_atom_id)

        #print "check: ", type_mat[idx_tmp2]

        #print "idx_tmp: ", idx_tmp
        #print "new atom id: ", new_atom_id
        #print "chian: : ", chain_id
        #print "chain type: ", chain_type
        #print"type bond ", type_bond
        #select particles
            #for row in bond_select.shape[0]:


        #print "idx_all: "
        ##print idx_all




    #print "chain_id: ", chain_id



    return chain_id, chain_type, other_id, chain_length, periodic_bool, bool_check




def index_mat(idx_mat, particle_to_detect):

    idx_out = np.zeros(len(particle_to_detect), )

    for i,atom in enumerate(particle_to_detect):

        idx_out[i] = np.where(idx_mat == atom)[0]

    return idx_out.astype(int)


def detect_particle_benzene(df_xyz, particle_1, particle_2, select_list=[9, 14, 15, 16]):
    # the reason for this function is that we would like to detet the benzene carbon at the center (Radue thesis)

    # first select_idsout
    id_mat = df_xyz.loc[:, ['id']].as_matrix().flatten()  # id mat
    idx_1 = np.where(id_mat == particle_1)[0]
    idx_2 = np.where(id_mat == particle_2)[0]

    xyz_mat = df_xyz.loc[:, ['x', 'y', 'z']].as_matrix()
    xyz_1 = xyz_mat[idx_1, :]  # coordinates for particle 1
    xyz_2 = xyz_mat[idx_2, :]  # coordinates for particle 2

    df_fun = df_xyz[df_xyz['type'].isin(select_list)]

    xyz_fun = df_fun.loc[:, ['x', 'y', 'z']].as_matrix()



    # now we have the coordinates coodinates for both p1 and p2, simply find the benzene that is closest to both
    D_1, _, _ = compute_distance(xyz_1, xyz_fun)
    D_2, _, _ = compute_distance(xyz_2, xyz_fun)



    d1 = np.mean(np.sort(D_1.flatten())[0:2])
    d2 = np.mean(np.sort(D_2.flatten())[0:2])


    # in this block we select out particle out

    if d2 < d1:
        particle_out = particle_2
        particle_none = particle_1
    else:
        particle_out = particle_1
        particle_none = particle_2

    return particle_out, particle_none



def benzene_density_count(df, node_id, benzene_type=4):

    #df -> dataframe input

    df_n = df.loc[df.loc['id'] == node_id]
    node_mat = df_n.loc[:, ['x', 'y', 'z']]

    df_benzene = df.loc[df.loc['type']==benzene_type]
    benzene_mat = df_benzene.loc[:, ['x', 'y', 'z']]

    D = compute_distance(node_mat, benzene_mat)




    return D




def check_fun_atoms(fun_in, cnt_in, df_xyz, type_in, D_thr = 2.0):

    ##this function checks if a bond exists between the functionalized atom and thesecond atom
    fun_out = []
    type_out = []
    cnt_out = []

    for idx, fun in enumerate(fun_in):

        #extract fun coordinates
        df_fun = df_xyz.loc[df_xyz['id']==fun]
        fun_xyz = df_fun.loc[:, ['x', 'y', 'z']].as_matrix()

        type_choose = type_in[idx]
        cnt_choose = cnt_in[idx]

        if type_choose == 12:
            mat_type = 2
        elif type_choose == 9 or type_choose == 15:
            mat_type = 4
        elif type_choose == 18:
            mat_type = 2
        else:
            mat_type = -1


        ##remember to doublecheck these values

        if mat_type != -1:

            df_mat = df_xyz.loc[df_xyz['type'] == mat_type]
            mat_xyz = df_mat.loc[:, ['x', 'y', 'z']].as_matrix()

            D, _, _ = (compute_distance(fun_xyz, mat_xyz))
            D_select = D[D < D_thr]

            if len(D_select) > 0:
                fun_out.append(fun)
                type_out.append(type_choose)
                cnt_out.append(cnt_choose)

    #print "***********************"
    #print "D_selct: ", D_select
    #print "Type in: ", type_in
    #print type_out
    #print fun_out

    return np.asarray(fun_out), np.asarray(cnt_out), np.asarray(type_out)



def get_circle_center(df, type_choose=22):

    df_cnt = df.loc[df['type'] == type_choose]

    xy_mat = df_cnt.loc[:, ['x', 'y']].as_matrix()

    x_m = np.mean(xy_mat[:, 0])
    y_m = np.mean(xy_mat[:, 1])
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(xy_mat[:, 0], xy_mat[:, 1]))
    Ri = calc_R(xy_mat[:, 0], xy_mat[:, 1], *center)
    R = Ri.mean()



    return center, R


def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)


def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)

    return Ri - Ri.mean()




def global_features(fun_id, cnt_id, center, df, N_max = 5):

    ##this function computes global function

    #epsilon_1 = epsilon 2 -> to distinguis single functionalized
    epsilon_1 = 0.10
    epsilon_2 = 0.10

    #this function identified the four global features
    df_fun = df[df['id'].isin(fun_id)]
    fun_xyz = df_fun.loc[:, ['x', 'y', 'z']].as_matrix()

    theta_all = []
    z_all = []

    #print "fun_xyz: "
    #print fun_xyz

    if len(fun_id) > 1:

        for i in range(fun_xyz.shape[0]):

            #first let's deal with theta

            v_1 = fun_xyz[i, 0:2] - center
            v_2 = fun_xyz[i+1:, 0:2] - center


            #print v_2, v_1


            if len(v_2) > 0:
                for j in range(len(v_2)):
                    v_2p = v_2[j, :]

                    theta_p = math.acos(np.dot(v_1, v_2p)/(np.linalg.norm(v_1)*np.linalg.norm(v_2p)))


                    theta_all.append(theta_p)


        theta_all = np.sort(np.asarray(theta_all))



        theta_out = theta_all[0:(fun_id.shape[0])]
        var_theta_1 =  epsilon_1 + np.mean(np.sin(theta_out/fun_xyz.shape[0]))
        var_theta_2 = epsilon_1 + np.sin(np.amax(theta_out/2))

        #print "theta_out: ", theta_out
        #print "var theta: ", np.sum(np.sin(theta_out/fun_xyz.shape[0]))


    else:

        theta_all.append(2*math.pi)
        theta_out = np.asarray(theta_all)
        var_theta_1 = epsilon_1
        var_theta_2 = epsilon_2
        #print "theta_out: ", theta_out


    #######This block of code detects how evenly the cnts are functionalized in the z-direction

    #cnt id: location o c

    cnt_mat, z_len = compute_z(df=df)

    if len(cnt_id) > 1:

        for i in range(0, len(cnt_id)):

            df_1 = df.loc[df['id']==cnt_id[i]]
            cnt_1 = df_1.loc[:, ['x', 'y', 'z']].as_matrix()
            z_1 = cnt_1[:,-1]


            for j in range(i+1, len(cnt_id)):

                df_2 = df.loc[df['id'] == cnt_id[j]]
                cnt_2 = df_2.loc[:, ['x', 'y', 'z']].as_matrix()

                #periodic cnt atom
                cnt_2p = check_periodic(cnt_z=cnt_2, z_len=z_len)

                delta_z1 = np.abs(cnt_1[:, -1] - cnt_2[:, -1])
                delta_z2 = np.abs(cnt_1[:, -1] - cnt_2p[:, -1])

                #print "delta_z1: ", delta_z1
                #print "delta z2: ", delta_z2
                #print "z_len: ", z_len

                delta_z = np.amin([delta_z1, delta_z2])


                #print "delta z: ", delta_z
                z_all.append(delta_z)

        z_out = np.sort(np.asarray(z_all))
        z_out = z_out[0:len(cnt_id)-1]

        #print "z_out: ", z_out
        var_z1 = epsilon_1 + np.mean(np.sin((z_out/(z_len/len(cnt_id)))*(math.pi/2)))
        var_z2 = epsilon_2 + np.sin(np.amax(z_out/(z_len/2))*(math.pi/2))



    else:

        delta_z = 0.0
        z_all.append(delta_z)
        var_z1 = epsilon_1
        var_z2 = epsilon_2

    globa_var = np.asarray([var_theta_1, var_theta_2, var_z1, var_z2, len(fun_id)/N_max])

    return globa_var


def global_features_NEWGEOM(fun_id, cnt_id, center, df, N_max = 5):

    ##this function computes global function

    #epsilon_1 = epsilon 2 -> to distinguis single functionalized
    epsilon_1 = 0.10
    epsilon_2 = 0.10

    #this function identified the four global features
    df_fun = df[df['id'].isin(fun_id)]
    fun_xyz = df_fun.loc[:, ['x', 'y', 'z']].as_matrix()

    x_all = []
    y_all = []

    #print "fun_xyz: "
    #print fun_xyz


    #######This block of code detects how evenly the cnts are functionalized in the z-direction

    #cnt id: location o c

    cnt_mat, z_len = compute_z(df=df)

    if len(cnt_id) > 1:

        for i in range(0, len(cnt_id)):

            df_1 = df.loc[df['id']==cnt_id[i]]
            cnt_1 = df_1.loc[:, ['x', 'y', 'z']].as_matrix()
            x_1 = cnt_1[:,-3]
            y_1 = cnt_1[:,-2]
            z_1 = cnt_1[:,-1]


            for j in range(i+1, len(cnt_id)):

                df_2 = df.loc[df['id'] == cnt_id[j]]
                cnt_2 = df_2.loc[:, ['x', 'y', 'z']].as_matrix()

                #periodic cnt atom
                cnt_2p = check_periodic(cnt_z=cnt_2, z_len=z_len)

                delta_z1 = np.abs(cnt_1[:, -1] - cnt_2[:, -1])
                delta_z2 = np.abs(cnt_1[:, -1] - cnt_2p[:, -1])

                #print "delta_z1: ", delta_z1
                #print "delta z2: ", delta_z2
                #print "z_len: ", z_len

                delta_z = np.amin([delta_z1, delta_z2])


                #print "delta z: ", delta_z
                z_all.append(delta_z)

        z_out = np.sort(np.asarray(z_all))
        z_out = z_out[0:len(cnt_id)-1]

        #print "z_out: ", z_out
        var_z1 = epsilon_1 + np.mean(np.sin((z_out/(z_len/len(cnt_id)))*(math.pi/2)))
        var_z2 = epsilon_2 + np.sin(np.amax(z_out/(z_len/2))*(math.pi/2))



    else:

        delta_z = 0.0
        z_all.append(delta_z)
        var_z1 = epsilon_1
        var_z2 = epsilon_2

    globa_var = np.asarray([var_theta_1, var_theta_2, var_z1, var_z2, len(fun_id)/N_max])

    return globa_var



def compute_z(df, type_Choose = 22):

    df_cnt = df.loc[df['type']==type_Choose]
    cnt_mat = df_cnt.loc[:, ['x', 'y', 'z']].as_matrix()
    z_len = np.amax(cnt_mat[:, 2]) - np.amin(cnt_mat[:, 2])



    return cnt_mat, z_len






def check_periodic(cnt_z, z_len):

    z_pick = cnt_z[:, -1]

    z_min = np.amin(cnt_z[:, 2])
    z_max = np.amax(cnt_z[:, 2])
    z_mid = 0.5*(z_min + z_max)



    print( cnt_z)

    if z_pick < z_mid:
        z_pick = z_pick + z_len
    else:
        z_pick = z_pick - z_len


    cnt_out = cnt_z.copy()
    cnt_out[:, -1] = z_pick


    return cnt_out



def thread_check(new_T, T_list, create_thread=False, **kwargs):

    idx_to_del = -10
    node_list = []


    if 'bool_array' in kwargs and 'bool_thread' in kwargs:
        bool_array = kwargs['bool_array']
        bool_thread = kwargs['bool_thread']
    else:
        bool_thread = []
        bool_array = np.zeros((1, 3))

    #new T-> new thread
    #T_list -> current list
    #create_thread -> new thread
    if 'og_list' in kwargs:
        og_list = kwargs['og_list']
        create_thread = True

    if 'n_value' in kwargs:
        n = kwargs['n_value']

    if 'fun_atom' in kwargs:
        fun_atom = kwargs['fun_atom']

    if 'node_list' in kwargs:
        node_list = kwargs['node_list']


    if 'og_node' in kwargs:
        og_node = kwargs['og_node']
    else:
        og_node = []

    if 'og_bool' in kwargs:
        og_bool = kwargs['og_bool']
    else:
        og_bool = np.zeros((1, 3))



    if 'adjacency_list' in kwargs and 'chain_length' in kwargs:
        adjacency_list = kwargs['adjacency_list']
        print( kwargs['chain_length'])
        chain_length = kwargs['chain_length'][0][0]
        chain_str = [chain_length]

    if 'og_adjacency' in kwargs:
        og_adjacency = kwargs['og_adjacency']
    else:
        og_adjacency = []

    ##KEEP TRACK OF END ATOMS!!!!
    #print "T_list init: "
    #print T_list

    count_idx = 0

    if len(T_list) == 0:
        T_list.append(new_T)
        og_list = []

        #

        ###this block is to keep track of all the nodes in our threads:
        if 'node_list' in kwargs:
            node_q = [new_T[0], new_T[-1]]
            node_list.append(node_q)
            #initialize og_node
            og_node = []


        #adjacency block
        if 'adjacency_list' in kwargs and 'chain_length' in kwargs:
            adjacency_list.append(chain_str)
            og_adjacency = []


        if 'bool_array' in kwargs and 'bool_thread' in kwargs:
            bool_out = np.concatenate((np.zeros((1, 3)), bool_array), axis=0)
            bool_thread.append(bool_out)
            og_bool = np.zeros((1, 3))

            print( "woo")


    elif create_thread == True:
        #scan through the existing list

        if n == 0 and 'fun_atom' in kwargs:
            mod_T = new_T

            if 'node_list' in kwargs:
                node_q = [new_T[0], new_T[-1]]
                node_list.append(node_q)

            if 'adjacency_list' in kwargs and 'chain_length' in kwargs:
                adjacency_list.append(chain_str)
                og_adjacency = []

            if 'bool_array' in kwargs and 'bool_thread' in kwargs:
                bool_out = np.concatenate((og_bool, bool_array), axis=0)
                bool_thread.append(bool_out)
                og_bool = np.zeros((1, 3))


        else:
            mod_T = og_list + new_T[1: ]

            if 'node_list' in kwargs:
                #print "og-node: ", og_node
                node_q = og_node + [new_T[-1]]
                node_list.append(node_q)

            if 'adjacency_list' in kwargs and 'chain_length' in kwargs:
                #print "case II adjacency: ", og_adjacency
                chain_out = og_adjacency + chain_str
                adjacency_list.append(chain_out)
                og_adjacency = []

            if 'bool_array' in kwargs and 'bool_thread' in kwargs:
                bool_new = og_bool[-1, :] + bool_array
                bool_out = np.concatenate((og_bool, bool_new), axis=0)
                bool_thread.append(bool_out)
                og_bool = np.zeros((1, 3))







        T_list.append(mod_T)
        #print "og list 1:", og_list
        og_list = []
        #print "mod T 1: ", mod_T

    else:
        for idx, list in enumerate(T_list):

            #print "list chck 0811: ", list
            if list[-1] == new_T[0]:
                og_list = list
                mod_T = list + new_T[1:] #[1: ] to indicate that we dont need the first element
                T_list.append(mod_T)
                idx_to_del = idx

                if 'node_list' in kwargs:
                    node_q = node_list[idx] + [new_T[-1]]
                    node_list.append(node_q)

                ###adjacency matrix setup:
                if 'adjacency_list' in kwargs and 'chain_length' in kwargs:
                    chain_out = adjacency_list[idx] + chain_str
                    adjacency_list.append(chain_out)
                    og_adjacency = adjacency_list[idx]
                    og_node = node_list[idx]

                ##extract bool
                if 'bool_array' in kwargs and 'bool_thread' in kwargs:
                    #print bool_array.shape
                    #print bool_thread[idx].shape
                    bool_choose = bool_thread[idx][-1, :]
                    bool_new = bool_choose + bool_array
                    #print bool_new.shape
                    #print bool_thread[idx].shape
                    bool_out = np.concatenate((bool_thread[idx], bool_new), axis=0)
                    #print bool_out.shape
                    bool_thread.append(bool_out)
                    og_bool = bool_thread[idx]




                break

                #print "og list 2:", og_list
                #print "mod T 2: ", mod_T
            else:

                count_idx += 1


            if count_idx == len(T_list):
                og_list = []
                mod_T = new_T
                T_list.append(mod_T)

                if 'node_list' in kwargs:
                    node_q = [new_T[0], new_T[-1]]
                    node_list.append(node_q)

                if 'adjacency_list' in kwargs and 'chain_length' in kwargs:
                    adjacency_list.append(chain_str)
                    og_adjacency = []

                if 'bool_array' in kwargs and 'bool_thread' in kwargs:
                    bool_out = np.concatenate((np.zeros((1, 3)), bool_array), axis=0)
                    bool_thread.append(bool_out)
                    og_bool = np.zeros((1, 3))


                break


                #print "og list 2B:", og_list




    print( "index to delete: ", idx_to_del)

    if idx_to_del >= 0:
        del T_list[idx_to_del]
        del node_list[idx_to_del]
        del adjacency_list[idx_to_del]
        del bool_thread[idx_to_del]

    return T_list, og_list, node_list, og_node, adjacency_list, og_adjacency, bool_thread, og_bool


##need a function to get the l2 distance(id_1, id_2,



def trace_origin(init_atom, node_list, node_type_list):

    epoxy_bool = False
    orig_atom = -1

    for idx, list in enumerate(node_list):

        #print "list: ", list
        #print "node list: ", node_type_list[idx]

        if list[-1] == init_atom:  #find where the init atom terminates
            orig_atom = list[0]
            node_seq = node_type_list[idx]
            #print "node list loop: ", node_type_list[idx]

            if node_seq[-2] == 4:
                epoxy_bool = False
            else:
                epoxy_bool = True
            break


    return orig_atom, epoxy_bool




def find_taken_C(init_atom, node_seq_list, type_seq_1):

    taken_C = -1

    print( "seq 1: --------------------------------------------------------------------")
    print( "seq 1: ", type_seq_1)
    if type_seq_1[1] == 4: #if resin sequence is N_N -> it indicates that bonded atom


        #track node_seq_list to see if it exists already
        for idx, list in enumerate(node_seq_list):

            if list[-1] == init_atom:
                taken_C = list[-2]
                break






    return taken_C





def check_reverse(atom_seq, node_seq_list):



    if len(node_seq_list) > 0:

        for list in (node_seq_list):

            if atom_seq == list.reverse():
                atom_seq = -1



    return atom_seq



def thread_check02(new_T, T_list, create_thread=False, **kwargs):

    idx_to_del = -10


    #new T-> new thread
    #T_list -> current list
    #create_thread -> new thread
    if 'og_list' in kwargs:
        og_list = kwargs['og_list']
        create_thread = True

    ##KEEP TRACK OF END ATOMS!!!!
    #print "T_list init: "
    #print T_list

    count_idx = 0

    if len(T_list) == 0:
        T_list.append(new_T)
        og_list = []
    elif create_thread == True:
        #scan through the existing list
        mod_T = og_list + new_T[1: ]
        T_list.append(mod_T)
        print( "og list 1:", og_list)
        og_list = []
        #print "mod T 1: ", mod_T

    else:
        for idx, list in enumerate(T_list):

            print( "list chck 0811: ", list)
            if list[-1] == new_T[0]:
                og_list = list
                mod_T = list + new_T[1:] #[1: ] to indicate that we dont need the first element
                T_list.append(mod_T)
                idx_to_del = idx

                break

                #print "og list 2:", og_list
                #print "mod T 2: ", mod_T
            else:

                count_idx += 1


            if count_idx == len(T_list):
                og_list = []
                mod_T = new_T
                T_list.append(mod_T)


                #print "og list 2B:", og_list




    #print "index to delete: ", idx_to_del
    if idx_to_del >= 0:
        del T_list[idx_to_del]

    return T_list, og_list




def get_adjusted_d(p_1, p_2, df_xyz, L_array, bool_check, d_thresh=2.0):

    #L_x = L_array[0]
    #L_y = L_array[1]
    #L_z = L_array[2]


    df_1 = df_xyz.loc[df_xyz['id'] == p_1]
    xyz_1 = df_1.loc[:, ['x', 'y', 'z']].as_matrix().flatten()


    df_2 = df_xyz.loc[df_xyz['id'] == p_2]
    xyz_2 = df_2.loc[:, ['x', 'y', 'z']].as_matrix().flatten()


    diff = (xyz_1 - xyz_2)
    idx = np.where(diff > d_thresh)

    xyz_2p = xyz_2.copy()

    for dim in range(0, 3):

        L_add = np.zeros((3, ))

        if np.abs(diff[dim]) > d_thresh:
            L_add[dim] = L_array[dim]

            if diff[dim] > 0:
                xyz_2p = xyz_2p + L_add
                bool_check[0, dim] = bool_check[0, dim] + 1
            else:
                xyz_2p = xyz_2p - L_add
                bool_check[0, dim] = bool_check[0, dim] - 1

    #modify second coordinate
    #xyz_1p = xyz_1 + np.multiply(L_array, n_1)
    #xyz_2p = xyz_2 + np.multiply(L_array, n_2)



    #print "xyz_1: ", xyz_1
    #print "xyz_2: ", xyz_2
    #print "xyz 2p: ", xyz_2p
    #print "diff: ", diff
    #print "index : ", idx

    d_1, _, _ = compute_distance(xyz_1[None, :], xyz_2p[None, :])

    d_1 = d_1.flatten()[0]

    #print "d_1: ", d_1

    #keep p_1 fixed, #check periodicity of p_2
    #p_choose = 100*np.ones(16, )

    #p_choose[0] = p_2 + np.asarray([L_x, 0, 0])
    #p_choose[1] = p_2 + np.asarray([-L_x, 0, 0])
    #p_choose[2] = p_2 + np.asarray([L_x, L_y, 0])
    #p_choose[3] = p_2 + np.asarray([L_x, -L_y, 0])
    #p_choose[4] = p_2 + np.asarray([-L_x, +L_y, 0])
    #p_choose[5] = p_2 + np.asarray([L_x, L_y, L_z])
    #p_choose[6] = p_2 + np.asarray([-L_x, L_y, L_z])
    #p_choose[6] = p_2 + np.asarray([-L_x, L_y, L_z])


    return d_1, bool_check




####this function indicates whether or not atom_seq exited the boundary or not
def periodic_fun(atom_seq, L_array, df_xyz, bool_check):

    bool_array = np.zeros((3, ))
    bool_check = bool_check.flatten()

    init_atom = atom_seq[0]
    df_1 = df_xyz.loc[df_xyz['id'] == init_atom]
    xyz_1 = df_1.loc[:, ['x', 'y', 'z']].as_matrix().flatten()


    end_atom = atom_seq[-1]
    df_2 = df_xyz.loc[df_xyz['id'] == end_atom]
    xyz_2 = df_2.loc[:, ['x', 'y', 'z']].as_matrix().flatten()

    diff = (xyz_1 - xyz_2)


    xyz_2p = xyz_2

    for dim in range(0, 3):

        L_add = np.zeros((3,))

        #if np.abs(diff[dim]) > d_thresh:
        if bool_check[dim] !=0:
            #print "woo 1"
            L_add[dim] = L_array[dim]
            #print "L_add: ", L_add
            #print "L_array: ", L_array

            if bool_check[dim] > 0:
                #print "woo 2"
                xyz_2p[dim] = xyz_2[dim] + L_add[dim]
                bool_array[dim] = bool_array[dim] + 1

            else:
                xyz_2p[dim] = xyz_2[dim] - L_add[dim]
                bool_array[dim] = bool_array[dim] - 1


    ##compute_distanc
    #d_1, _, _ = compute_distance(xyz_1, xyz_2p)
    #d_1 = d_1.flatten()[-1]

    #bool_array = bool_array[:, None]


    return bool_array, xyz_1, xyz_2, xyz_2p



def extract_graph_features(node_list, bool_list, df_xyz, max_degree, num_features=10):

    path_features = np.zeros((len(node_list), max_degree, num_features))
    #for idx in range(0, len(node_list)):
    for idx, list in enumerate(node_list):
        thread_node = node_list[idx]

        x_1 = get_type_mat(thread_node_list=list, df_xyz=df_xyz)
        x_2 = feature_periodic(thread_periodic=bool_list[idx])
        x_3 = get_CNT_distance(thread_node_list=list, df_xyz=df_xyz)

        #print x_1.shape
        #print x_2.shape
        #print x_3.shape
        #print node_list[idx]

        x = np.concatenate((x_1, x_2, x_3), axis=1)

        ##this block of code for special cases where max_degree < len(x) -> multiple ends
        #this occurs when more more than one thread has the same endpoint
        if len(x) > max_degree:
            path_features[idx, 0:max_degree, :] = x[0:max_degree, :]
        else:
            path_features[idx, 0:len(x), :] = x





    return path_features




def get_type_mat(thread_node_list, df_xyz, ref_list=[2, 4, 9, 12, 15, 16, 18]):

    x_mat = np.zeros((len(thread_node_list), len(ref_list)))

    for idx, atom_id in enumerate(thread_node_list):

        df_1 = df_xyz[df_xyz['id']==atom_id]
        type_mat = df_1.loc[:, ['type']].as_matrix().flatten()
        type_1 = type_mat[0]

        #print "atom id: ", atom_id
        #print "type 1: ", type_1

        idx_select = np.where(np.asarray(ref_list)==type_1)[0]

        idx_c = idx_select.flatten()[0]

        x_mat[idx, idx_c] = 1


    return x_mat



def feature_periodic(thread_periodic):

    x_mat = np.zeros((len(thread_periodic), 2))



    abs_array = np.absolute(thread_periodic)

    #take the periodic in z
    x_mat[:, 1] = abs_array[:, 2]
    x_mat[:, 0] = np.sum(abs_array[:, 0:2], axis=1)





    return x_mat



def get_CNT_distance(thread_node_list, df_xyz, type_choose = 22, L_norm=45.0):

    x_mat = np.zeros((len(thread_node_list), 1))

    df_cnt = df_xyz[df_xyz['type']==type_choose]
    mat_1 = df_cnt.loc[:, ['x', 'y', 'z']].as_matrix()

    for idx, atom_id in enumerate(thread_node_list):
        df_1 = df_xyz[df_xyz['id'] == atom_id]
        mat_2 = df_1.loc[:, ['x', 'y', 'z']].as_matrix()

        d_1, _, _ = compute_distance(mat_1, mat_2)
        d_min = np.amin(d_1.flatten())

        x_mat[idx] = 1 - d_min/L_norm




    return x_mat


def compute_gamma_d(atom_seq, df_xyz):

    init_atom = atom_seq[0]
    df_1 = df_xyz[df_xyz['id'] == init_atom]
    mat_1 = df_1.loc[:, ['x', 'y', 'z']].as_matrix()

    end_atom = atom_seq[-1]
    df_2 = df_xyz[df_xyz['id'] == end_atom]
    mat_2 = df_2.loc[:, ['x', 'y', 'z']].as_matrix()

    d_1, _, _ = compute_distance(mat_1, mat_2)
    d_1 = d_1.flatten()[-1]



    return d_1


def get_adjacency_matrix(a_list, max_degree):

    A_mat = np.zeros((len(a_list), max_degree, max_degree))
    A_tilda = np.zeros_like(A_mat)


    for idx, list in enumerate(a_list):

        for i in range(len(list)):



            if i < max_degree - 1:
                 #this block of code to accomodate cases where there are more than one endpoints
                 A_mat[idx, i, i+1] = list[i]
                 A_mat[idx, i+1, i] = list[i]

        A_tilda[idx, :, :] = A_mat[idx, :, :] + np.identity(max_degree)

    return A_tilda


def get_diag_mat(A_mat):

    D_mat = np.zeros_like(A_mat)

    for idx in range(A_mat.shape[0]):

        d = 1/np.sqrt(np.sum(A_mat[idx, :, :], axis=1))
        D_mat[idx, :, :] = np.diag(d)




    return D_mat



def get_Z_mat(D, A, X):

    Z = np.zeros((X.shape[0], X.shape[1], X.shape[2]))

    for idx in range(D.shape[0]):
        z_1 = np.matmul(D[idx, :, :], A[idx, :, :])
        z_2 = np.matmul(z_1, D[idx, :, :])
        z_3 = np.matmul(z_2, X[idx, :, :])

        Z[idx, :, :] = z_3



    return Z



def get_unique_Z(Z):

    #print "****************"

    Z_1 = np.zeros((Z.shape[1], Z.shape[2]))
    Z_2 = np.zeros_like(Z_1)

    for row in range(0, Z.shape[1]):

        Z_p = (Z[:, row, :])
        Z_u = np.vstack({tuple(c) for c in Z_p})
        Z_s = np.sum(Z_u, axis=0)
        Z_max = np.max(Z_u, axis=0)

        Z_1[row, :] = Z_s
        Z_2[row, :] = Z_max
        #print Z[:, row, :].shape



    return Z_1, Z_2, Z_u

