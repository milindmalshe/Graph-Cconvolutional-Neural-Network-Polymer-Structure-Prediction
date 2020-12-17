import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance
import mpi4py
import pandas as pd
import re
import glob


from enum import Enum

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

    def __init__(self, MD_file, f_value, chain_file, classname=BFDGE):

        ##MD file -> MD file sin question
        #f_value -> force value
        self.classname = classname
        self.file_name = MD_file
        self.f_value = f_value
        self.chain_file = chain_file
        #this class identifies each MD model, and for each model, identify all the clusters

        #seq_to_pick = self.load_resin(self, class_name=BFDGE, seq_type=O_C)

        d_thr, self.id_fun, id_cnt, self.fun_type, self.df_xyz, self.df_bond = self.detect_fun_atoms()
        print "troubleshoot: ", d_thr, self.id_fun, id_cnt, self.fun_type

        #find_chain_particles(sequence=[12, 2, 2, 3, 4, 4, 4, 4,  2, 4, 4, 4, 4, 3, 2, 2, 12], init_id=3770, df_xyz=self.df_xyz, df_bond=self.df_bond)

        ###this block loads the chains
        self.execute_chain()

        #if ('force_wpattern' in kwargs):

           #force_wpattern = kwargs['force_wpattern']
           #self.MD_file_list, self.f_file_list = read_MD_files(list_of_files=MD_wpattern, forcefile=force_wpattern)


        #else:

            #self.MD_file_list, _ = read_MD_files(list_of_files=MD_wpattern)



    def execute_chain(self, max_degree=10):

        #this function takes in the inital atom id as input and generates a staggered list containing the atom ids
        #this should be done for each new funtionalized atom
        #max_degree -> maximum degree of removal from the cnt atom

        #initialize lists
        next_atoms = []
        next_atom_types = []

        stag_list = []

        for idx, fun_id in enumerate(self.id_fun):

            for n in range(0, max_degree):

                print "-------------------------------------New degree of removal---------------------"

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



                for i in range(0, len(init_id)):

                    print "!!!!!!!!!New atom id !!!!!!!!!!!!!!"
                    print init_id

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
                    seq_1, seq_2, num_pathway = self.load_resin_molecule(fun_bool=fun_bool,
                                                                                    fun_type=fun_val, end_type=end_type)

                    #print "Seq 1: ", seq_1
                    #print "Seq 2: ", seq_2, fun_bool, fun_val
                    #print seq_1, seq_2, num_pathway
                    if seq_1 == -1:
                        break

                    atom_seq_1, type_seq_1 = self.track_atoms(sequence_atoms=seq_1, init_atom=init_pick)
                    end_atom, end_type = atom_seq_1[-1], type_seq_1[-1]

                    if len(next_atoms) > 0:
                        next_atoms = next_atoms + [end_atom]
                        next_atom_types = next_atom_types + [next_atom_types]

                    else:
                        next_atoms = [end_atom]
                        next_atom_types = [end_type]


                    #print "atom seq 1: ", atom_seq_1

                    if num_pathway == 2:
                        print "Alternate Branch Entry: "
                        print seq_2
                        atom_seq_2, type_seq_2 = self.track_atoms(sequence_atoms=seq_2, init_atom=init_pick)
                        end_atom_2, end_type_2 = atom_seq_2[-1], type_seq_2[-1]
                        next_atoms = next_atoms + [end_atom_2]
                        next_atom_types = next_atom_types + [end_type_2]
                        print "atom_Seq_2: ", atom_seq_2


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

                    print "atom seq 1: ", atom_seq_1
                    print "type seq 1: ", type_seq_1
                    print "atom seq 2: ", atom_seq_2
                    print "type seq 2: ", type_seq_2
                    print "next atoms: ", next_atoms
                    print "next atom types: ", next_atom_types
                    print "fun id: ", fun_id


        print "next_atoms: ", next_atoms
        print "next_atom_types: ", next_atom_types
        print "all_list: ", np.unique(all_list).tolist()
        print "all type: ", all_type
        print "all length ", len(np.unique(all_list).tolist())
        print "fun id: ", self.id_fun




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

        print "troubleshoot", skip_count, vel_count, bond_count

        df_xyz = pd.read_table(self.chain_file, delim_whitespace=True, header=None, skiprows=skip_count,  nrows=(vel_count - skip_count))
        df_bond = pd.read_table(self.chain_file, delim_whitespace=True, header=None, skiprows=bond_count)



        return df_xyz, df_bond


    def load_resin_molecule(self, fun_bool, fun_type, end_type):

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
                seq_type = resin_sequence.resin_key
            elif end_type == 16:
                num_pathway = 2
                seq_type = resin_sequence.resin_key
                seq_type_2 = resin_sequence.resin_key
            elif end_type == 2: #in case of two
                num_pathway =2
                seq_type = resin_sequence.C_N
                seq_type_2 = resin_sequence.C_rC
            else:
                seq_type = -1
                seq_type_2 = -1


        print "seq_type: ", seq_type
        print "seq_type 2: ", seq_type_2

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



    def track_atoms(self, init_atom, sequence_atoms):


        ###define a function to track along the resin molecule
        #track the sequence of atoms


        #track if there are multiple options:

        for j in range(len(sequence_atoms)):

            seq_to_track = sequence_atoms[j]
            seq_tmp, type_tmp = find_chain_particles(sequence=seq_to_track, init_id=init_atom,
                                                     df_xyz=self.df_xyz, df_bond=self.df_bond) #change this

            #if the final sequence ends with -1, then we know we have the right sequence

            if seq_tmp[-1] != -1:

                seq_out = seq_tmp
                type_out = type_tmp

                break

        #def traverse_chain(fun_array, resin_type==1):


        return seq_out, type_out





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

    for i in range(0, len(file_list)):

        file_num = (file_list[i]).rsplit('.')[-1]
        id_f[i] = int(file_num)



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



def find_chain_particles(sequence, init_id, df_xyz, df_bond, symmetry_bool = False):



    id_mat = df_xyz.loc[:, ['id']].as_matrix()
    type_mat = df_xyz.loc[:, ['type']].as_matrix()
    bond_mat = df_bond.loc[:, ['particle_1', 'particle_2'] ].as_matrix()
    #This function takes in two argiments: sequence -> list of types for the sequence
    #and init_id -> the particle id of the first particle
    #Symmetry_bool -> indicates wether

    chain_id = [init_id]
    chain_type = [sequence[0]]
    count = 1

    #Init troubleshoot
    #idx_p = np.where(id_mat == 4131)
    #type_p = type_mat[idx_p[0]].flatten()

    #print "type p: ", type_p

    print "troubleshoot: ", init_id


    for type_choose in sequence[1:]:

        #type -> the type of atom
        #find where the atom sequence matches
        idx_all = np.where(bond_mat == init_id) #find all bonding where init_id is involved
        #bond_choose
        #extract out the rows
        bond_select =  bond_mat[idx_all[0]]
        other_particle = np.sort(bond_select[bond_select!=init_id]) #find the other bonded particle

        # print ""
        print "bond_select: ", bond_select
        print "other_particle: ", other_particle
        #print "id_mat: ", id_mat

        #Indexing out the "other particle"
        idx_tmp = index_mat(id_mat.flatten(), other_particle.flatten())
        #idx_tmp = np.where(id_mat == other_particle)

        print "idx_temp troubleshoot: ", idx_tmp
        #type_bond = type_mat[(idx_tmp[0])].flatten() ####this is the confusing part
        type_bond = type_mat[idx_tmp].flatten()

        print "type_bond: ", type_bond
        #print "type bond 2: ", type_bond_second
        idx_tmp = np.where(type_bond == type_choose)

        print "idx_temp: ", idx_tmp
        print "count: ", count
        ###this block of code ensures that when tracking benezene atoms

        if len(idx_tmp[0]) > 1 and count < len(sequence) and type_choose == 4:
            print "benzene block: "
            particle_tmp = other_particle[idx_tmp]
            print "particle tmp: ", particle_tmp
            idx_to_del =  np.where(np.isin(particle_tmp, np.asarray(chain_id)))
            particle_tmp = np.delete(particle_tmp, idx_to_del)
            print "particle tmp: ", particle_tmp
            #new_atom_id = particle_tmp[0] ### cehck this for benzene

            ##testing out the code for benzene
            #if
            if symmetry_bool == False and len(particle_tmp) > 1:
                new_atom_id = detect_particle_benzene(df_xyz=df_xyz, particle_1=particle_tmp[0],
                                                      particle_2=particle_tmp[1])
            else:
                new_atom_id = particle_tmp[0]




            #once the previous a

        elif len(idx_tmp[0]) > 0:

            new_atom_id = other_particle[idx_tmp][0]

        else:

            new_atom_id = -1


        #renew init_id
        chain_id.append(new_atom_id)
        chain_type.append(type_choose)

        if new_atom_id == -1:
            break

        init_id = new_atom_id
        count += 1
        #indx out from the main id matrix where new_atom id
        idx_tmp2 = np.where(id_mat == new_atom_id)

        #print "check: ", type_mat[idx_tmp2]

        #print "idx_tmp: ", idx_tmp
        print "new atom id: ", new_atom_id
        print "chian: : ", chain_id
        print "chain type: ", chain_type
        #print"type bond ", type_bond
        #select particles
            #for row in bond_select.shape[0]:


        #print "idx_all: "
        ##print idx_all




    print "chain_id: ", chain_id


    return chain_id, chain_type




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
    else:
        particle_out = particle_1

    return particle_out