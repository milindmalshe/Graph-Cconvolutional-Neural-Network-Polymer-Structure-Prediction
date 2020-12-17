import numpy as np
import pandas as pd

import feature_extract_01

def make_changes(file_in, file_out, sim_file_in, sim_file_out, change_dict):

    #delete output file
    f = open(file_out, 'r+')
    f.truncate(0)


    copy_data_file(file_in, file_out)
    #detect number of atoms in file_out
    total_new_atoms = len(change_dict['new_atoms'])

    with open(file_out, "r") as f1:
        lines = f1.readlines()

    line_words = lines[3].split()
    num_atoms = int(line_words[0])
    total_atoms = num_atoms + total_new_atoms
    my_list = [str(total_atoms), line_words[1], line_words[2]]
    my_list = " ".join(my_list)

    #lines[3] = (" ".join(my_list)).rstrip() + "\n"
    lines[3] = my_list.rstrip() + "\n"

    #this block is for preparing the "new atoms"
    out_str = []
    mass_list = change_dict['Masses']


    for i in range(0, total_new_atoms):
        atom_type = i + 1 + num_atoms
        mass_i = mass_list[i]
        my_list = [str(atom_type), str(mass_i)]
        my_list = (" ".join(my_list)).rstrip() + "\n"
        lines.insert(11 + num_atoms + i, my_list)


    #block to index out_str
    with open(file_out, "w") as f2:
        f2.writelines(lines)


    #This block is to copy the simulation files:
    copy_data_file(sim_file_in, sim_file_out)

    with open(sim_file_out, "r") as f3:
        sim_lines = f3.readlines()


    for i, line in enumerate(sim_lines):
        line_words = line.split()

        if len(line_words) > 1 and line_words[0]=='set' and line_words[1]=='group':
            idx = i
            print "idx: ", idx
            break


    #this block is to prepare commands to insert into the main text file
    coords_array = change_dict['coords']
    for i in range(0, total_new_atoms):
        atom_type = i + 1 + num_atoms
        mass_i = mass_list[i]
        x_val = coords_array[i, 0]
        y_val = coords_array[i, 1]
        z_val = coords_array[i, 2]
        my_list = ['create_atoms', str(atom_type), 'single', str(x_val), str(y_val), str(z_val)]
        my_list = (" ".join(my_list)).rstrip() + "\n"
        sim_lines.insert(idx + 1, my_list)



    #mpw write to file
    with open(sim_file_out, "w") as f4:
        f4.writelines(sim_lines)


    return my_list




def copy_data_file(file_in, file_out):
    with open(file_in) as f:

        count_atoms = -1
        with open(file_out, "w") as f1:
            for line in f:
                f1.write(line)



    return None


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


def detect_bond_length_id(df_list, id_1, id_2):


    dist_mat = []
    for t in range(0, len(df_list)):

        df = df_list[t]

        p_1 = df.loc[df['id']==id_1]
        p_2 = df.loc[df['id']==id_2]
        print( p_1)
        print( p_2)

        p_1 = p_1.loc[:, ['x', 'y', 'z']].as_matrix()
        p_2 = p_2.loc[:, ['x', 'y', 'z']].as_matrix()


        dist = np.sum(((p_1 - p_2) ** 2), axis=1)
        dist = np.sqrt(dist)
        #dist = dist.flatten()
        dist_mat.append(dist)




    return np.asarray(dist_mat)
