import numpy as np
import sys
import pandas as pd

file_to_read = str(sys.argv[1]) #list of sequence ID's and forces
opt_fun = int(sys.argv[2]) #selection of functionals
f_p = float(sys.argv[3]) #pullout force
model_file = str(sys.argv[4]) #modelfile used to
seq_file = str(sys.argv[5]) #init data csv
cntid = int(sys.argv[6]) #CNT ID
funid = int(sys.argv[7]) #functional id

def update_forcefile(filename, f_p):


    #first let us update the forcefile and extract id
    Z = np.loadtxt(fname=filename, skiprows=1, delimiter=",")
    seq_id = Z[:, 0]


    new_id = seq_id[-1] + 1
    new_arr = np.asarray([new_id, f_p])
    Z2 = np.vstack((Z, new_arr))

    np.savetxt('test.csv', Z2, delimiter=",")


    return new_id, Z2



def update_fun(filename, modelfile, opt_fun, cntid, funid):

    df = pd.read_table(filename, delimiter=",")


    #get index of columns
    idx_cnt1 = df.columns.get_loc("cnt1")
    idx_opt1 = df.columns.get_loc("opt1")
    idx_fun1 = df.columns.get_loc("fun1")

    lastfile = df['file_new'].iloc[-1]
    str_out, num_choose = edit_datafile(modelfile=lastfile, fill_num=5)

    #we need to update this last file



    if np.any((df['file_new'] == model_file).as_matrix()):

        df_choose = df.where(df['file_new']==modelfile)
        df_choose = df_choose.dropna().copy()
        used_fun = df_choose.loc[:, ['opt1', 'opt2', 'opt3', 'opt4', 'opt5']].as_matrix()
        idx = len((used_fun[used_fun > 0]))  # list of used function

    else:

        idx = 0
        df_choose = df[0:1].copy()



    # changing df_choose to reflect modifications to the new dataframe
    print( df_choose)
    df_choose = df_choose.reset_index(drop=True)
    print( df_choose)
    df_choose.ix[0, 'seq_ID'] = num_choose

    df_choose.iloc[0, idx_cnt1 + idx] = cntid
    df_choose.iloc[0, idx_fun1 + idx] = funid
    df_choose.iloc[0, idx_opt1 + idx] = opt_fun

    # put in new and old filename
    df_choose.ix[0, 'file_old'] = modelfile
    df_choose.ix[0, 'file_new'] = str_out

    df = df.append(df_choose, ignore_index=True)
    df.to_csv('datatest.csv', sep=',')

    #print df

    return df


def edit_datafile(modelfile, fill_num):

    #fill_num -> how many digits

    str_choose = modelfile.split('.')[1]
    num_choose = int(str_choose) + 1
    str_choose = str(num_choose).rjust(fill_num, '0')

    str_out = 'data.' + str(str_choose)

    return str_out, num_choose


if __name__ == "__main__":
    new_id, z2 = update_forcefile(filename=file_to_read, f_p=f_p)

    df_choose = update_fun(filename=seq_file, modelfile=model_file, opt_fun=opt_fun, cntid=cntid, funid=funid)


    print( new_id)
    print( df_choose)
