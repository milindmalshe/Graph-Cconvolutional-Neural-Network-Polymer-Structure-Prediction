import make_datafile
import numpy as np
from  scipy.optimize import curve_fit
import feature_extract_01
import tensorflow_functions_01

import matplotlib.pyplot as plt

import task_3
import gp_forcepredict

file1 = 'data.3rr1epo'
file2 = 'data.3rr1epo_copy'
sim_file1 = 'in_slide2.txt'
sim_file2 = 'in_slide2_copy.txt'

change_dict = {'new_atoms': ['O'], 'Masses': [15.99], 'coords': np.array([[24.89, 28.87, 17.00]])}

df = make_datafile.read_datafile(file1)
df_type_unique = (df.loc[:, 'type']).unique()

df_fun, df_cnt = feature_extract_01.detect_locations(df=df,  threshold=4.0, fun_type=12)

#detect cnt atoms
df_fun02, df_R = feature_extract_01.detect_locations(df=df, threshold=2.0, type_choose=2, fun_id=4989)

#detct the hydrogen atom:
df_fun02, df_h = feature_extract_01.detect_locations(df=df, threshold=2.0, type_choose=1, fun_id=4989)


pointC = np.asarray([30.488, 36.77, 46.77])
pointO = np.asarray([31.73, 35.79, 46.48])
pointCNT = np.asarray([31.46, 34.20, 46.84])
print "distance: ", np.linalg.norm(pointCNT-pointO)
print "distance2: ", np.linalg.norm(pointC-pointO)

a = np.random.rand(100, 48, 61, 13)/10
b = np.random.rand(61, 13)/10
c = np.tensordot(a, b, 2)
d = np.ones_like(c)

#
z = np.linspace(3.0, 0.5, 100)
F2 = 1
F1 = 350
k_bar = np.divide(F1*(1-np.exp(-F2*z**2)), z**2)

print "k_bar: ", k_bar
print "mean(k_bar)", np.mean(k_bar)


#checking for syntaxt
#results =  tensorflow_functions_01.train_cnn(a, c)


v0 = np.load('v_nofun.npy')
v1 = np.load('v_funO.npy')
v2 = np.load('v_funC.npy')
v3 = np.load('v_funN1.npy')
v4 = np.load('v_funN2.npy')

v_CO = np.load('v_cnt.CO.npy')

f = np.load('f_file.npy')

max_idx = int(0.8*len(f))

#plt.plot(f, v0, label='CNT w/ oxygen in hydroxyl group in epoxy')
plt.plot(f[0:max_idx], v0[0:max_idx], label='No functionalization')
plt.plot(f[0:max_idx], v1[0:max_idx], label='CNT w/ oxygen in hydroxyl')
plt.plot(f[0:max_idx], v2[0:max_idx], label='CNT w/ carbon in epoxide')
plt.plot(f[0:max_idx], v3[0:max_idx], label='CNT w/ nitrogen in primary amine')
plt.plot(f[0:max_idx], v4[0:max_idx], label='CNT w/ nitrogen in secondary amine')
plt.legend(loc='upper left')
plt.xlabel('Pull-out force per atom (kcal/mol-$\AA$)')
plt.ylabel('Mean Velocity ($\AA$/fs)')
#plt.legend(loc='upper left')
plt.savefig('fig_velocity.eps')
plt.savefig('fig_pullout_velocity.svg', format='svg', dpi=5000)
plt.show()




#plt.legend(loc='upper left')

d0 = np.load('d_nofun.npy')
d1 = np.load('d_funO.npy')
d2 = np.load('d_funC.npy')
d3 = np.load('d_funN1.npy')
d4 = np.load('d_funN2.npy')


dv0 = np.gradient(d0.flatten(), 10*np.arange(0, len(d0)))

#plt.plot(f, v0, label='CNT w/ oxygen in hydroxyl group in epoxy')
plt.plot(f[0:max_idx], d0[0:max_idx], label='No functionalization')
plt.plot(f[0:max_idx], d1[0:max_idx], label='CNT w/ oxygen in hydroxyl', linewidth=2.0)
plt.plot(f[0:max_idx], d2[0:max_idx], label='CNT w/ carbon in epoxide', linewidth=2.0)
plt.plot(f[0:max_idx], d3[0:max_idx], label='CNT w/ nitrogen in primary amine', linewidth=2.0)
plt.plot(f[0:max_idx], d4[0:max_idx], label='CNT w/ nitrogen in secondary amine', linewidth=2.0)
plt.legend(loc='upper right')
plt.xlabel('Cummulative displacement (kcal/mol-$\AA$)')
plt.ylabel('Mean Velocity ($\AA$/fs)')
plt.savefig('fig_pullout.eps')
plt.show()

stride_0 = 20
res_0 = 500
delta_f = 1e-6
delta_0 = 0
dt = 0.1

f_0 = 10*np.load('f_0825.npy')
v_0 = np.load('v_0825.npy')
d_0 = np.load('d_0825.npy')

v_cntO = np.load('v_cnt.O.npy')
v_cntN1 = np.load('v_cnt.N1.npy')
#curve fit
t = dt*res_0*stride_0*np.arange(0, len(f_0))

#get features
X = gp_forcepredict.prep_features(f=f_0, f_min0=0, f_approx=0.06, t=t, tau_max=100000)




f_1 = 0.03 + 10*np.load('f_0827.npy')
v_1 = np.load('v_0827.npy')
d_1 = np.load('d_0827.npy')





t1 = dt*res_0*stride_0*np.arange(0, len(f_1))



f_2 = 0.02 + 10*np.load('f_0830.npy')
v_2 = np.load('v_0830.npy')
d_2 = np.load('d_0830.npy')

f_3 = 0.04 + 10*np.load('f_0830A.npy')
v_3 = np.load('v_0830A.npy')
d_3 = np.load('d_0830A.npy')


f_A = 0.03 + 10*np.load('f_0827A.npy')
v_A = np.load('v_0827A.npy')
d_A = np.load('d_0827A.npy')

f_B = 0.07 + 10*np.load('f_0827B.npy')
v_B = np.load('v_0827B.npy')
d_B = np.load('d_0827B.npy')

f_B = 0.07 + 10*np.load('f_0827B.npy')
v_B = np.load('v_0827B.npy')
d_B = np.load('d_0827B.npy')

plt.plot(f_0, v_0, 'b-', linewidth=3.0, label='f $\in$ [0, 0.1]')
plt.plot(f_2, v_2, 'g-', linewidth=3.0, label='f $\in$ [0.02, 0.07]')
#plt.plot(f_1, v_1, 'r-', linewidth=3.0, label='f $\in$ [0.03, 0.07]')
plt.plot(f_3, v_3, 'm-', linewidth=3.0, label='f $\in$ [0.04, 0.07]')
plt.axhline(y=0.01, color='k', linestyle='--', xmin=0, xmax=0.634, linewidth=3.0)
plt.axvline(x=0.06353, color='k', linestyle='--', ymin=0, ymax=0.230769, linewidth=3.0)
plt.axvline(x=0.05726, color='k', linestyle='--', ymin=0, ymax=0.230769, linewidth=3.0)
plt.axvline(x=0.06120, color='k', linestyle='--', ymin=0, ymax=0.230769, linewidth=3.0)
plt.xlim([0, 0.1])
plt.ylim([-0.002, 0.05])
plt.legend(loc='upper left')
plt.xlabel('Pull-out force per atom (kcal/mol-$\AA$)')
plt.ylabel('Mean Velocity ($\AA$/fs)')
plt.savefig('fig_short_intv.png')
plt.savefig('fig_short_intv.eps')
plt.show()

plt.plot(f_0, v_cntO,  'y-', label='CNT w/ oxygen in epoxide', linewidth=3.0)
plt.plot(f_0, v_0,  'g-', label='CNT w/ carbon in epoxide', linewidth=3.0)
plt.plot(f_0, v_cntN1,  'r-',  label='CNT w/ nitrogen in primary amine', linewidth=3.0)
plt.axhline(y=0.01, color='k', linestyle='--', xmin=0, xmax=0.9033, linewidth=3.0)
plt.axvline(x=0.06355, color='k', linestyle='--', ymin=0, ymax=0.1, linewidth=3.0)
plt.axvline(x=0.066830, color='k', linestyle='--', ymin=0, ymax=0.1, linewidth=3.0)
plt.axvline(x=0.09033, color='k', linestyle='--', ymin=0, ymax=0.1, linewidth=3.0)
plt.xlim([0, 0.1])
plt.ylim([0, 0.1])
plt.legend(loc='upper left')
plt.xlabel('Pull-out force per atom (kcal/mol-$\AA$)')
plt.ylabel('Mean Velocity ($\AA$/fs)')
plt.savefig('fig_extendedfig.eps')
plt.show()

f_CO = np.load('f_0916.npy')
f_ON = np.load('f_ON_1207.npy')
f_OO = np.load('f_OO_1205.npy')
v_ON = np.load('v_ON_1207.npy')
v_OO = np.load('v_OO_1205.npy')

f_00050 = np.load('f_00050.npy')
v_00050 = np.load('v_00050.npy')

f_00004 = np.load('f_00004.npy')
v_00004 = np.load('v_00004.npy')

f_3rr5epo_00031 = np.load('f_3rr5epo_00031.npy')
v_3rr5epo_00031 = np.load('v_3rr5epo_00031.npy')

f_3rr4epo_00023 = np.load('f_3rr4epo_00023.npy')
v_3rr4epo_00023 = np.load('v_3rr4epo_00023.npy')

plt.plot(f_0, v_cntO,  'r-', linewidth=3.0)
#plt.plot(f_0, v_0,  'g-', label='CNT w/ carbon in epoxide', linewidth=3.0)
#plt.plot(f_0, v_cntN1,  'r-',  label='CNT w/ nitrogen in primary amine', linewidth=3.0)
#plt.plot(f_00004, v_00004,  'm-', label='CNT w/ nitrogen in secondary amine', linewidth=3.0)
#plt.plot(f_CO, v_CO,  'k-', label='CNT w/ carbon followed by oxygen', linewidth=3.0)
plt.plot(f_3rr5epo_00031, v_3rr5epo_00031,  'b-', linewidth=3.0)
plt.plot(f_3rr4epo_00023, v_3rr4epo_00023,  'k-',  linewidth=3.0)
#plt.plot(f_00050, v_00050,  'b-', label='5 CNT atoms functionalized (3 C + 2 O)', linewidth=3.0)
plt.axhline(y=0.01, color='k', linestyle='--', xmin=0, xmax=1.0, linewidth=3.0)
#plt.axvline(x=0.06355, color='k', linestyle='--', ymin=0, ymax=0.2157, linewidth=2.0)
#plt.axvline(x=0.066830, color='k', linestyle='--', ymin=0, ymax=0.2157, linewidth=3.0)
plt.axvline(x=0.09033, color='k', linestyle='--', ymin=0, ymax=0.3225, linewidth=3.0)
#plt.axvline(x=0.1125, color='k', linestyle='--', ymin=0, ymax=0.2157, linewidth=2.0)
plt.axvline(x=0.16965, color='k', linestyle='--', ymin=0, ymax=0.3225, linewidth=3.0)
plt.axvline(x=0.293, color='k', linestyle='--', ymin=0, ymax=0.3225, linewidth=3.0)
#plt.axvline(x=0.1463, color='k', linestyle='--', ymin=0, ymax=0.2157, linewidth=3.5)
#plt.axvline(x=0.2714, color='k', linestyle='--', ymin=0, ymax=0.2157, linewidth=2.5)
plt.xlim([0, 0.32])
plt.ylim([-0.001, 0.03])
plt.legend(loc='upper right')
plt.xlabel('Pull-out force per atom (kcal/mol-$\AA$)', fontsize=18)
plt.ylabel('Mean velocity ($\AA$/fs)', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('fig3_pulout.svg', bbox_inches="tight")
#plt.savefig('fig_00050.eps')
plt.show()


#plt.plot(f_0, v_cntO,  'r-', linewidth=3.0)
#plt.plot(f_0, v_0,  'g-', label='CNT w/ carbon in epoxide', linewidth=3.0)
#plt.plot(f_0, v_cntN1,  'r-',  label='CNT w/ nitrogen in primary amine', linewidth=3.0)
#plt.plot(f_00004, v_00004,  'm-', label='CNT w/ nitrogen in secondary amine', linewidth=3.0)
#plt.plot(f_CO, v_CO,  'k-', label='CNT w/ carbon followed by oxygen', linewidth=3.0)
plt.plot(f_3rr5epo_00031, v_3rr5epo_00031,  'k-', linewidth=5.0)
#plt.plot(f_3rr4epo_00023, v_3rr4epo_00023,  'k-',  linewidth=3.0)
#plt.plot(f_00050, v_00050,  'b-', label='5 CNT atoms functionalized (3 C + 2 O)', linewidth=3.0)
#plt.axhline(y=0.01, color='k', linestyle='--', xmin=0, xmax=1.0, linewidth=3.0)
#plt.axvline(x=0.06355, color='k', linestyle='--', ymin=0, ymax=0.2157, linewidth=2.0)
#plt.axvline(x=0.066830, color='k', linestyle='--', ymin=0, ymax=0.2157, linewidth=3.0)
#plt.axvline(x=0.09033, color='k', linestyle='--', ymin=0, ymax=0.3225, linewidth=3.0)
#plt.axvline(x=0.1125, color='k', linestyle='--', ymin=0, ymax=0.2157, linewidth=2.0)
#plt.axvline(x=0.16965, color='k', linestyle='--', ymin=0, ymax=0.3225, linewidth=3.0)
#plt.axvline(x=0.293, color='k', linestyle='--', ymin=0, ymax=0.3225, linewidth=3.0)
#plt.axvline(x=0.1463, color='k', linestyle='--', ymin=0, ymax=0.2157, linewidth=3.5)
#plt.axvline(x=0.2714, color='k', linestyle='--', ymin=0, ymax=0.2157, linewidth=2.5)
plt.xlim([0, 0.20])
plt.ylim([-0.001, 0.03])
plt.xticks(np.arange(0, 0.2, step=0.04))
#plt.legend(loc='upper right')
plt.xlabel('Pull-out force per atom (kcal/mol-$\AA$)', fontsize=18)
plt.ylabel('Mean velocity ($\AA$/fs)', fontsize=18)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig('fig13a_pulout.svg', bbox_inches="tight")
plt.show()

#f_NO_120

plt.plot(f_CO, v_CO,  'k-',  linewidth=3.0,  label='CNT w/ carbon followed by oxygen')
plt.plot(f_OO, v_OO,  'b-',  linewidth=3.0,  label='CNT w/ oxygen followed by oxygen')
plt.plot(f_ON, v_ON,  'r-',  linewidth=3.0,  label='CNT w/ secondary amine nitrogen followed by oxygen')
#plt.axhline(y=0.01, color='k', linestyle='--', xmin=0, xmax=0.807222, linewidth=2.0)
#plt.axvline(x=0.1463, color='k', linestyle='--', ymin=0, ymax=0.099, linewidth=2.0)
plt.xlim([0.05, 0.15])
plt.ylim([-0.001, 0.01])
plt.xlabel('Pull-out force per atom (kcal/mol-$\AA$)')
plt.ylabel('Mean velocity ($\AA$/fs)')
plt.legend(loc='upper left')
plt.savefig('fig_vCO2.png')
plt.show()


plt.plot(f_0, d_0, 'k-')
plt.plot(f_1, d_1, 'r-')
plt.plot(f_2, d_2, 'g-')
plt.plot(f_3, d_3, 'm-')
plt.plot(f_A, d_A, 'b-')
plt.show()


F_c = np.linspace(0.00, 0.08, 9, dtype=float)

v_c = np.asarray([3.709e-4, 8.77e-4, 1.359e-3, 1.859e-3, 2.340e-3, 2.899e-3, 3.780e-3, 4.269e-3, 4.9137e-3])

v_n1 = np.asarray([4e-5, 3.813e-4, 8.70e-4, 1.239e-3, 1.851e-3, 2.217e-3, 2.90e-3, 3.52e-3, 4.10e-3])

F_O = np.linspace(0.00, 0.10, 11, dtype=float)
v_O = np.asarray([4.7e-5, 2.3e-4, 4.2e-4, 1.019e-3, 1.142e-3, 2.037e-3, 2.04e-3, 2.636e-3, 3.0586e-3, 3.340e-3, 3.719e-3])

plt.plot(F_O, v_O, 'yo', markersize=6, linewidth=3.0, label='CNT w/ oxygen in epoxide')
plt.plot(F_c, v_c, 'go', markersize=6, linewidth=3.0, label='CNT w/ carbon in epoxide')
plt.plot(F_c, v_n1, 'ro', markersize=6, linewidth=3.0, label='CNT w/ nitrogen in primary amine')

plt.axhline(y=3.5e-3, color='k', linestyle='--', xmin=0, xmax=1.0, linewidth=1.5)
#plt.axvline(x=0.06355, color='g', linestyle='--', ymin=0, ymax=0.7, linewidth=1.5)

plt.xlabel('Minimum pull-out force per atom in simulation, $F_{min}$ (kcal/mol-$\AA$)')
plt.ylabel('Mean Velocity at the end of 2000 fs, $v\'$ ($\AA$/fs)')
plt.legend(loc='upper left')
plt.xlim([-1e-4, 0.1])
plt.ylim([0, 5e-3])
plt.savefig('fig_2000fs_C.png')
plt.show()


v_1 = np.loadtxt('file1.txt')
v_0 = np.loadtxt('file0.txt')
v_3 = np.loadtxt('file3.txt')

f = np.linspace(0.0, 0.10, 51)

Z = np.vstack((f[0:20], v_3[0:20]))
np.savetxt('Z.txt', np.transpose(Z))

plt.plot(f, v_0, 'go', label='CNT w/ carbon in epoxide')
plt.plot(f, v_1, 'ro', label='CNT w/ nitrogen in amine')
plt.plot(f, v_3, 'yo', label='CNT w/ oxygen in hydroxyl')
plt.axhline(y=3.5e-3, color='k', linestyle='--', xmin=0, xmax=1.0, linewidth=1.5)
plt.axvline(x=0.0580, color='k', linestyle='--', ymin=0, ymax=0.5, linewidth=1.5)
plt.axvline(x=0.0690, color='k', linestyle='--', ymin=0, ymax=0.5, linewidth=1.5)
plt.axvline(x=0.0880, color='k', linestyle='--', ymin=0, ymax=0.5, linewidth=1.5)
plt.xlim([-1e-4, 0.1])
plt.ylim([0, 0.007])
plt.xlabel('Minimum pull-out force per atom in simulation, $F_{min}$ (kcal/mol-$\AA$)')
plt.ylabel('Mean Velocity at the end of 2000 fs, $v\'$ ($\AA$/fs)')
plt.legend(loc='upper left')
plt.savefig('fig_2000fs_all.svg')
plt.savefig('fig_2000fs_all.eps')
plt.show()

###save data for f_next:




#####this block to show
f06 = 0.06 + np.load('f_O_06.npy')
v_O_06 = np.load('v_O_06.npy')
d_O_06 = np.load('d_O_06.npy')

f07 = 0.07 + np.load('f_O_07.npy')
v_O_07 = np.load('v_O_07.npy')

f05 = 0.05 + np.load('f_O_05.npy')
v_O_05 = np.load('v_O_05.npy')

plt.plot(f_0, v_cntO,  'b-', label='f $\in$ [0, 0.10]', linewidth=3.0)
plt.plot(f05, v_O_05,  'r-', label='f $\in$ [0.05, 0.10]', linewidth=3.0)
plt.plot(f06, v_O_06,  'g-', label='f $\in$ [0.06, 0.10]', linewidth=3.0)
plt.plot(f07, v_O_07,  'm-', label='f $\in$ [0.07, 0.10]', linewidth=3.0)

plt.axvline(x=0.09965, color='k', linestyle='--', ymin=0, ymax=0.25, linewidth=3.0)
plt.axvline(x=0.09033, color='k', linestyle='--', ymin=0, ymax=0.25, linewidth=3.0)
plt.axvline(x=0.09550, color='k', linestyle='--', ymin=0, ymax=0.25, linewidth=3.0)
plt.axvline(x=0.07595, color='k', linestyle='--', ymin=0, ymax=0.25, linewidth=3.0)
plt.axhline(y=0.01, color='k', linestyle='--', xmin=0, xmax=0.9965, linewidth=3.0)
plt.xlim([0, 0.10])
plt.ylim([0, 0.04])
plt.legend(loc='upper left')
plt.xlabel('Pull-out force per atom (kcal/mol-$\AA$)')
plt.ylabel('Mean velocity ($\AA$/fs)')
plt.savefig('fig_O_interval.svg')
plt.show()




f_ND = 0.5 + np.load('f_ND_05.npy')
v_ND = np.load('v_ND_05.npy')

plt.plot(f_ND, v_ND, 'k-', linewidth=3.0)
plt.axvline(x=0.6554, color='k', linestyle='--', ymin=0, ymax=0.125, linewidth=3.0)
plt.axhline(y=0.01, color='k', linestyle='--', xmin=0, xmax=0.97125, linewidth=3.0)
plt.xlim([0.5, 0.66])
plt.ylim([0, 0.08])
plt.xlabel('Pull-out force per atom (kcal/mol-$\AA$)')
plt.ylabel('Mean velocity ($\AA$/fs)')
plt.tight_layout()
plt.savefig('fig_ND05.svg')
plt.show()