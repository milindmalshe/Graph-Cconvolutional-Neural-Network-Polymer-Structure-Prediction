import numpy as np
import hotspot_file_01




#file_list = './MDfiles/select02/3rr1epo.00050'
file_list = ['./MDfiles/select03/3rr1epo.*', './MDfiles/select03/3rr2epo.*', './MDfiles/select03/3rr5epo.*',
             './MDfiles/select03/3rr3epo.*', './MDfiles/select03/3rr4epo.*'] #list of all the functionalized files
forcefile = ['./MDfiles/select03/f_3rr1epo.csv', './MDfiles/select03/f_3rr2epo.csv', './MDfiles/select03/f_3rr5epo.csv',
             './MDfiles/select03/f_3rr3epo.csv', './MDfiles/select03/f_3rr4epo.csv'] #list of the forcefiles

#clusterfile_list = ['./MDfiles/select03/cluster_analysis/chain1epo.*', './MDfiles/select03/cluster_analysis/chain2epo.*',
#             './MDfiles/select03/cluster_analysis/chain5epo.*', './MDfiles/select03/cluster_analysis/c3epo.*', './MDfiles/select03/cluster_analysis/chain4epo.*']

clusterfile_list = ['./MDfiles/select03/chain_analysis/b1epo.*', './MDfiles/select03/chain_analysis/b2epo.*',
             './MDfiles/select03/chain_analysis/b5epo.*', './MDfiles/select03/chain_analysis/b3epo.*', './MDfiles/select03/chain_analysis/b4epo.*'] #list of clusterfile_list


MD_models = hotspot_file_01.MD_models(list_of_files=file_list, forcefile=forcefile, str='vz', cluster_file=clusterfile_list)

