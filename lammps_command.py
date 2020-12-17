from lammps import PyLammps, lammps
import numpy as np
import mpi4py


#def run_lammps(filename):
#filename = "/home/aowabin/PycharmProjects/MDSim01/in.tension"
filename = "/home/aowabin/PycharmProjects/MDSim01/in.adap.txt"
#filename="/home/aowabin/PycharmProjects/MDSim01/in.nanowire_mini.txt"

lmp = lammps()
lmp.command('variable id string 1epo')
lmp.command('variable dir equal 1')
lmp.file(filename)
lmp.close()

#return None