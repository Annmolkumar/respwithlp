class Psi4Writer:
    Psi4header='''#!/usr/bin/env python
from __future__ import division, absolute_import, print_function
import pytest
import sys
import psi4
from localresp import resp
import numpy as np
mem="%s"
cpu=%s
psi4.set_num_threads(cpu)
psi4.set_memory(mem)
psi4.core.set_output_file("%s.out", False)

'''

    Psi4moltmpl='''
psi4_xyz="""
%s
no_reorient
no_com
"""
mol=psi4.geometry(psi4_xyz)
mol.update_geometry() # This update is required for psi4 to load the molecule
'''

    Psi4lptmpl='''
lp="""
%s
"""
'''

    Psi4footer='''

options = {'N_VDW_LAYERS'       : 4,
           'VDW_SCALE_FACTOR'   : 1.4,
           'VDW_INCREMENT'      : 0.2,
           'VDW_POINT_DENSITY'  : 1.0,
           'METHOD_ESP'         : "%s",
           'BASIS_ESP'          : "%s",
           'resp_a'             : 0.0005,
           'RESP_B'             : 0.1,
           'LPCOOR'             : lp
           }

# Call for first stage fit
charges1 = resp.resp([mol], [options])
print('Electrostatic Potential Charges')
print(charges1[0][0])
print('Restrained Electrostatic Potential Charges')
print(charges1[0][1])

'''
    
    def __init__(self,outdir,prefname,resn='resn',rescharge=0,multiplicity=1,coor=None,lpcoor=None,mem="1000Mb",cpu=4,lot="scf",basis="6-31g*"):
        self.resn=resn
        self.reschrg=rescharge
        self.multiplicity=multiplicity
        runfn = outdir+"/"+prefname 
        Psi4header=self.Psi4header %(mem,cpu,runfn)
        Psi4footer=self.Psi4footer %(lot,basis)
        runf=open(runfn+".py","w")
        xyz=self.Psi4xyz(coor,self.reschrg,self.multiplicity)
        lp=self.Psi4lpxyz(lpcoor)
        runf.write(Psi4header)
        runf.write(xyz)
        runf.write(lp)
        runf.write(Psi4footer)
        runf.write('\n')
        runf.close()
        return 
        
    def Psi4xyz(self,coor,reschrg,multiplicity):
        xyz='%d %d\n' %(reschrg,multiplicity)
        xyz='    '+xyz
        n = 0
        geoline = ""
        for key in coor.keys():
            if key != "RBI": n = n + 1
            if key[0:2] != "LP" and key[0:1] != "D" and key[0:3] != "RBI":
               geoline  = geoline + key[0:1] + "  " + str(coor[key][0]) + "  " + str(coor[key][1]) + "  " + str(coor[key][2]) + " \n"
        xyz=xyz+geoline
        moltmpl=self.Psi4moltmpl %(xyz)
        return(moltmpl)

    def Psi4lpxyz(self,lpcoor):
        xyz='' 
        n = 0
        geoline = ""
        for key in lpcoor.keys():
            if key[0:2] == "LP": 
               geoline  = geoline + key + "  " + str(lpcoor[key][0]) + "  " + str(lpcoor[key][1]) + "  " + str(lpcoor[key][2]) + " \n"
        xyz=xyz+geoline
        moltmpl=self.Psi4lptmpl %(xyz)
        return(moltmpl)

