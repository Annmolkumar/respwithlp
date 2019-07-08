from __future__ import division, absolute_import, print_function

def runPsi4(outdir,prefname,resn='resn',rescharge=0,multiplicity=1,coor=None,lpcoor=None,mem="1000Mb",cpu=4,lot="scf",basis="6-31g*"):
    import pytest
    import sys
    import psi4
    from localresp import resp
    import numpy as np

    psi4.set_num_threads(cpu)
    psi4.set_memory(mem)
    psi4.core.set_output_file("%s.out", False)
    
    xyz='%d %d\n' %(rescharge,multiplicity)
    xyz='    '+xyz
    n = 0
    geoline = ""
    for key in coor.keys():
        if key != "RBI": n = n + 1
        if key[0:2] != "LP" and key[0:1] != "D" and key[0:3] != "RBI":
            geoline  = geoline + key[0:1] + "  " + str(coor[key][0]) + "  " + str(coor[key][1]) + "  " + str(coor[key][2]) + " \n"
    xyz=xyz+geoline
    psi4_xyz="""
    %s
    no_reorient
    no_com
    """%(xyz)
    mol=psi4.geometry(psi4_xyz)
    mol.update_geometry() # This update is required for psi4 to load the molecule
    
    xyz='' 
    n = 0
    geoline = ""
    for key in lpcoor.keys():
        if key[0:2] == "LP": 
           geoline  = geoline + key + "  " + str(lpcoor[key][0]) + "  " + str(lpcoor[key][1]) + "  " + str(lpcoor[key][2]) + " \n"
    xyz=xyz+geoline
    lp="""
    %s
    """%(xyz)
    
    options = {'N_VDW_LAYERS'       : 4,
               'VDW_SCALE_FACTOR'   : 1.4,
               'VDW_INCREMENT'      : 0.2,
               'VDW_POINT_DENSITY'  : 1.0,
               'resp_a'             : 0.0005,
               'RESP_B'             : 0.1,
               'LPCOOR'             : lp,
               'BASIS_ESP'          : basis,
               'psi4_options'       : {'df_basis_scf':'def2-tzvpp-jkfit','df_basis_mp2':'def2-tzvppd-ri', 'maxiter':250},
               'METHOD_ESP'         : lot,
               'RADIUS'             : {'BR':1.97,'I':2.19}
               }
    
    # Call for first stage fit
    charges1 = resp.resp([mol], [options])
    print('Electrostatic Potential Charges')
    print(charges1[0][0])
    print('Restrained Electrostatic Potential Charges')
    print(charges1[0][1])
    
    # Call for second stage fit
    newrest = []
    for i in range(len(charges1[0][1])):
        if i < 5:
           newrest.append(charges1[0][1][i])
        elif i == 5:
           newrest.append(0.0)
        elif i == 6:
           newrest.append(0.0)
        elif i == 7:
           newrest.append(charges1[0][1][i])
        elif i == 8 or i == 9:
           char = charges1[0][1][5]/2.0
           newrest.append(charges1[0][1][i]+char)
        elif i == 10 or i == 11:
           char = charges1[0][1][6]/2.0
           newrest.append(charges1[0][1][i]+char)
    print(newrest)
    print(sum(newrest))
   # sys.exit()
    stage2=resp.stage2_helper()
    #stage2.set_stage2_constraint(mol,charges1[0][1],options,cutoff=1.2)
    stage2.set_stage2_constraint(mol,newrest,options,cutoff=1.2)
    options['resp_a'] = 0.001
    options['grid'] = '1_%s_grid.dat' %mol.name()
    options['esp'] = '1_%s_grid_esp.dat' %mol.name()
    mol.set_name('stage2')
    if options.get('constraint_group')==[]:
       print('Stage1 equals Stage2')
    else:
       charges2 = resp.resp([mol], [options])
       print('RESP Charges')
       print(charges2[0][1])

