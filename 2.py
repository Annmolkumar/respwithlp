""")

    mol.update_geometry()

    options = {'N_VDW_LAYERS'       : 4,
               'VDW_SCALE_FACTOR'   : 1.4,
               'VDW_INCREMENT'      : 0.2,
               'VDW_POINT_DENSITY'  : 20.0,
               'resp_a'             : 0.0005,
               'RESP_B'             : 0.1,
               'BASIS_ESP':'Sadlej',
               'psi4_options':{'df_basis_scf':'def2-tzvpp-jkfit','df_basis_mp2':'def2-tzvppd-ri', 'maxiter':250},
               'METHOD_ESP':'mp2',
               'RADIUS':{'BR':1.97,'I':2.19}
    }

    # Call for first stage fit
    mol.set_name('stage1')
    charges1 = resp.resp([mol], [options])
    print('Electrostatic Potential Charges')
    print(charges1[0][0])
    print('Restrained Electrostatic Potential Charges')
    print(charges1[0][1])

    # Call for second stage fit
    stage2=resp.stage2_helper()
    stage2.set_stage2_constraint(mol,charges1[0][1],options,cutoff=1.2)
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
    
    

calculate()
