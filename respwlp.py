from __future__ import division, absolute_import, print_function
import sys
import os
import math
import numpy as np
import argparse
from collections import OrderedDict

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        rsarg = []
        with values as f:
            argline = f.read()
            arglist = argline.split()
      
        parser.parse_args(nrsarg, namespace)


def runPsi4(outdir,prefname,resn='resn',rescharge=0,multiplicity=1,coor=None,lpcoor=None,lplist=None,mem="1000Mb",cpu=4,lot="scf",basis="6-31g*"):
    import pytest
    import sys
    import psi4
    from localresp import resp
    import numpy as np
    from collections import OrderedDict

    psi4.set_num_threads(cpu)
    psi4.set_memory(mem)
    psi4.core.set_output_file(outdir+"/"+prefname+".out", False)
    
    xyz='%s %s\n' %(rescharge,multiplicity)
    xyz='    '+xyz
    n = 0
    geoline = ""
    for key in coor:
        if key[0] != "RBI": n = n + 1
        if key[0][0:2] != "LP" and key[0][0:1] != "D" and key[0][0:3] != "RBI":
            geoline  = geoline + key[0][0:1] + "  " + str(key[1]) + "  " + str(key[2]) + "  " + str(key[3]) + " \n"
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
    for key in lpcoor:
        if key[0][0:2] == "LP": 
           geoline  = geoline + key[0] + "  " + str(key[1]) + "  " + str(key[2]) + "  " + str(key[3]) + " \n"
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
    
    mol.set_name('stage1')
    # Call for first stage fit
    charges1 = resp.resp([mol], [options])
    #print('Electrostatic Potential Charges')
    #print(charges1[0][0])
    #print('Restrained Electrostatic Potential Charges')
    #print(charges1[0][1])
    
    allcoor = coor + lpcoor
    respchar = OrderedDict()
    for i in range(len(allcoor)):
        respchar[allcoor[i][0]]  = charges1[0][1][i]
    
    for key in list(respchar.keys()):
        if key in list(lplist.keys()):
           tobedist = respchar[key]/len(lplist[key])
           respchar[key] = 0.0
           for val in lplist[key]:
               respchar[val] = respchar[val] + tobedist
    #print (list(respchar.values()))       
    # Call for second stage fit
    stage2=resp.stage2_helper()
    stage2.set_stage2_constraint(mol,list(respchar.values()),options,cutoff=1.2)
    options['resp_a'] = 0.001
    options['grid'] = '1_%s_grid.dat' %mol.name()
    options['esp'] = '1_%s_grid_esp.dat' %mol.name()
    fout = open(outdir+"/"+"resp.dat","w")
    #if options.get('constraint_group')==[]:
    #   fout.write('Stage1 equals Stage2')
    #else:
    charges2 = resp.resp([mol], [options])
    for i in range(len(allcoor)):
        respchar[allcoor[i][0]]  = charges2[0][1][i]
    for key,value in respchar.items():
        fout.write("%s  %7.3f \n"%(key, value))
    os.remove('1_%s_grid.dat' %mol.name()) 
    os.remove('1_%s_grid_esp.dat' %mol.name()) 

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  radi = math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
  return math.degrees(radi)
  
def colinear(ar,scale,hcoor=[]):
    pos = [] 
    if len(hcoor) != 2:
       return

    r_a = list(map(float,hcoor[0][0:3]))
    r_b = list(map(float,hcoor[1][0:3]))

    slope = [r_ai - r_bi for r_ai, r_bi in zip(r_a, r_b)] 
    slope = slope / np.linalg.norm(slope)
    new_x = r_a[0] - ar*scale*slope[0]
    new_y = r_a[1] - ar*scale*slope[1]
    new_z = r_a[2] - ar*scale*slope[2]
    pos = [new_x,new_y,new_z] 
    return pos   

def relative(ar,ang,dih,hcoor=[]):
    pos = [] 
    if len(hcoor) != 3:
       return
    r_i = list(map(float,hcoor[0][0:3]))
    a_i = list(map(float,hcoor[1][0:3]))
    d_i = list(map(float,hcoor[2][0:3]))

    ar = ar
    ang = ang * np.pi / 180.0
    dih = dih * np.pi / 180.0
    
    sinTheta = np.sin(ang)
    cosTheta = np.cos(ang)
    sinPhi = np.sin(dih)
    cosPhi = np.cos(dih)

    ax = ar * cosTheta
    ay = ar * cosPhi * sinTheta
    az = ar * sinPhi * sinTheta
    
    ad = [a_ii - d_ii for a_ii, d_ii in zip(a_i, d_i)] 
    ra = [r_ii - a_ii for r_ii, a_ii in zip(r_i, a_i)] 
    ra = ra / np.linalg.norm(ra)
    nv = np.cross(ad, ra)
    nv = nv / np.linalg.norm(nv)
    ncra = np.cross(nv, ra)
    
    new_x = r_i[0] - ra[0] * ax + ncra[0] * ay + nv[0] * az
    new_y = r_i[1] - ra[1] * ax + ncra[1] * ay + nv[1] * az
    new_z = r_i[2] - ra[2] * ax + ncra[2] * ay + nv[2] * az
    pos = [new_x,new_y,new_z] 
    return pos   


def readpdbpsf(pdbname,psfname):
   bndlst = {}
   anam = []
   pos = []
   filein = open(pdbname, "r") 
   for d in filein:
       if d[:4] == 'ATOM' or d[:6] == "HETATM":
          splitted_line = [d[:6], d[6:11], d[12:16], d[17:21], d[21], d[22:26], d[30:38], d[38:46], d[46:54]]
          resi = splitted_line[3]
          anam.append(splitted_line[2].strip()) 
          pos.append(splitted_line[6:9])
   filein.close()
   filein = open(psfname, "r") 
   data = [line.split() for line in filein]
   filein.close()
   readbond = False
   for ind,field in enumerate(data):
       if len(field) > 2 and field[1] == "!NBOND:":
          readbond = True
          strtread = ind + 1
       if len(field) > 2 and field[1] == "!NTHETA:":
          readbond = False
          break
       if readbond and ind >= strtread:
          for i in range(0,len(field),2):
              if 'LP' not in [anam[int(field[i])-1][:2], anam[int(field[i+1])-1][:2]] and 'D' not in [anam[int(field[i])-1][:1], anam[int(field[i+1])-1][:1]]:
                  try:
                      bndlst[anam[int(field[i])-1]].append(anam[int(field[i+1])-1])
                  except KeyError:
                      bndlst[anam[int(field[i])-1]]=[anam[int(field[i+1])-1]]
                  try:
                      bndlst[anam[int(field[i+1])-1]].append(anam[int(field[i])-1])
                  except KeyError:
                      bndlst[anam[int(field[i+1])-1]]=[anam[int(field[i])-1]]

   return (resi, anam, pos, bndlst) 

def readmol2(mol2name):
   foundatm=False
   foundbnd=False
   bndlst = OrderedDict()
   anam = []
   pos = []
   with open(mol2name,"r") as filein:
     for ind,line in enumerate(filein):
        field = line.split()  
        if len(field) != 0 and not foundatm :
           if field[0] == "@<TRIPOS>ATOM":
              foundatm=True
              foundbnd=False
              anam = []
              pos = []
              pass
        elif len(field) != 0 and not foundbnd:
            if field[0] == "@<TRIPOS>BOND":
               foundbnd = True
               foundatm = False
        if foundatm and len(field) >= 4:
           try:
              anam.append(field[1]) 
              pos.append(field[2:5])
           except (IndexError,ValueError):
              foundatm = False
        if foundbnd and len(field) == 4:
           try:
              bndlst[anam[int(field[1])-1]].append(anam[int(field[2])-1])
           except KeyError:
              bndlst[anam[int(field[1])-1]]=[anam[int(field[2])-1]]
           try:
              bndlst[anam[int(field[2])-1]].append(anam[int(field[1])-1])
           except KeyError:
              bndlst[anam[int(field[2])-1]]=[anam[int(field[1])-1]]
   filein.close()
   return ("resi", anam, pos, bndlst)       

def createlonepair(anam,pos,bndlst):
    natoms = len(anam)
    lpbndlst = OrderedDict()
    lpbnddict = OrderedDict()
    nlpbndlst = OrderedDict()
    cor = OrderedDict()        
    predlpcor = []       
    corlist = []
    natmwolp = 0
    for i in range (natoms):
        cor[anam[i]] = [float(pos[i][0]),float(pos[i][1]),float(pos[i][2])]
        corlist.append([anam[i],float(pos[i][0]),float(pos[i][1]),float(pos[i][2])])
        if anam[i][0:2] != "LP" and anam[i][0:1] != "D" and anam[i][0:3] != "RBI":
            natmwolp = natmwolp + 1 

    vect = {}
    v1 = []
    v2 = []
    v3 = []
    vc = []
    vi = []
    n = 0
    for i in range(0, len(anam)):
        if anam[i][0:1] == "O" and len(bndlst.get(anam[i])) == 1 or anam[i][0:1] == "S" and len(bndlst.get(anam[i])) == 1:
              ata = bndlst.get(anam[i])[0]
              if ata[0:1] == "P": pass
              atb = ""
              if len(bndlst.get(ata)) > 1:
                    for atms in bndlst.get(ata): 
                        if anam[i] != atms: 
                           if atms[0:1] != "H":
                              atb = atms 
                              break
                    if atb == "": 
                       for atms in bndlst.get(ata): 
                           if anam[i] != atms: 
                              atb = atms 
                              break

              n = n+1
              lpname = "LP"+str(n) 
              lpbndlst[anam[i]] = [lpname]
              lpbnddict[lpname] = [anam[i]]
              nlpbndlst[i] = [n + natmwolp - 1]
              hcoor = [cor[anam[i]],cor[ata],cor[atb]]
              if anam[i][0:1] == "O": 
                 ##predlpcor[lpname] = relative(0.35,110.0,90.0,hcoor) # Out of plane lone pairs
                 #predlpcor[lpname] = relative(0.35,110.0,0.0,hcoor)
                 predlpcor.append([lpname] + relative(0.35,110.0,0.0,hcoor))
              if anam[i][0:1] == "S": 
                 ##predlpcor[lpname] = relative(0.75,95.0,260.0,hcoor)
                 #predlpcor[lpname] = relative(0.75,95.0,0.0,hcoor)
                 predlpcor.append([lpname]+relative(0.75,95.0,0.0,hcoor))
    
              n = n+1
              lpname = "LP"+str(n) 
              lpbndlst[anam[i]].append(lpname)
              lpbnddict[lpname] = [anam[i]]
              nlpbndlst[i].append(n + natmwolp - 1)
           #  hcoor = [cor[anam[i]],cor[atb],cor[ata]] # Order flipped 
              if anam[i][0:1] == "O": 
                 ##predlpcor[lpname] = relative(0.35,110.0,90.0,hcoor)
                 #predlpcor[lpname] = relative(0.35,110.0,180.0,hcoor)
                 predlpcor.append([lpname]+relative(0.35,110.0,180.0,hcoor))
              if anam[i][0:1] == "S": 
                 ##predlpcor[lpname] = relative(0.75,95.0,260.0,hcoor)
                 #predlpcor[lpname] = relative(0.75,95.0,180.0,hcoor)
                 predlpcor.append([lpname]+relative(0.75,95.0,180.0,hcoor))
    
        if anam[i][0:1] == "N" and len(bndlst.get(anam[i])) == 1:
              slope = []
              ata = bndlst.get(anam[i])[0]
              hcoor = [cor[anam[i]],cor[ata]]
              n = n+1
              lpname = "LP"+str(n) 
              lpbndlst[anam[i]] = [lpname]
              lpbnddict[lpname] = [anam[i]]
              nlpbndlst[i] = [n + natmwolp - 1]
              #predlpcor[lpname] = colinear(0.35,1.00,hcoor)
              predlpcor.append([lpname]+colinear(0.35,1.00,hcoor))
              
        if anam[i][0:1] == "O" and len(bndlst.get(anam[i])) == 2: 
              # lonepair bisector
              ata = bndlst.get(anam[i])[0]
              atb = bndlst.get(anam[i])[1]
              cor["RBI"] = [(ca-cb)/2.0 for ca,cb in zip(cor[ata],cor[atb])]
              cor["RBI"] = [ca+cb for ca,cb in zip(cor[atb],cor["RBI"])]
              #v_anamata = [a_i - b_i for a_i, b_i in zip(cor[ata],cor[anam[i]]) 
              #v_anamatb = [a_i - b_i for a_i, b_i in zip(cor[atb],cor[anam[i]]) 
              #bisec = abs(angle(v_anamata,v_anamatb))
               
              hcoor = [cor[anam[i]],cor["RBI"],cor[atb]]
              n = n+1
              lpname = "LP"+str(n) 
              lpbndlst[anam[i]] = [lpname]
              lpbnddict[lpname] = [anam[i]]
              nlpbndlst[i] = [n + natmwolp - 1]
              #predlpcor[lpname] = relative(0.35,110.0,90.0,hcoor)
              predlpcor.append([lpname]+relative(0.35,110.0,90.0,hcoor))
    
              n = n+1
              lpname = "LP"+str(n) 
              lpbndlst[anam[i]].append(lpname)
              lpbnddict[lpname] = [anam[i]]
              nlpbndlst[i].append(n + natmwolp - 1)
              #predlpcor[lpname] = relative(0.35,110.0,270.0,hcoor)
              predlpcor.append([lpname]+relative(0.35,110.0,270.0,hcoor))
    
        if anam[i][0:1] == "S" and len(bndlst.get(anam[i])) == 2:
              # lonepair bisector
              ata = bndlst.get(anam[i])[0]
              atb = bndlst.get(anam[i])[1]
              cor["RBI"] = [(ca-cb)/2.0 for ca,cb in zip(cor[ata],cor[atb])]
              cor["RBI"] = [ca+cb for ca,cb in zip(cor[atb],cor["RBI"])]
    
              hcoor = [cor[anam[i]],cor["RBI"],cor[atb]]
              n = n+1
              lpname = "LP"+str(n) 
              lpbndlst[anam[i]] = [lpname]
              lpbnddict[lpname] = [anam[i]]
              nlpbndlst[i] = [n + natmwolp - 1]
              #predlpcor[lpname] = relative(0.70,95.0,100.0,hcoor)
              predlpcor.append([lpname]+relative(0.70,95.0,100.0,hcoor))
    
              n = n+1
              lpname = "LP"+str(n) 
              lpbndlst[anam[i]].append(lpname)
              lpbnddict[lpname] = [anam[i]]
              nlpbndlst[i].append(n + natmwolp - 1)
              hcoor = [cor[anam[i]],cor[atb],cor["RBI"]]  # Flipped order
              #predlpcor[lpname] = relative(0.70,95.0,100.0,hcoor)
              predlpcor.append([lpname]+relative(0.70,95.0,100.0,hcoor))
                
        if anam[i][0:1] == "N" and len(bndlst.get(anam[i])) == 2 or anam[i][0:1] == "P" and len(bndlst.get(anam[i])) == 2:
               # lonepair bisector
              ata = bndlst.get(anam[i])[0]
              atb = bndlst.get(anam[i])[1]
              cor["RBI"] = [(ca-cb)/2.0 for ca,cb in zip(cor[ata],cor[atb])]
              cor["RBI"] = [ca+cb for ca,cb in zip(cor[atb],cor["RBI"])]
              n = n+1
              lpname = "LP"+str(n) 
              lpbndlst[anam[i]] = [lpname]
              lpbnddict[lpname] = [anam[i]]
              nlpbndlst[i] = [n + natmwolp - 1 ]
              hcoor = [cor[anam[i]],cor["RBI"],cor[atb]]
              if anam[i][0:1] == "N": 
                 #predlpcor[lpname] = relative(0.30,180.0,180.0,hcoor)
                 predlpcor.append([lpname]+relative(0.30,180.0,180.0,hcoor))
              if anam[i][0:1] == "P": 
                 #predlpcor[lpname] = relative(0.70,180.0,180.0,hcoor)
                 predlpcor.append([lpname]+relative(0.70,180.0,180.0,hcoor))
    
        if anam[i][0:1] == "N" and len(bndlst.get(anam[i])) == 3 or anam[i][0:1] == "P" and len(bndlst.get(anam[i])) == 3:
              ata = bndlst.get(anam[i])[0]
              atb = bndlst.get(anam[i])[1]
              atc = bndlst.get(anam[i])[2]
              v1.append(cor.get(ata)[0] - cor.get(atb)[0])
              v1.append(cor.get(ata)[1] - cor.get(atb)[1])
              v1.append(cor.get(ata)[2] - cor.get(atb)[2])
              v2.append(cor.get(ata)[0] - cor.get(atc)[0])
              v2.append(cor.get(ata)[1] - cor.get(atc)[1])
              v2.append(cor.get(ata)[2] - cor.get(atc)[2])
              vc = np.cross(v1,v2)
              v3.append(cor.get(anam[i])[0] - cor.get(ata)[0])
              v3.append(cor.get(anam[i])[1] - cor.get(ata)[1])
              v3.append(cor.get(anam[i])[2] - cor.get(ata)[2])
              poav = abs(angle(vc,v3))
              if poav <= 90.0:
                 poav = 180.0 - poav
              v1 = []
              v2 = [] 
              v3 = []
              v1.append(cor.get(anam[i])[0] - cor.get(ata)[0])
              v1.append(cor.get(anam[i])[1] - cor.get(ata)[1])
              v1.append(cor.get(anam[i])[2] - cor.get(ata)[2])
              v2.append(cor.get(anam[i])[0] - cor.get(atb)[0])
              v2.append(cor.get(anam[i])[1] - cor.get(atb)[1])
              v2.append(cor.get(anam[i])[2] - cor.get(atb)[2])
              vi = np.cross(v1,v2)
              impr = abs(angle(vi,vc))
              if impr < 90.0:
                 impr = 90.0 - impr
              elif impr > 90.0:
                 impr = 180.0 - impr
                 impr = 90.0 - impr
              v1 = []
              v2 = []
              v3 = []
              if poav > 100.0:
                 hcoor = [cor[anam[i]],cor[ata],cor[atb]]
                 n = n+1
                 lpname = "LP"+str(n) 
                 lpbndlst[anam[i]] = [lpname]
                 lpbnddict[lpname] = [anam[i]]
                 nlpbndlst[i] = [n + natmwolp - 1]
                 if anam[i][0:1] == "N": 
                    #predlpcor[lpname] = relative(0.30,poav,impr,hcoor)
                    predlpcor.append([lpname]+relative(0.30,poav,impr,hcoor))
                 if anam[i][0:1] == "P": 
                    #predlpcor[lpname] = relative(0.70,poav,impr,hcoor)
                    predlpcor.append([lpname]+relative(0.70,poav,impr,hcoor))
    return (corlist,predlpcor,lpbndlst,nlpbndlst,lpbnddict)

def printxyz(outdir,prefname,cor,predlpcor):
    f = open(outdir+"/"+prefname,"w") 
    n = 0
    for key in cor:
        if key[0] != "RBI": n = n + 1
        if key[0][0:2] != "LP" and key[0][0:1] != "D" and key[0][0:3] != "RBI":
           f.write("{:4s}   {:8.3f} {:8.3f} {:8.3f}\n".format(key[0],key[1],key[2],key[2]))
    for key in predlpcor:
        n = n + 1
        f.write("{:4s}   {:8.3f} {:8.3f} {:8.3f}\n".format(key[0],key[1],key[2],key[3]))
    f.close() 


def printpdb(outdir,prefname,resi,cor,predlpcor):
    f = open(outdir+"/"+prefname,"w") 
    n = 0
    for key in cor:
        if key[0] != "RBI": n = n + 1
        if key[0][0:2] != "LP" and key[0][0:1] != "D" and key[0][0:3] != "RBI":
           f.write("{:6s}{:5d} {:^4s}{:4s}  {:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:4s}\n".format('ATOM',n,key[0],resi,1,key[1],key[2],key[3],0.0,0.0,resi))
    for key in predlpcor:
        n = n + 1
        f.write("{:6s}{:5d} {:^4s}{:4s}  {:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:4s}\n".format('ATOM',n,key[0],resi,1,key[1],key[2],key[3],0.0,0.0,resi))
    f.write("{:6s}{:5d}     {:4s}  {:4d}\n".format('TER',n+1,resi,1))
    f.write("END")
    f.close() 

def printallele(outdir,prefname,cor,predlpcor):
    f = open(outdir+"/"+prefname,"w") 
    for key in cor:
        if key[0][0:2] != "LP" and key[0][0:1] != "D" and key[0][0:3] != "RBI":
           f.write("{:4s}\n".format(key[0])) 
    for key in predlpcor:
        f.write("{:4s}\n".format(key[0]))
    f.close() 

def findbonds(coorv,lpcoorv,bndlstv,lplist,lpdict):
    listallbonds = {**bndlstv,**lplist}
    listallbonds = {**listallbonds,**lpdict}
    #print ("bonds",listallbonds)
    atomlist = []
    for k in coorv:
        atomlist.append(k[0]) 
    for k in lpcoorv:
        atomlist.append(k[0]) 
    bonds = []
    bondnum = []
    for key,value in list(listallbonds.items()):
        if key[0:2] == "LP":
          for val in value:
            if [val,key] not in bonds:
                bonds.append([key,val])
                bondnum.append([atomlist.index(key),atomlist.index(val)])
                #Issue1 Requires an exception so that if an atom is not defined but a bond exists it fails. But it should have failed above already!!!
    return(bonds,bondnum)

def printlpbnd(outdir,prefname,bndlist):
    f = open(outdir+"/"+prefname,"w") 
    for key in bndlist:
        f.write("%s  %s\n"%(key[0],key[1]))
    f.close() 
    
def findangles(coorv,lpcoorv,bndlstv,lplist,lpdict):
    listallbonds = {**bndlstv,**lplist}
    listallbonds = {**listallbonds,**lpdict}
    #print ("angles",listallbonds)
    atomlist = []
    for k in coorv:
        atomlist.append(k[0]) 
    for k in lpcoorv:
        atomlist.append(k[0]) 
    angles = []
    anglenum = []
    for k0,v0 in list(listallbonds.items()):
        if len(v0) > 1:
           for i in range(0,(len(v0)-1)):
               ang2 = k0
               ang1 = v0[i]
               for j in range(i+1,len(v0)):
                   ang3 = v0[j]
                   angles.append([ang1,ang2,ang3])
                   anglenum.append([atomlist.index(ang1),atomlist.index(ang2),atomlist.index(ang3)])
                   #anglenum.append([atomlist[ang1],atomlist[ang2],atomlist[ang3]])
    #print (angles, anglenum)               
    return(angles,anglenum)

def printlpang(outdir,prefname,anglist):
    f = open(outdir+"/"+prefname,"w") 
    for key in anglist:
        f.write("%s  %s\n"%(key[0],key[2]))
    f.close() 

def finddihedrals(coorv,lpcoorv,bndlstv,lplist,lpdict):
    listallbonds = {**bndlstv,**lplist}
    #print ("before",listallbonds)   
    listallbonds = {**listallbonds,**lpdict}
    atomlist = []
    for k in coorv:
        atomlist.append(k[0]) 
    for k in lpcoorv:
        atomlist.append(k[0]) 
    #print ("after",listallbonds)   
    dihedrals = []
    dihedralnum = []
    for k0,v0 in list(listallbonds.items()):
        dih0 = k0
        for k1 in v0:
            if len(listallbonds[k1]) != 1:
               dih1 = k1
               for k2 in listallbonds[k1]:
                     if len(listallbonds[k2]) != 1 and k2 != dih0:
                         dih2 = k2
                         for k3 in listallbonds[k2]:
                             #if len(listofbonds[k3]) != 1 and k3 != dih1 and k3 != dih0:
                             if k3 != dih1 and k3 != dih0:
                                dih3 = k3
                                if [dih3,dih2,dih1,dih0] not in dihedrals:
                                   dihedrals.append([dih0,dih1,dih2,dih3])
                                   dihedralnum.append([atomlist.index(dih0),atomlist.index(dih1),atomlist.index(dih2),atomlist.index(dih3)])
                                  # self.dihedralnum.append([atomlist[dih0],atomlist[dih1],atomlist[dih2],atomlist[dih3]])
    return(dihedrals,dihedralnum)

def printlpdih(outdir,prefname,dihlist):
    f = open(outdir+"/"+prefname,"w") 
    for key in dihlist:
        f.write("%s  %s\n"%(key[0],key[3]))
    f.close() 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mol2","--mol2file",help="Required Argument: Provide Mol2 File")
    parser.add_argument("-pdb","--pdbfile",help="Required Argument: Provide Pdb file") 
    parser.add_argument("-psf","--psffile",help="Required Argument: Pdb file requires psf file")
    parser.add_argument("-dir","--workdir",type=str,default=".",help="Enter were the output files will be saved")
    parser.add_argument("-mem","--memory",type=str,default="1000Mb",help="Memory")
    parser.add_argument("-cpu","--nthreads",type=int,default=4,help="Number of threads")
    parser.add_argument("-lot","--theory",type=str,default="scf",help="Enter level of theory")
    parser.add_argument("-basis","--basis",type=str,default="6-31g*",help="Basis set")
    parser.add_argument("-c","--charge",type=int,default=0,help="Charge of the molecule, default is 0")
    parser.add_argument("-m","--multiplicity",type=int,default=1,help="Multiplicity of the molecule, default is 1")
    parser.add_argument("-f","--file",type=open,action=LoadFromFile)
    args = parser.parse_args()
    
    if args.mol2file is None: 
       parser.print_help()
       sys.exit()
    if args.pdbfile is not None and args.psffile is None:
       parser.print_help()
       sys.exit()
    if args.mol2file:
       mol2name = args.mol2file
       prefname = os.path.basename(mol2name)
       psi4name = prefname.strip().split(".")[0]
       resname, anamv, posv, bndlstv = readmol2(mol2name)
    if args.pdbfile and args.psffile:
       pdbname = args.pdbfile
       psfname = args.psffile
       prefname = os.path.basename(pdbname)
       psi4name = prefname.strip().split(".")[0]
       resname, anamv, posv, bndlstv = readpdbpsf(pdbname,psfname)
     
    coorv,lpcoorv,lplist,nlplist,lpdict = createlonepair(anamv, posv, bndlstv)
    
    if not args.workdir:
       outdir = "."
    else:
       outdir = args.workdir
    if not os.path.exists(outdir):
       os.mkdir(outdir)
    
    nbndwolp = len(bndlstv)
    printallele(outdir,"element_anm.dat",coorv,lpcoorv) 
    bnds,bndn = findbonds(coorv,lpcoorv,bndlstv,lplist,lpdict) 
    printlpbnd(outdir,"lp.bonds",bndn)
    angs,angn = findangles(coorv,lpcoorv,bndlstv,lplist,lpdict) 
    printlpang(outdir,"allwlp.angles",angn)
    dihs,dihn= finddihedrals(coorv,lpcoorv,bndlstv,lplist,lpdict) 
    printlpdih(outdir,"allwlp.dihedrals",dihn)
    #printpdb(outdir,prefname,resname,coorv,lpcoorv)
    runPsi4(outdir,psi4name,resn=resname,rescharge=args.charge,multiplicity=args.multiplicity,coor=coorv,lpcoor=lpcoorv,lplist=lplist,mem=args.memory,cpu=args.nthreads,lot=args.theory,basis=args.basis)
    
    
if __name__ == "__main__":
   main()




