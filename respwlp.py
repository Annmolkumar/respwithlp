from __future__ import division, absolute_import, print_function
import sys
import os
import math
import numpy as np
import argparse
from collections import OrderedDict

atnumas = {'LP': ('0', '0'), 'H': ('1', '1'), 'D': ('1', '2'), 'T': ('1', '3'), 'He': ('2', '4'), 'Li': ('3', '7'), 'Be': ('4', '9'), 'B': ('5', '11'), 
'C': ('6', '14'), 'N': ('7', '15'), 'O': ('8', '18'), 'F': ('9', '19'), 'Ne': ('10', '22'), 'Na': ('11', '23'), 'Mg': ('12', '26'), 'Al': ('13', '27'),
'Si': ('14', '30'), 'P': ('15', '31'), 'S': ('16', '36'), 'Cl': ('17', '37'), 'Ar': ('18', '40'), 'K': ('19', '41'), 'Ca': ('20', '48'), 'Sc': ('21', '45'),
'Ti': ('22', '50'), 'V': ('23', '51'), 'Cr': ('24', '54'), 'Mn': ('25', '55'), 'Fe': ('26', '58'), 'Co': ('27', '59'), 'Ni': ('28', '64'), 'Cu': ('29', '65'), 
'Zn': ('30', '70'), 'Ga': ('31', '71'), 'Ge': ('32', '76'), 'As': ('33', '75'), 'Se': ('34', '82'), 'Br': ('35', '81'), 'Kr': ('36', '86'), 'Rb': ('37', '87'), 
'Sr': ('38', '88'), 'Y': ('39', '89'), 'Zr': ('40', '96'), 'Nb': ('41', '93'), 'Mo': ('42', '100'), 'Tc': ('43', '99'), 'Ru': ('44', '104'), 'Rh': ('45', '103'), 
'Pd': ('46', '110'), 'Ag': ('47', '109'), 'Cd': ('48', '116'), 'In': ('49', '115'), 'Sn': ('50', '124'), 'Sb': ('51', '123'), 'Te': ('52', '130'), 'I': ('53', '127'),
'Xe': ('54', '136'), 'Cs': ('55', '133'), 'Ba': ('56', '138'), 'La': ('57', '139'), 'Ce': ('58', '142'), 'Pr': ('59', '141'), 'Nd': ('60', '150'), 'Pm': ('61', '147'), 
'Sm': ('62', '154'), 'Eu': ('63', '153'), 'Gd': ('64', '160'), 'Tb': ('65', '159'), 'Dy': ('66', '164'), 'Ho': ('67', '165'), 'Er': ('68', '170'), 'Tm': ('69', '169'), 
'Yb': ('70', '176'), 'Lu': ('71', '176'), 'Hf': ('72', '180'), 'Ta': ('73', '181'), 'W': ('74', '186'), 'Re': ('75', '187'), 'Os': ('76', '192'), 'Ir': ('77', '193'), 
'Pt': ('78', '198'), 'Au': ('79', '197'), 'Hg': ('80', '204'), 'Tl': ('81', '205'), 'Pb': ('82', '208'), 'Bi': ('83', '209'), 'Po': ('84', '210'), 'At': ('85', '211'), 
'Rn': ('86', '222'), 'Fr': ('87', '223'), 'Ra': ('88', '228'), 'Ac': ('89', '227'), 'Th': ('90', '232'), 'Pa': ('91', '231'), 'U': ('92', '238'), 'Np': ('93', '237'), 
'Pu': ('94', '244'), 'Am': ('95', '243'), 'Cm': ('96', '248'), 'Bk': ('97', '249'), 'Cf': ('98', '252'), 'Es': ('99', '252'), 'Fm': ('100', '257'), 'Md': ('101', '260'), 
'No': ('102', '259'), 'Lr': ('103', '262'), 'Rf': ('104', '267'), 'Db': ('105', '268'), 'Sg': ('106', '271'), 'Bh': ('107', '272'), 'Hs': ('108', '270'), 'Mt': ('109', '276'),
'Ds': ('110', '281'), 'Rg': ('111', '280'), 'Cn': ('112', '285'), 'Nh': ('113', '284'), 'Fl': ('114', '289'), 'Mc': ('115', '288'), 'Lv': ('116', '293'), 'Ts': ('117', '292'),'Og': ('118', '294')}


class Psi4input(): 
    def __init__(self,mol2name,**kwargs): 
        # Kwargs ###################
        chrg=None
        mult=None
        outdir=None
        mem="1000Mb"
        cpu=4
        lot="mp2"
        basis="6-31g*"
        qqm = True
        qlp = True
        quiet = False
        for kwa,vwa in kwargs.items():
            if kwa.lower() == "chrg":   chrg   = vwa 
            if kwa.lower() == "mult":   mult   = vwa 
            if kwa.lower() == "outdir": outdir = vwa 
            if kwa.lower() == "mem":    mem    = vwa 
            if kwa.lower() == "cpu":    cpu    = vwa 
            if kwa.lower() == "lot":    lot    = vwa 
            if kwa.lower() == "basis":  basis  = vwa 
            if kwa.lower() == "qqm":    qqm    = vwa 
            if kwa.lower() == "qlp":    qlp    = vwa 
            if kwa.lower() == "quiet":  quiet  = vwa 
        # Kwargs Done ###################
        mol2info = self.readmol2(mol2name)
        _,mol2info["bondnums"] = self.findbonds(mol2info["atomnames"],mol2info["bonds"])
        _,mol2info["anglnums"] = self.findangles(mol2info["atomnames"],mol2info["bonds"])
        _,mol2info["dihenums"] = self.finddihedrals(mol2info["atomnames"],mol2info["bonds"])
        self.mol2 = OrderedDict()
        self.mol2["atomposi"] = mol2info["atomposi"]
        print (self.mol2)
        if chrg is None:
            chrg = mol2info["rescharge"]
        if mult is None:
            ele = []
            for key,val in mol2info["atomposi"].items(): 
                ele.append(val[1]) 
            mult = self.calc_multip(ele,chrg) 
        psi4out = str(os.path.basename(mol2name)).split(".")[0] + ".out"
        if outdir is None: outdir = "."
        if qqm:
            x,mx,y,my,z,mz=self.runPsi4(outdir,psi4out,resn='resn',rescharge=chrg,multiplicity=mult,atomposi=mol2info["atomposi"],mem=mem,cpu=cpu,lot=lot,basis=basis,quiet=quiet)
            print (self.mol2)
            self.atomic_pol(self.mol2["atomposi"],mol2info["bondnums"],[x,mx,y,my,z,mz]) 

    def readmol2(self,mol2name,resname="resi"):
        """ Read atomnames, element, position, charges, and bonds"""
        foundatm=False
        foundbnd=False
        
        with open(mol2name,"r") as filein:
            ind = 0
            chrg = 0.0
            for line in filein:
                field = line.split()  
                if "@<TRIPOS>" in field[0].strip() and field[0].strip("@<TRIPOS>") not in ["MOLECULE","ATOM","BOND"]: break 
                if len(field) != 0 and not foundatm:
                    if field[0] == "@<TRIPOS>ATOM":
                        foundatm=True
                        foundbnd=False
                        totchrg=None
                        atminfo = OrderedDict()
                        posinfo = OrderedDict()
                        inam = {}
                        continue
                elif len(field) != 0 and not foundbnd:
                    if field[0] == "@<TRIPOS>BOND":
                        foundbnd = True
                        foundatm = False
                        bndlist = OrderedDict()
                        continue
                if foundatm and len(field) >= 4:
                    try:
                        ind += 1
                        ele = field[5].split(".")[0]
                        atnam = ele.upper()+str(ind)
                        atminfo[atnam] = ind
                        inam[ind] = atnam
                        pos = np.array(list(map(lambda x:float(x),field[2:5])))
                        posinfo[ind] = [atnam,ele,pos]
                        chrg = chrg + float(field[-1]) 
                    except (IndexError,ValueError):
                        foundatm = False
                    continue

                if foundbnd and len(field) == 4:
                    try:
                        bndlist[inam[int(field[1])]].append(inam[int(field[2])])
                    except KeyError:
                        bndlist[inam[int(field[1])]]=[inam[int(field[2])]]
                    try:
                        bndlist[inam[int(field[2])]].append(inam[int(field[1])])
                    except KeyError:
                        bndlist[inam[int(field[2])]]=[inam[int(field[1])]]
                    continue    
        totchrg = int(round(chrg))
        filein.close()
        return ({"resname":"resi","atomnames":atminfo,"atomposi":posinfo,"bonds":bndlist,"rescharge":totchrg})   

    def findbonds(self,atomlist,listofbonds):
        """Make a list of list of bond pairs with atomnames and atomindices"""
        bonds = []
        bondnum = []
        for key,value in list(listofbonds.items()):
            for val in value:
                if [val,key] not in bonds:
                    bonds.append([key,val])
                    bondnum.append([atomlist[key],atomlist[val]])
        return(bonds,bondnum)

    def findangles(self,atomlist,listofbonds):
        """Make a list of list of angle triplets with atomnames and atomindices"""
        angles = []
        anglenum = []
        for k0,v0 in list(listofbonds.items()):
            if len(v0) > 1:
               for i in range(0,(len(v0)-1)):
                   ang2 = k0
                   ang1 = v0[i]
                   for j in range(i+1,len(v0)):
                       ang3 = v0[j]
                       angles.append([ang1,ang2,ang3])
                       anglenum.append([atomlist[ang1],atomlist[ang2],atomlist[ang3]])
        return(angles,anglenum)

    def finddihedrals(self,atomlist,listofbonds):
        """Make a list of list of dihedral quads with atomnames and atomindices"""
        dihedrals = []
        dihedralnum = []
        for k0,v0 in list(listofbonds.items()):
            dih0 = k0
            for k1 in v0:
                if len(listofbonds[k1]) != 1:
                   dih1 = k1
                   for k2 in listofbonds[k1]:
                         if len(listofbonds[k2]) != 1 and k2 != dih0:
                             dih2 = k2
                             for k3 in listofbonds[k2]:
                                 if k3 != dih1 and k3 != dih0:
                                    dih3 = k3
                                    if [dih3,dih2,dih1,dih0] not in dihedrals:
                                       dihedrals.append([dih0,dih1,dih2,dih3])
                                       dihedralnum.append([atomlist[dih0],atomlist[dih1],atomlist[dih2],atomlist[dih3]])
        return(dihedrals,dihedralnum)
    
    def calc_multip(self,ele,charge):
        mult=None
        electrons=0
        totelec = 0
        for e in ele:
            totelec = totelec + int(atnumas[e][0])
        if (totelec-charge)%2==0:
            mult=1
        else:
            mult=2
        return mult

    def createlonepair(self,anam,pos,bndlst):
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
        
        return (corlist,predlpcor,lpbndlst,nlpbndlst,lpbnddict)

    def runPsi4(self,outdir,prefname,resn='resn',rescharge=0,multiplicity=1,atomposi=None,mem="1000Mb",cpu=4,lot="scf",basis="6-31g*",quiet=False):
        import pytest
        import sys
        import psi4
        from localresp import resp
        import numpy as np
        from collections import OrderedDict
    
        psi4.set_num_threads(cpu)
        psi4.set_memory(mem)
        if not quiet:
            psi4.core.set_output_file(outdir+'/'+prefname, False)
        else:
            psi4.core.be_quiet()
        
        xyz='%s %s\n' %(rescharge,multiplicity)
        xyz='    '+xyz
        n = 0
        geoline = ""
        for key,value in atomposi.items():
            geoline  = geoline + value[1] + "  " + str(value[2][0]) + "  " + str(value[2][1]) + "  " + str(value[2][2]) + " \n"
        xyz=xyz+geoline
        psi4_xyz="""
        %s
        symmetry c1
        """%(xyz)
        mol=psi4.geometry(psi4_xyz)
        mol.update_geometry() # This update is required for psi4 to load the molecule
        for i in range(mol.natom()):
            self.mol2["atomposi"][i+1][2] = np.array([mol.x(i),mol.y(i),mol.z(i)])
        opttheory=lot
        optbasis=basis
        options={'scf_type':'df','g_convergence':'gau','freeze_core':'true','mp2_type':'df','df_basis_scf':'def2-tzvpp-jkfit','df_basis_mp2':'def2-tzvppd-ri','perturb_h':'true','perturb_with':'dipole','perturb_dipole':[0.0008, 0, 0]}
        psi4.set_options(options)
        grad,wfn=psi4.gradient(opttheory+"/"+optbasis,return_wfn=True) 
        #psi4.gdma(wfn) 
        #self.createdma("xcontrol.dma","x.fchk",,1)
        psi4.fchk(wfn,'x.fchk')
        psi4.gdma(wfn,datafile="xcontrol.dma") 
        xdma_results = psi4.variable('DMA DISTRIBUTED MULTIPOLES')
        options={'scf_type':'df','g_convergence':'gau','freeze_core':'true','mp2_type':'df','df_basis_scf':'def2-tzvpp-jkfit','df_basis_mp2':'def2-tzvppd-ri','perturb_h':'true','perturb_with':'dipole','perturb_dipole':[-0.0008, 0, 0]}
        psi4.set_options(options)
        grad,wfn=psi4.gradient(opttheory+"/"+optbasis,return_wfn=True) 
        #psi4.gdma(wfn) 
        psi4.fchk(wfn,'mx.fchk')
        psi4.gdma(wfn,datafile="mxcontrol.dma") 
        mxdma_results = psi4.variable('DMA DISTRIBUTED MULTIPOLES')
        options={'scf_type':'df','g_convergence':'gau','freeze_core':'true','mp2_type':'df','df_basis_scf':'def2-tzvpp-jkfit','df_basis_mp2':'def2-tzvppd-ri','perturb_h':'true','perturb_with':'dipole','perturb_dipole':[0, 0.0008, 0]}
        psi4.set_options(options)
        grad,wfn=psi4.gradient(opttheory+"/"+optbasis,return_wfn=True) 
        #psi4.gdma(wfn) 
        psi4.fchk(wfn,'y.fchk')
        psi4.gdma(wfn,datafile="ycontrol.dma") 
        ydma_results = psi4.variable('DMA DISTRIBUTED MULTIPOLES')
        options={'scf_type':'df','g_convergence':'gau','freeze_core':'true','mp2_type':'df','df_basis_scf':'def2-tzvpp-jkfit','df_basis_mp2':'def2-tzvppd-ri','perturb_h':'true','perturb_with':'dipole','perturb_dipole':[0, -0.0008, 0]}
        psi4.set_options(options)
        grad,wfn=psi4.gradient(opttheory+"/"+optbasis,return_wfn=True) 
        #psi4.gdma(wfn) 
        psi4.fchk(wfn,'my.fchk')
        psi4.gdma(wfn,datafile="mycontrol.dma") 
        mydma_results = psi4.variable('DMA DISTRIBUTED MULTIPOLES')
        options={'scf_type':'df','g_convergence':'gau','freeze_core':'true','mp2_type':'df','df_basis_scf':'def2-tzvpp-jkfit','df_basis_mp2':'def2-tzvppd-ri','perturb_h':'true','perturb_with':'dipole','perturb_dipole':[0, 0 , 0.0008]}
        psi4.set_options(options)
        grad,wfn=psi4.gradient(opttheory+"/"+optbasis,return_wfn=True) 
        #psi4.gdma(wfn) 
        psi4.fchk(wfn,'z.fchk')
        psi4.gdma(wfn,datafile="zcontrol.dma") 
        zdma_results = psi4.variable('DMA DISTRIBUTED MULTIPOLES')
        options={'scf_type':'df','g_convergence':'gau','freeze_core':'true','mp2_type':'df','df_basis_scf':'def2-tzvpp-jkfit','df_basis_mp2':'def2-tzvppd-ri','perturb_h':'true','perturb_with':'dipole','perturb_dipole':[0, 0 , -0.0008]}
        psi4.set_options(options)
        grad,wfn=psi4.gradient(opttheory+"/"+optbasis,return_wfn=True) 
        psi4.fchk(wfn,'mz.fchk')
        psi4.gdma(wfn,datafile="mzcontrol.dma") 
        mzdma_results = psi4.variable('DMA DISTRIBUTED MULTIPOLES')
        #geo =psi4.geometry() #print_out() 
        return(xdma_results.np,mxdma_results.np,ydma_results.np,mydma_results.np,zdma_results.np,mzdma_results.np)

    def createdma(self,fchk,density="SCF",nmultipoles=1):
        towrite = """File %s Density %s \nAngstrom \nMultipoles \nLimit %s \nStart \nFinish"""%(fchk,density,nmultipoles) 
        print (towrite)

    def atomic_pol(self,atomposi,bondlist,qlist,field=0.0008,unit=0.529177249**3): 
        natoms = len(atomposi) 
        nrings = 0
        nbonds = len(bondlist) 
        gq=np.zeros((natoms,6))                    # Create empty charge array (+zero rows for ring conditions)
        gd=np.zeros((natoms,3,6))                        # Create empty atomic dipole array
        for j in range(len(qlist)):
            for i in range(len(qlist[j])):
                gq[i,j]= qlist[j][i][0] 
                gd[i,0,j]=qlist[j][i][1] 
                gd[i,1,j]=qlist[j][i][2] 
                gd[i,2,j]=qlist[j][i][3] 
        print("Substract overall charge ("+str(gq[:,0].sum())+")?")
        #answer=input("............ yes/no   ")
        for j in range(6):
            ch=gq[:,j].sum()/natoms
            for i in range(natoms):
                gq[i,j]-=ch
        
        print("")
        print("---------------------------------------------------------")
        print("------------------Starting calculation-------------------")
        print("---------------------------------------------------------")
        print("---------------------------------------------------------")
        print("------------------Atomic Polarizability------------------")
        #========================================================================================
        #   Atomic polarizability
        #========================================================================================
        #This part calculates the polarizability arising from non-uniform
        # distribution of electrons around the core
        #========================================================================================
        
        a_xx=(gd[:,0,0]-gd[:,0,3])/(2*field)*(-1)*unit     #numeric differentiation, yields polarizability
        a_yy=(gd[:,1,1]-gd[:,1,4])/(2*field)*(-1)*unit
        a_zz=(gd[:,2,2]-gd[:,2,5])/(2*field)*(-1)*unit
        a_tot_p=(a_xx+a_yy+a_zz)/3
        print("Polarization contribution:")
        print("%4s %7s %7s %7s %7s" % ("Name","a_xx","a_yy","a_zz","a_tot"))
        for i in range(natoms):
            print("%7.2f %7.2f %7.2f %7.2f " % (a_xx[i],a_yy[i],a_zz[i],a_tot_p[i]))
        print("Summed up contributions:")
        print(a_tot_p[:].sum())
        print("---------------------------------------------------------")
        print("---------------------Charge Transfer---------------------")
        #=============================================================================================
        #   CHARGE TRANSFER
        #=============================================================================================
        #This part calculates the connectivity matrix "n", the matrix with indices 
        #of each bond charge "index" the bond charges "b" and then the atomic 
        #dipoles "mu" and the polarizability "a" for the charge transfer
        #=============================================================================================
        
        
        n=np.zeros((natoms,natoms))                                        # Create empty connectivity matrix
        for i in range(nbonds):
            first=bondlist[i][0]-1                   # Read atom numbers of each bond
            second=bondlist[i][1]-1
            if first<second:                                             # Write entry in antisymmetric connectivity matrix
                n[first,second]=1
                n[second,first]=-1
            else:
                n[first,second]=-1
                n[second,first]=1
        index=np.zeros((nbonds,2))                                        # Create index matrix (which bond charge contains which atoms)
        print ("index")
        print (index)
        print (bondlist)
        print (n)
        ctr=0
        for i in range(natoms):
            for j in range(i,natoms):
                if n[i,j]==1:
                    index[ctr,0]=i
                    index[ctr,1]=j
                    ctr+=1
        a=np.zeros((natoms+nrings,nbonds))                                  # Create matrix for linear equations ab=q
        for i in range(natoms):
            for j in range(nbonds):
                if i==index[j,0]:
                    a[i,j]=1
                if i==index[j,1]:
                    a[i,j]=-1         
        for i in range(nrings):                                           # Write ring conditions into matrix a                                 
            structure=re.findall(r"\d+",nring[i])
            for j in range(len(structure)):
                if j ==len(structure)-1:
                    k=0
                else:
                    k=j+1
                first=int(structure[j])-1
                second=int(structure[k])-1
        
                factor=1
                if second<first:
                    save=first
                    first=second
                    second=save
                    factor=-1
                # Find the correct bond charge which is involved in the ring
                element=np.where(np.all(index==np.array([[first,second]]),axis=1))[0][0]              
                a[natoms+i,element]=factor                                  # Write entry in a
        print (a)    
        #Calculate bond charges and mu
        mu=np.zeros((natoms,3,6))
        for k in range(6):
            b=np.linalg.lstsq(a,gq[:,k],rcond=-1)[0]                                 # Solve linear equations ab=q for b
            print (b) 
        
            for i in range(natoms):                                         # Calculate atomic dipole from charge transfer as sum of bond charges times vector of the
                for j in range(nbonds):                                     # atom in direction of each bond 
                    if i==index[j,0]:
                        mu[i,:,k]+=(atomposi[i+1][2]-(atomposi[i+1][2]+atomposi[int(index[j,1])+1][2])/2)*b[j]
                    if i==index[j,1]:
                        mu[i,:,k]+=(atomposi[i+1][2]-(atomposi[i+1][2]+atomposi[int(index[j,0])+1][2])/2)*b[j]*(-1)
            print (mu) 
        
        a_xx=(mu[:,0,0]-mu[:,0,3])/(2*field)*1.889725989*(-1)*unit   #numeric differentiation, yields polarizability
        a_yy=(mu[:,1,1]-mu[:,1,4])/(2*field)*1.889725989*(-1)*unit
        a_zz=(mu[:,2,2]-mu[:,2,5])/(2*field)*1.889725989*(-1)*unit
        a_tot_c=(a_xx+a_yy+a_zz)/3
        print("")
        print("Results:")
        print("")
        print("Charge transfer contribution:")
        print("%4s %7s %7s %7s %7s" % ("Name","a_xx","a_yy","a_zz","a_tot"))
        for i in range(natoms):
            print("%4s %7.2f %7.2f %7.2f %7.2f " % (atomposi[i+1][0],a_xx[i],a_yy[i],a_zz[i],a_tot_c[i]))
        print("Summed up contributions:")
        print(a_tot_c[:].sum())
        
        print("Total polarizability:")
        print("%4s %7s" % ("Name","a"))
        for i in range(natoms):
             print("%4s %7.2f " % (atomposi[i+1][0],a_tot_p[i]+a_tot_c[i]))
        print("Summed up contributions:")
        print(a_tot_p[:].sum()+a_tot_c[:].sum())
        
        np.savetxt("atomic_pol.dat",np.c_[a_tot_p,a_tot_c,a_tot_p+a_tot_c])
        
        print("---------------------------------------------------------")
        print("-------------------------Goodbye-------------------------")
        print("---------------------------------------------------------")

#psi4=""" 
#    lot="scf",basis="6-31g*"):
#    options = {'N_VDW_LAYERS'       : 4,
#               'VDW_SCALE_FACTOR'   : 1.4,
#               'VDW_INCREMENT'      : 0.2,
#               'VDW_POINT_DENSITY'  : 1.0,
#               'resp_a'             : 0.0005,
#               'RESP_B'             : 0.1,
#               'BASIS_ESP'          : basis,
#               'psi4_options'       : {'df_basis_scf':'def2-tzvpp-jkfit','df_basis_mp2':'def2-tzvppd-ri', 'maxiter':250},
#               'METHOD_ESP'         : lot,
#               'RADIUS'             : {'BR':1.97,'I':2.19}
#               }
#"""
#
#Psi4pol="""
#set perturb_h true
#set perturb_with dipole
#set perturb_dipole [0.0008, 0, 0]
#grad, wfn = gradient('mp2', return_wfn=True)
#fchk(wfn,'x.fchk')
#set perturb_h true
#set perturb_with dipole
#set perturb_dipole [-0.0008, 0, 0]
#grad, wfn = gradient('mp2', return_wfn=True)
#fchk(wfn,'mx.fchk')
#set perturb_h true
#set perturb_with dipole
#set perturb_dipole [0, 0.0008, 0]
#grad, wfn = gradient('mp2', return_wfn=True)
#fchk(wfn,'y.fchk')
#set perturb_h true
#set perturb_with dipole
#set perturb_dipole [0, -0.0008, 0]
#grad, wfn = gradient('mp2', return_wfn=True)
#fchk(wfn,'my.fchk')
#set perturb_h true
#set perturb_with dipole
#set perturb_dipole [0, 0, 0.0008]
#grad, wfn = gradient('mp2', return_wfn=True)
#fchk(wfn,'z.fchk')
#set perturb_h true
#set perturb_with dipole
#set perturb_dipole [0, 0, -0.0008]
#grad, wfn = gradient('mp2', return_wfn=True)
#fchk(wfn,'mz.fchk')
#"""
#
#psi4polesp = """
#mol.update_geometry()
#options = {'N_VDW_LAYERS' : 4,
#'VDW_SCALE_FACTOR' : 1.4,
#'VDW_INCREMENT' : 0.2,
#'VDW_POINT_DENSITY' : 20.0,
#'resp_a' : 0.0005,
#'RESP_B' : 0.1,
#'BASIS_ESP':'3-21G',
#'METHOD_ESP':'HF',
#'RADIUS':{'BR':1.97,'I':2.19}
#}
## Call for first stage fit
#charges1 = resp.resp([mol], [options])
#calculate()
#"""
#
#psi4polesp2="""
#set {
#basis Sadlej
#e_convergence 6
#d_convergence 8
#scf_type df
#df_basis_scf def2-tzvpp-jkfit
#df_basis_mp2 def2-tzvppd-ri
#}
#set perturb_h true
#set perturb_with dipole
#set perturb_dipole [0.0008, 0, 0]
#property('mp2', properties=['grid_esp','dipole'])
#"""
#    Psi4moltmpl='''
#psi4_xyz="""
#%s
#%s
#"""
#mol=psi4.geometry(psi4_xyz)
#mol.update_geometry() # This update is required for psi4 to load the molecule
#'''
#
#    Psi4smoltmpl='''
#psi4_xyz="""
#%s
#"""
#mol=psi4.geometry(psi4_xyz)
#mol.update_geometry() # This update is required for psi4 to load the molecule
#'''
#
#    Psi4polarloop='''
#opttheory="%s"
#optbasis="%s"
#psi4.prop(opttheory+"/"+optbasis, properties=["DIPOLE"])
#zero_ene=psi4.energy(opttheory+"/"+optbasis)
#field=%f
#fnames={0:'x',1:'y',2:'z'}
#polars={}
#for i in range(3):
#    tmpfield=[0,0,0]
#    tmpfield[i]=abs(field)
#    psi4.set_options({'scf_type': 'df', 'g_convergence':'gau','freeze_core':'true','mp2_type':'df','perturb_h':'true','perturb_with':'dipole','perturb_dipole':tmpfield})
#    p_ene=psi4.energy(opttheory+'/'+optbasis)
#    tmpfield[i]=abs(field)*-1
#    psi4.set_options({'scf_type': 'df', 'g_convergence':'gau','freeze_core':'true','mp2_type':'df','perturb_h':'true','perturb_with':'dipole','perturb_dipole':tmpfield})
#    n_ene=psi4.energy(opttheory+'/'+optbasis)
#    enes=np.array([zero_ene,p_ene,n_ene,field],dtype='float64') #Lets ensure it floats are always with numpy
#    pol_i=abs((enes[1]-(2*enes[0])+enes[2])/(enes[3]**2))
#    polars[fnames[i]+fnames[i]]=pol_i
#
#total=np.sum(np.array(list(polars.values()),dtype='float64'))/3.0
#psi4.core.print_out('Polar RESULTS XX:'+str(polars['xx'])+' YY:'+str(polars['yy'])+' ZZ:'+str(polars['zz'])+' Total:'+str(total)+'\\n')
#'''
#
#
#    def z2c(self,coor,reschrg,multiplicity,nosymm=None,extraspace=False,notmpl=False):
#        '''
#        This function takes up a coordinate dictionary or geometry object and returns an xyz format for Psi4:
#        xyz , this is a simple xyz coordinate file element names as the first character. It also contains charge and multiplicity information
#        If charge is not given it is set to 0. If multiplicity is not given it is set to 1. Most of the time multiplicity is 1.
#        If you need extraspace, which is the case when using coordinates for Psi4 binary set extraspace to True.
#        '''
#        #We have extra fields with other residue numbering we remove this for psi4 below in for loop.
#        if not nosymm:
#            nosymm=""
#        xyz='%d %d\n' %(reschrg,multiplicity)
#        if extraspace:
#            xyz='    '+xyz
#        for coord in coor:
#            atcrd=str('%-5s' %coord[1][0])+'\t'+str('%-7f' %float(coord[2]))+'\t'+str('%-7f' %float(coord[3]))+'\t'+str('%-7f' %float(coord[4]))+'\n'
#            if extraspace:
#                atcrd='    '+atcrd
#            xyz=xyz+atcrd
#        if notmpl:
#            return(xyz)
#        moltmpl=self.Psi4moltmpl %(xyz,nosymm)
#        return(moltmpl)
#
#def Psi4coor(resn='resn',rescharge=0,multiplicity=1,coor=None,
#    chrgmult='%s %s\n\n'%(rescharge,multiplicity)
#    xyz='    '+xyz
#    n = 0
#    geoline = ""
#    for key in coor:
#        if key[0] != "RBI": n = n + 1
#        if key[0][0:2] != "LP" and key[0][0:1] != "D" and key[0][0:3] != "RBI":
#            geoline  = geoline + key[0][0:1] + "  " + str(key[1]) + "  " + str(key[2]) + "  " + str(key[3]) + " \n"
#    xyz=xyz+geoline
#    psi4_xyz="""
#    %s
#    no_reorient
#    no_com
#    """%(xyz)
#    mol=psi4.geometry(psi4_xyz)
#    mol.update_geometry() 
#    
#    
#    lot="scf",basis="6-31g*"):
#    options = {'N_VDW_LAYERS'       : 4,
#               'VDW_SCALE_FACTOR'   : 1.4,
#               'VDW_INCREMENT'      : 0.2,
#               'VDW_POINT_DENSITY'  : 1.0,
#               'resp_a'             : 0.0005,
#               'RESP_B'             : 0.1,
#               'BASIS_ESP'          : basis,
#               'psi4_options'       : {'df_basis_scf':'def2-tzvpp-jkfit','df_basis_mp2':'def2-tzvppd-ri', 'maxiter':250},
#               'METHOD_ESP'         : lot,
#               'RADIUS'             : {'BR':1.97,'I':2.19}
#               }
#    qlp=True,lpcoor=None,lplist=None):
#    if qlp:
#        xyz='' 
#        n = 0
#        geoline = ""
#        for key in lpcoor:
#            if key[0][0:2] == "LP": 
#               geoline  = geoline + key[0] + "  " + str(key[1]) + "  " + str(key[2]) + "  " + str(key[3]) + " \n"
#        xyz=xyz+geoline
#        lp="""
#        %s
#        """%(xyz)
#        
#        options['LPCOOR'] = lp
#
#    mol.set_name('stage1')
#    charges1 = resp.resp([mol], [options])
#    
#    if qlp:
#       allcoor = coor + lpcoor
#    else:
#       allcoor = coor
#
#    respchar = OrderedDict()
#    for i in range(len(allcoor)):
#        respchar[allcoor[i][0]]  = charges1[0][1][i]
#    
#    if qlp:
#        for key in list(respchar.keys()):
#            if key in list(lplist.keys()):
#                tobedist = respchar[key]/len(lplist[key])
#                respchar[key] = 0.0
#                for val in lplist[key]:
#                    respchar[val] = respchar[val] + tobedist
#
#    stage2=resp.stage2_helper()
#    stage2.set_stage2_constraint(mol,list(respchar.values()),options,cutoff=1.2)
#    options['resp_a'] = 0.001
#    options['grid'] = '1_%s_grid.dat' %mol.name()
#    options['esp'] = '1_%s_grid_esp.dat' %mol.name()
#    fout = open(outdir+"/"+"resp.dat","w")
#    
#    charges2 = resp.resp([mol], [options])
#
#    for i in range(len(allcoor)):
#        respchar[allcoor[i][0]]  = charges2[0][1][i]
#    for key,value in respchar.items():
#        fout.write("%s  %7.3f \n"%(key, value))
#    os.remove('1_%s_grid.dat' %mol.name()) 
#    os.remove('1_%s_grid_esp.dat' %mol.name()) 
#
#
#    def writegeom(self,inpdata,Psi4header,options=None):
#        #field is set by default to 0.0008
#        shortnames={'optimize':'opt','frequency':'freq','dipole':'dipo','polar':'pol','energy':'ene'}
#        if not options:
#            options="'scf_type': 'df', 'g_convergence':'gau','freeze_core':'true'"
#        Psi4footer=self.Psi4footer%(self.theory,self.basis,options)
#        runfn=self.outpath+'/'+self.outname if self.outname else self.outpath+'/%s_%s.py' %(self.resn.lower(),'_'.join([shortnames[i] for i in self.todo]))
#        try:
#            runf=open(runfn,"w")
#        except FileNotFoundError:
#            return None
#        xyz=self.Psi4xyz(inpdata['coor'],self.reschrg,self.multiplicity,nosymm=self.nosymm)
#        runf.write(Psi4header)
#        runf.write(xyz)
#        runf.write(Psi4footer)
#        runf.write('\n')
#        if 'optimize' in self.todo:
#            runf.write('psi4.optimize(opttheory+"/"+optbasis,mol=mol)\n')
#        if 'frequency' in self.todo:
#            runf.write('psi4.core.print_out("Hessian START\\n")\n')
#            runf.write('E,wfn=psi4.frequency(opttheory+"/"+optbasis,return_wfn=True)\n')
#            runf.write('wfn.hessian().print_out()\n')
#            runf.write('psi4.core.print_out("Hessian END\\n")\n')
#        if 'polar' in self.todo:
#            field=0.0008
#            Psi4polar=self.Psi4polarloop%(self.theory,self.basis,field)
#            runf.write('psi4.core.print_out("Dipole START\\n")\n')
#            runf.write(Psi4polar)
#            runf.write('psi4.core.print_out("Dipole END\\n")\n')
#        if 'polar' not in self.todo and 'dipole' in self.todo:
#            runf.write('psi4.core.print_out("Dipole START\\n")\n')
#            runf.write('psi4.prop(opttheory+"/"+optbasis, properties=["DIPOLE"])\n')
#            runf.write('psi4.core.print_out("Dipole END\\n")\n')
#        if 'energy' in self.todo:
#            runf.write('psi4.energy(opttheory+"/"+optbasis)\n')
#        runf.close()
#        return(runfn)
#
#
#
#  
#
#
#
# 
#
#        

    

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        rsarg = []
        with values as f:
            argline = f.read()
            arglist = argline.split()
      
        parser.parse_args(nrsarg, namespace)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mol2","--mol2file",help="Required Argument: Provide Mol2 File")
#    parser.add_argument("-pdb","--pdbfile",help="Required Argument: Provide Pdb file") 
#    parser.add_argument("-psf","--psffile",help="Required Argument: Pdb file requires psf file")
    parser.add_argument('-qqm','--noqm', action='store_false',default=True,help='Insert this flag if you want not to perform qm calculation.')
    parser.add_argument('-quiet','--quiet', action='store_true',default=False,help='Insert this flag if you want not to perform qm calculation.')
    parser.add_argument("-dir","--workdir",type=str,default=".",help="Enter were the output files will be saved")
    parser.add_argument("-qlp","--qlonepair",type=bool,default=True,help="No if you dont want to attach lone pairs to the acceptor atoms")
    parser.add_argument("-mem","--memory",type=str,default="1000Mb",help="Memory")
    parser.add_argument("-cpu","--nthreads",type=int,default=4,help="Number of threads")
    parser.add_argument("-lot","--theory",type=str,default="scf",help="Enter level of theory")
    parser.add_argument("-basis","--basis",type=str,default="6-31g*",help="Basis set")
    parser.add_argument("-c","--charge",type=int,default=0,help="Charge of the molecule, default is 0")
    parser.add_argument("-m","--multiplicity",type=int,default=1,help="Multiplicity of the molecule, default is 1")
    parser.add_argument("-f","--file",type=open,action=LoadFromFile)
    args = parser.parse_args()
    Psi4input(args.mol2file,chrg=args.charge,mult=args.multiplicity,basis=args.basis,lot=args.theory,mem=args.memory,cpu=args.nthreads,outdir=args.workdir,qqm=args.noqm,quiet=args.quiet)    

if __name__ == "__main__":
   main()

#    def _masses(self,fil="atomic_info.dat"):
#        f = open(fil,"r")
#        poo = f.readlines()
#        
#        atinfo = OrderedDict()
#        n = 0
#        for line in poo:
#            if line.startswith("Atomic Number"):
#               atnum = line.split()[-1]   
#               n = n + 1
#            if line.startswith("Atomic Symbol"):
#               atsym  = line.split()[-1] 
#               n = n + 1
#            if line.startswith("Mass Number"):
#               atmss  = line.split()[-1] 
#               n = n + 1   
#            if n%3 == 0:
#                atinfo[atsym] = (atnum,atmss)
#        return(atinfo)        
#def _readpdbpsf(self,pdbname,psfname):
#    bndlst = {}
#    anam = []
#    pos = []
#    filein = open(pdbname, "r") 
#    for d in filein:
#        if d[:4] == 'ATOM' or d[:6] == "HETATM":
#            splitted_line = [d[:6], d[6:11], d[12:16], d[17:21], d[21], d[22:26], d[30:38], d[38:46], d[46:54]]
#            resi = splitted_line[3]
#            anam.append(splitted_line[2].strip()) 
#            pos.append(splitted_line[6:9])
#    filein.close()
#    filein = open(psfname, "r") 
#    data = [line.split() for line in filein]
#    filein.close()
#    
#    # get charges and bondlist from psf
#    readbond = False
#    for ind,field in enumerate(data):
#        if len(field) > 2 and field[1] == "!NBOND:":
#            readbond = True
#            strtread = ind + 1
#    
#        if len(field) > 2 and field[1] == "!NBOND:":
#            readbond = True
#            strtread = ind + 1
#        if len(field) > 2 and field[1] == "!NTHETA:":
#            readbond = False
#            break
#        if readbond and ind >= strtread:
#            for i in range(0,len(field),2):
#                if 'LP' not in [anam[int(field[i])-1][:2], anam[int(field[i+1])-1][:2]] and 'D' not in [anam[int(field[i])-1][:1], anam[int(field[i+1])-1][:1]]:
#                    try:
#                        bndlst[anam[int(field[i])-1]].append(anam[int(field[i+1])-1])
#                    except KeyError:
#                        bndlst[anam[int(field[i])-1]]=[anam[int(field[i+1])-1]]
#                    try:
#                        bndlst[anam[int(field[i+1])-1]].append(anam[int(field[i])-1])
#                    except KeyError:
#                        bndlst[anam[int(field[i+1])-1]]=[anam[int(field[i])-1]]
#    
#    return (resi, anam, pos, bndlst) 
#
#
#
#def _printxyz(self,outdir,prefname,cor,predlpcor):
#    f = open(outdir+"/"+prefname,"w") 
#    n = 0
#    for key in cor:
#        if key[0] != "RBI": n = n + 1
#        if key[0][0:2] != "LP" and key[0][0:1] != "D" and key[0][0:3] != "RBI":
#           f.write("{:4s}   {:8.3f} {:8.3f} {:8.3f}\n".format(key[0],key[1],key[2],key[2]))
#    for key in predlpcor:
#        n = n + 1
#        f.write("{:4s}   {:8.3f} {:8.3f} {:8.3f}\n".format(key[0],key[1],key[2],key[3]))
#    f.close() 
#
#
#def _printpdb(self,outdir,prefname,resi,cor,predlpcor):
#    f = open(outdir+"/"+prefname,"w") 
#    n = 0
#    for key in cor:
#        if key[0] != "RBI": n = n + 1
#        if key[0][0:2] != "LP" and key[0][0:1] != "D" and key[0][0:3] != "RBI":
#           f.write("{:6s}{:5d} {:^4s}{:4s}  {:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:4s}\n".format('ATOM',n,key[0],resi,1,key[1],key[2],key[3],0.0,0.0,resi))
#    for key in predlpcor:
#        n = n + 1
#        f.write("{:6s}{:5d} {:^4s}{:4s}  {:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:4s}\n".format('ATOM',n,key[0],resi,1,key[1],key[2],key[3],0.0,0.0,resi))
#    f.write("{:6s}{:5d}     {:4s}  {:4d}\n".format('TER',n+1,resi,1))
#    f.write("END")
#    f.close() 
#
#def printallele(outdir,prefname,cor,predlpcor):
#    f = open(outdir+"/"+prefname,"w") 
#    for key in cor:
#        if key[0][0:2] != "LP" and key[0][0:1] != "D" and key[0][0:3] != "RBI":
#           
#           f.write("{:4s}\n".format(key[0].strip("0123456789"))) 
#    for key in predlpcor:
#        f.write("{:4s}\n".format(key[0].strip("0123456789")))
#    f.close() 
# def findbonds(coorv,lpcoorv,bndlstv,lplist,lpdict):
#     listallbonds = {**bndlstv,**lplist}
#     listallbonds = {**listallbonds,**lpdict}
#     #print ("bonds",listallbonds)
#     atomlist = []
#     for k in coorv:
#         atomlist.append(k[0]) 
#     for k in lpcoorv:
#         atomlist.append(k[0]) 
#     bonds = []
#     bondnum = []
#     for key,value in list(listallbonds.items()):
#         if key[0:2] == "LP":
#           for val in value:
#             if [val,key] not in bonds:
#                 bonds.append([key,val])
#                 bondnum.append([atomlist.index(key),atomlist.index(val)])
#     return(bonds,bondnum)

# def printlpbnd(outdir,prefname,nbnd,bndlist):
#     f = open(outdir+"/"+prefname,"w") 
#     n = nbnd
#     for key in bndlist:
#         n = n + 1
#         f.write("%s  %s  %s  %s \n"%(n, key[0],key[1], "1"))
#     f.close() 
    
# def findangles(coorv,lpcoorv,bndlstv,lplist,lpdict):
#     listallbonds = {**bndlstv,**lplist}
#     listallbonds = {**listallbonds,**lpdict}
#     atomlist = []
#     for k in coorv:
#         atomlist.append(k[0]) 
#     for k in lpcoorv:
#         atomlist.append(k[0]) 
#     angles = []
#     anglenum = []
#     for k0,v0 in list(listallbonds.items()):
#         if len(v0) > 1:
#            for i in range(0,(len(v0)-1)):
#                ang2 = k0
#                ang1 = v0[i]
#                for j in range(i+1,len(v0)):
#                    ang3 = v0[j]
#                    angles.append([ang1,ang2,ang3])
#                    anglenum.append([atomlist.index(ang1),atomlist.index(ang2),atomlist.index(ang3)])
#                    #anglenum.append([atomlist[ang1],atomlist[ang2],atomlist[ang3]])
#     #print (angles, anglenum)               
#     return(angles,anglenum)

# def printlpang(outdir,prefname,anglist):
#     f = open(outdir+"/"+prefname,"w") 
#     for key in anglist:
#         f.write("%s  %s\n"%(key[0],key[2]))
#     f.close() 

# def finddihedrals(coorv,lpcoorv,bndlstv,lplist,lpdict):
#     listallbonds = {**bndlstv,**lplist}
#     listallbonds = {**listallbonds,**lpdict}
#     atomlist = []
#     for k in coorv:
#         atomlist.append(k[0]) 
#     for k in lpcoorv:
#         atomlist.append(k[0]) 
#     dihedrals = []
#     dihedralnum = []
#     for k0,v0 in list(listallbonds.items()):
#         dih0 = k0
#         for k1 in v0:
#             if len(listallbonds[k1]) != 1:
#                dih1 = k1
#                for k2 in listallbonds[k1]:
#                      if len(listallbonds[k2]) != 1 and k2 != dih0:
#                          dih2 = k2
#                          for k3 in listallbonds[k2]:
#                              #if len(listofbonds[k3]) != 1 and k3 != dih1 and k3 != dih0:
#                              if k3 != dih1 and k3 != dih0:
#                                 dih3 = k3
#                                 if [dih3,dih2,dih1,dih0] not in dihedrals:
#                                    dihedrals.append([dih0,dih1,dih2,dih3])
#                                    dihedralnum.append([atomlist.index(dih0),atomlist.index(dih1),atomlist.index(dih2),atomlist.index(dih3)])
#                                   # self.dihedralnum.append([atomlist[dih0],atomlist[dih1],atomlist[dih2],atomlist[dih3]])
#     return(dihedrals,dihedralnum)

# def printlpdih(outdir,prefname,dihlist):
#     f = open(outdir+"/"+prefname,"w") 
#     for key in dihlist:
#         f.write("%s  %s\n"%(key[0],key[3]))
#     f.close()

# def dotproduct(v1, v2):
#       return sum((a*b) for a, b in zip(v1, v2))

# def length(v):
#       return math.sqrt(dotproduct(v, v))

# def angle(v1, v2):
#       radi = math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
#       return math.degrees(radi)

#        if anam[i][0:1] == "N" and len(bndlst.get(anam[i])) == 3 or anam[i][0:1] == "P" and len(bndlst.get(anam[i])) == 3:
#              ata = bndlst.get(anam[i])[0]
#              atb = bndlst.get(anam[i])[1]
#              atc = bndlst.get(anam[i])[2]
#              v1.append(cor.get(ata)[0] - cor.get(atb)[0])
#              v1.append(cor.get(ata)[1] - cor.get(atb)[1])
#              v1.append(cor.get(ata)[2] - cor.get(atb)[2])
#              v2.append(cor.get(ata)[0] - cor.get(atc)[0])
#              v2.append(cor.get(ata)[1] - cor.get(atc)[1])
#              v2.append(cor.get(ata)[2] - cor.get(atc)[2])
#              vc = np.cross(v1,v2)
#              v3.append(cor.get(anam[i])[0] - cor.get(ata)[0])
#              v3.append(cor.get(anam[i])[1] - cor.get(ata)[1])
#              v3.append(cor.get(anam[i])[2] - cor.get(ata)[2])
#              poav = abs(angle(vc,v3))
#              if poav <= 90.0:
#                 poav = 180.0 - poav
#              v1 = []
#              v2 = [] 
#              v3 = []
#              v1.append(cor.get(anam[i])[0] - cor.get(ata)[0])
#              v1.append(cor.get(anam[i])[1] - cor.get(ata)[1])
#              v1.append(cor.get(anam[i])[2] - cor.get(ata)[2])
#              v2.append(cor.get(anam[i])[0] - cor.get(atb)[0])
#              v2.append(cor.get(anam[i])[1] - cor.get(atb)[1])
#              v2.append(cor.get(anam[i])[2] - cor.get(atb)[2])
#              vi = np.cross(v1,v2)
#              impr = abs(angle(vi,vc))
#              if impr < 90.0:
#                 impr = 90.0 - impr
#              elif impr > 90.0:
#                 impr = 180.0 - impr
#                 impr = 90.0 - impr
#              v1 = []
#              v2 = []
#              v3 = []
#              if poav > 100.0:
#                 hcoor = [cor[anam[i]],cor[ata],cor[atb]]
#                 n = n+1
#                 lpname = "LP"+str(n) 
#                 lpbndlst[anam[i]] = [lpname]
#                 lpbnddict[lpname] = [anam[i]]
#                 nlpbndlst[i] = [n + natmwolp - 1]
#                 if anam[i][0:1] == "N": 
#                    #predlpcor[lpname] = relative(0.30,poav,impr,hcoor)
#                    predlpcor.append([lpname]+relative(0.30,poav,impr,hcoor))
#                 if anam[i][0:1] == "P": 
#                    #predlpcor[lpname] = relative(0.70,poav,impr,hcoor)
#                    predlpcor.append([lpname]+relative(0.70,poav,impr,hcoor))
