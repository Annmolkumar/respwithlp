import sys
import os
import math
import numpy as np
from create_psi4inpfile import Psi4Writer as psw

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
   bndlst = {}
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
   return (anam, pos, bndlst)       

def createlonepair(anam,pos,bndlst):
    natoms = len(anam)
    cor = {}        
    predlpcor = {}        
    lplist = []
    for i in range (natoms):
        cor[anam[i]] = [float(pos[i][0]),float(pos[i][1]),float(pos[i][2])]
        if anam[i][0:2] == "LP" and anam[i][0:3] != "LPX":
            lplist.append(anam[i])
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
                        if anam[i] != atms: # bndlst.get(ata)[0]:
                           if atms[0:1] != "H":
                              atb = atms #bndlst.get(ata)[bndlst.get(ata).index(atms)]
                              break
                    if atb == "": # bndlst.get(ata)[0]:
                       for atms in bndlst.get(ata): 
                           if anam[i] != atms: # bndlst.get(ata)[0]:
                              atb = atms #bndlst.get(ata)[bndlst.get(ata).index(atms)]
                              break
    
              lpname = lplist[n]  
              n = n+1
              hcoor = [cor[anam[i]],cor[ata],cor[atb]]
              if anam[i][0:1] == "O": 
                 #predlpcor[lpname] = relative(0.35,110.0,90.0,hcoor) # Out of plane lone pairs
                 predlpcor[lpname] = relative(0.35,110.0,0.0,hcoor)
              if anam[i][0:1] == "S": 
                 #predlpcor[lpname] = relative(0.75,95.0,260.0,hcoor)
                 predlpcor[lpname] = relative(0.75,95.0,0.0,hcoor)
    
              lpname = lplist[n]  
              n = n+1
           #  hcoor = [cor[anam[i]],cor[atb],cor[ata]] # Order flipped 
              if anam[i][0:1] == "O": 
                 #predlpcor[lpname] = relative(0.35,110.0,90.0,hcoor)
                 predlpcor[lpname] = relative(0.35,110.0,180.0,hcoor)
              if anam[i][0:1] == "S": 
                 #predlpcor[lpname] = relative(0.75,95.0,260.0,hcoor)
                 predlpcor[lpname] = relative(0.75,95.0,180.0,hcoor)
    
        if anam[i][0:1] == "N" and len(bndlst.get(anam[i])) == 1:
              slope = []
              ata = bndlst.get(anam[i])[0]
              hcoor = [cor[anam[i]],cor[ata]]
              lpname = lplist[n]  
              n = n+1
              predlpcor[lpname] = colinear(0.35,1.00,hcoor)
              
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
              lpname = lplist[n]  
              n = n+1
              predlpcor[lpname] = relative(0.35,110.0,90.0,hcoor)
    
              lpname = lplist[n]  
              n = n+1
              predlpcor[lpname] = relative(0.35,110.0,270.0,hcoor)
    
        if anam[i][0:1] == "S" and len(bndlst.get(anam[i])) == 2:
              # lonepair bisector
              ata = bndlst.get(anam[i])[0]
              atb = bndlst.get(anam[i])[1]
              cor["RBI"] = [(ca-cb)/2.0 for ca,cb in zip(cor[ata],cor[atb])]
              cor["RBI"] = [ca+cb for ca,cb in zip(cor[atb],cor["RBI"])]
    
              hcoor = [cor[anam[i]],cor["RBI"],cor[atb]]
              lpname = lplist[n]  
              n = n+1
              predlpcor[lpname] = relative(0.70,95.0,100.0,hcoor)
    
              lpname = lplist[n]  
              n = n+1
              hcoor = [cor[anam[i]],cor[atb],cor["RBI"]]  # Flipped order
              predlpcor[lpname] = relative(0.70,95.0,100.0,hcoor)
                
        if anam[i][0:1] == "N" and len(bndlst.get(anam[i])) == 2 or anam[i][0:1] == "P" and len(bndlst.get(anam[i])) == 2:
               # lonepair bisector
              ata = bndlst.get(anam[i])[0]
              atb = bndlst.get(anam[i])[1]
              cor["RBI"] = [(ca-cb)/2.0 for ca,cb in zip(cor[ata],cor[atb])]
              cor["RBI"] = [ca+cb for ca,cb in zip(cor[atb],cor["RBI"])]
              lpname = lplist[n]  
              n = n+1
              hcoor = [cor[anam[i]],cor["RBI"],cor[atb]]
              if anam[i][0:1] == "N": 
                 predlpcor[lpname] = relative(0.30,180.0,180.0,hcoor)
              if anam[i][0:1] == "P": 
                 predlpcor[lpname] = relative(0.70,180.0,180.0,hcoor)
    
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
                 lpname = lplist[n]  
                 n = n+1
                 if anam[i][0:1] == "N": 
                    predlpcor[lpname] = relative(0.30,poav,impr,hcoor)
                 if anam[i][0:1] == "P": 
                    predlpcor[lpname] = relative(0.70,poav,impr,hcoor)
    return (cor,predlpcor)

def printxyz(outdir,prefname,cor,predlpcor):
    f = open(outdir+"/"+prefname,"w") 
    n = 0
    for key in cor.keys():
        if key != "RBI": n = n + 1
        if key[0:2] != "LP" and key[0:1] != "D" and key[0:3] != "RBI":
           f.write("{:4s}   {:8.3f} {:8.3f} {:8.3f}\n".format(key,cor[key][0],cor[key][1],cor[key][2]))
    for key in predlpcor.keys():
        n = n + 1
        f.write("{:4s}   {:8.3f} {:8.3f} {:8.3f}\n".format(key,predlpcor[key][0],predlpcor[key][1],predlpcor[key][2]))
    f.close() 


def printpdb(outdir,prefname,resi,cor,predlpcor):
    f = open(outdir+"/"+prefname,"w") 
    n = 0
    for key in cor.keys():
        if key != "RBI": n = n + 1
        if key[0:2] != "LP" and key[0:1] != "D" and key[0:3] != "RBI":
           f.write("{:6s}{:5d} {:^4s}{:4s}  {:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:4s}\n".format('ATOM',n,key,resi,1,cor[key][0],cor[key][1],cor[key][2],0.0,0.0,resi))
    for key in predlpcor.keys():
        n = n + 1
        f.write("{:6s}{:5d} {:^4s}{:4s}  {:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:4s}\n".format('ATOM',n,key,resi,1,predlpcor[key][0],predlpcor[key][1],predlpcor[key][2],0.0,0.0,resi))
    f.write("{:6s}{:5d}     {:4s}  {:4d}\n".format('TER',n+1,resi,1))
    f.write("END")
    f.close() 


def main():
    mol2name = None; pdbname = None; psfname = None
    for arg in sys.argv:
        exten = arg.strip().split(".")[1]
        if exten == "mol2":
           mol2name = arg
        elif exten == "pdb":
           pdbname = arg
        elif exten == "psf":
           psfname = arg

    if mol2name:
       prefname = os.path.basename(mol2name)
       psi4name = prefname.strip().split(".")[0]
       prefdir = os.path.dirname(mol2name)
       a,b,c = readmol2(mol2name)
    if pdbname and psfname:
       prefname = os.path.basename(pdbname)
       psi4name = prefname.strip().split(".")[0]
       prefdir = os.path.dirname(pdbname)
       r,a,b,c = readpdbpsf(pdbname,psfname)
    if not mol2name and not psfname:
       print ("PDB file requires psf for bond information")
       sys.exit()
     
    d,f = createlonepair(a,b,c)

    if prefdir == "":
       outdir = "coorwithlp"
    else:
       outdir = prefdir+"/coorwithlp"
    if not os.path.exists(outdir):
       os.mkdir(outdir)

    printpdb(outdir,prefname,r,d,f)
    psw(outdir,psi4name,resn='resn',rescharge=0,multiplicity=1,coor=d,lpcoor=f,mem="1000Mb",cpu=4,lot="scf",basis="6-31g*")
    from subprocess import call
    call(["python", outdir+"/"+psi4name+".py"])
    
if __name__ == "__main__":
   print ("Either provide (mol2) file or (pdb and psf) files")
   main()


