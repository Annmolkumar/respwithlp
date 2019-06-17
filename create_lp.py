import sys
import os
import math
import numpy as np

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

foundatm=False
foundbnd=False
atomind = []
anam = []
pos = []
bndlst = {}
mol2file = None; pdbfile = None; psffile = None
if sys.argv[1].split(".")[1] == "mol2":
   mol2file = sys.argv[1]
if sys.argv[1].split(".")[1] == "pdb":
   pdbfile = sys.argv[1]
   psffile = sys.argv[2]

predpdb = os.path.basename(pdbfile).split("_")[0]
outdir = "predlp"
os.system("mkdir -p "+ outdir)

if pdbfile is not None:
   filein = open(pdbfile, "r") 
   data = [line.split() for line in filein]
   filein.close()
   anam = []
   pos = []
   for d in data:
       if d[0] == "ATOM" and len(d) > 6:
          resi = d[3]
          anam.append(d[2]) 
          pos.append(d[5:8])
          
if psffile is not None:   
   filein = open(psffile, "r") 
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
              #Note on the change: I need the bond list information for lone pairs below to transfer the charge to main atom. So I will remove them later.
              if 'LP' not in [anam[int(field[i])-1][0:2], anam[int(field[i+1])-1][0:2]] and 'D' not in [anam[int(field[i])-1][0:1], anam[int(field[i+1])-1][0:1]]:
                  try:
                      bndlst[anam[int(field[i])-1]].append(anam[int(field[i+1])-1])
                  except KeyError:
                      bndlst[anam[int(field[i])-1]]=[anam[int(field[i+1])-1]]
                  try:
                      bndlst[anam[int(field[i+1])-1]].append(anam[int(field[i])-1])
                  except KeyError:
                      bndlst[anam[int(field[i+1])-1]]=[anam[int(field[i])-1]]
          

          
if mol2file is not None: 
   with open(mol2file,"r") as filein:
     for ind,line in enumerate(filein):
        field = line.split()  
        if len(field) != 0 and not foundatm :
           if field[0] == "@<TRIPOS>ATOM":
              foundatm=True
              foundbnd=False
              atomind = []
              anam = []
              pos = []
              pass
        elif len(field) != 0 and not foundbnd:
            if field[0] == "@<TRIPOS>BOND":
               foundbnd = True
               foundatm = False
        if foundatm and len(field) >= 4:
           try:
              atomind.append(field[0]) 
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
   #           bndlst[field[2]].append(field[1])
           except KeyError:
              bndlst[anam[int(field[2])-1]]=[anam[int(field[1])-1]]
             # bndlst[field[2]]=[field[1]]
           #  break
natoms = len(anam)
panam = anam
cor = {}        
predlpcor = {}        
lplist = []
for i in range (natoms):
    cor[anam[i]] = [float(pos[i][0]),float(pos[i][1]),float(pos[i][2])]
    if anam[i][0:2] == "LP" and anam[i][0:3] != "LPX":
        lplist.append(anam[i])
#print (lplist)
f = open(outdir+"/"+predpdb+"_predlp.pdb","w") 
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
