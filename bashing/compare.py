import sys
import numpy as np

pdb1 = sys.argv[1]
pdb2 = sys.argv[2]
cor = {}
pdcor = {}
if pdb1 is not None:
   filein = open(pdb1, "r") 
   data = [line.split() for line in filein]
   filein.close()
   anam = []
   pos = []
   for d in data:
       if d[0] == "ATOM" and len(d) > 6 and d[2][0:2] == "LP" and d[2][0:3] != "LPX":
          cor[d[2]] = [float(d[5]),float(d[6]),float(d[7])]

if pdb2 is not None:
   filein = open(pdb2, "r") 
   data = [line.split() for line in filein]
   filein.close()
   anam = []
   pos = []
   for d in data:
       if d[0] == "ATOM" and len(d) > 6 and d[2][0:2] == "LP" and d[2][0:3] != "LPX":
          pdcor[d[2]] = [float(d[5]),float(d[6]),float(d[7])]

if len(cor) != len(pdcor):
   print ("Number of lone pairs are not equal")
   sys.exit()

for key in cor.keys(): 
    diff = [lp - plp for lp,plp in zip(cor[key], pdcor[key])]
    magdiff =  np.linalg.norm(diff)
    print (magdiff)
