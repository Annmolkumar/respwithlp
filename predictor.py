#!/usr/bin/python

import sys
import re
import os
import glob

wrkDir = "temporary_files_predictor"
os.system("mkdir -p "+ wrkDir)
psffile = sys.argv[1]
psf = open(psffile, 'r')
#charge = sys.argv[2]
#molname=os.path.splitext(os.path.basename(psffile))[0]
#print(molname)
typefile1 = wrkDir + '/' + 'dtypelist.dat'
typefile = wrkDir + '/' + 'typelist.dat'
atomfile = wrkDir + '/' + 'atomlist.dat'
bondfile = wrkDir + '/' + 'bondlist.dat'

tdata = [tline.split() for tline in psf]
readatom = False
readbond = False
strtaread = 0
strtbread = 0
anam = []
atyp = []
blist = []
char=[]
for idx, typ in enumerate(tdata):
    if len(typ) == 2 and typ[1] == "!NATOM":
       readatom = True
       readbond = False
       strtaread = idx + 1
    if len(typ) > 2 and typ[1] == "!NBOND:":
       readbond = True
       readatom = False
       strtbread = idx + 1
    if len(typ) > 2 and typ[1] == "!NTHETA:":
       readbond = False
       readatom = False
       break
    if readatom and idx >= strtaread and len(typ) > 5: 
          resi = typ[3]
          anam.append(typ[4]) 
          atyp.append(typ[5]) 
          char.append(float(typ[6]))
    if readbond and idx >= strtbread: 

       for i in range(0,len(typ),2):
            if 'LP' not in [anam[int(typ[i])-1][0:2], anam[int(typ[i+1])-1][0:2]] and 'D' not in [anam[int(typ[i])-1][0:1], anam[int(typ[i+1])-1][0:1]]:
                #print(anam)
                blist.append([anam[int(typ[i])-1], anam[int(typ[i+1])-1]])
charge = int(sum(char))                
t = open(typefile1,'w')                
af = open(atomfile,'w')                
bf = open(bondfile,'w')                
for a in anam:
  if 'LP' not in a[0:2] and 'D' not in a[0:1]:
    af.write("%s \n"%(a))
for a in atyp:
  if 'LP' not in a[0:2] and 'D' not in a[0:1]:
    t.write("%s \n"%(a))
for b in blist:
  bf.write("%s %s \n"%(b[0], b[1]))
t.close()
af.close()
bf.close()


def multiple_replace(dict, text):
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)

adict = eval(open(sys.argv[2]).read())
#print(adict)
with open(typefile1, 'r') as text:
   new_text = multiple_replace(adict, text.read())
   #print(new_text)
with open(typefile, "w") as result:
   result.write(new_text)

directory="."
predict_increment="yes"
predict_neural_net="yes"
calc_lonepairs="yes"
verbose="no"
os.system("python setup/topology_and_increment.py " + directory+" "+ str(charge) +"  " +predict_increment +" "+ predict_neural_net+" " +calc_lonepairs +" "+verbose+" > temporary_files_predictor/python_drude_"+ resi.lower() +".log")
