from __future__ import print_function
import re
import numpy as np
import sys

directory=sys.argv[1]
charge=int(float(sys.argv[2]))
predict_increment=sys.argv[3]
predict_neural_net=sys.argv[4]
calc_lonepairs=sys.argv[5]
verbose=sys.argv[6]
manual=False
if len(sys.argv)==8:
    manual_charge=sys.argv[7]
    m_charge=int(manual_charge)
    manual=True
    print("Manual charge: "+str(m_charge))

with open("temporary_files_predictor/atomlist.dat") as f:
    name = f.readlines()
name = [x.strip() for x in name]

with open("temporary_files_predictor/bondlist.dat") as f:
    bond = f.readlines()
bond = [x.strip() for x in bond]

d={}
for i in range(len(name)):
    d[name[i]]=str(i+1)

bonds=[]
for i in range(len(bond)):
    s=bond[i]
    pattern = re.compile(r'\b(' + '|'.join(d.keys()) + r')\b')
    result=pattern.sub(lambda x: d[x.group()], s)
    bonds.append((int(result.split()[0]),int(result.split()[1])))

angles=[]
for i in bonds:
    a=i[0]
    b=i[1]
    attached_to_a=[i for i, v in enumerate(bonds) if v[0] == a or v[1] == a]
    for k in range(len(attached_to_a)):
        if b not in bonds[attached_to_a[k]]:
            c=[x for x in bonds[attached_to_a[k]] if x!=a][0]
            if c<b:
                angles.append((c,a,b))
            
    attached_to_b=[i for i, v in enumerate(bonds) if v[0] == b or v[1] == b]
    for k in range(len(attached_to_b)):
        if a not in bonds[attached_to_b[k]]:
            c=[x for x in bonds[attached_to_b[k]] if x!=b][0]
            if a>c:
                angles.append((a,b,c))

dihedrals=[]
for i in angles:                                                                #iterates through angles array
    a=int(i[0])
    b=int(i[1])
    c=int(i[2])
    for j in bonds:                                                             #iterates through bond array
        x=int(j[0])
        y=int(j[1])
        if b not in j:                                                          #if middle atom of angle not in j(bonds)
            if a==x:                                                            #and if either side atom of angle is in j(bonds)
                if y<c:                                                         #the other atom in j(bonds) must be connected to
                    dihedrals.append((y,x,b,c))                                 #one of the side atoms, thereby forming a dihedral
                else:
                    dihedrals.append((c,b,x,y))
            elif a==y:
                if x<c:
                    dihedrals.append((x,y,b,c))
                else:
                    dihedrals.append((c,b,y,x))
            elif c==x:
                if a<y:
                    dihedrals.append((a,b,x,y))
                else:
                    dihedrals.append((y,x,b,a))
            elif c==y:
                if a<x:
                    dihedrals.append((a,b,y,x))
                else:
                    dihedrals.append((x,y,b,a))
dihedrals=list(set(dihedrals))

if verbose=="yes":
    print("Found "+str(len(bonds))+" bonds:")
    for i in range(len(bonds)):
        print(bonds[i][0],bonds[i][1])
    print("Found "+str(len(angles))+" angles:")
    for i in range(len(angles)):
        print(angles[i][0],angles[i][1],angles[i][2])
    print("Found "+str(len(dihedrals))+" dihedrals:")
    for i in range(len(dihedrals)):
        print(dihedrals[i][0],dihedrals[i][1],dihedrals[i][2],dihedrals[i][3])




# setup matrix
with open("temporary_files_predictor/typelist.dat") as f:
    types = f.readlines()
types = [x.strip() for x in types]

with open(directory+"/setup/typelist.dat") as f:
    atomtypes = f.readlines()
atomtypes = [x.strip() for x in atomtypes]
num_types=len(atomtypes)
X=np.zeros((len(name),num_types*4))

for i in range(len(name)):
    index=np.where(np.array(atomtypes)==types[i])[0][0]
    X[i,index]=1
if len(bonds)>0:
    for i in range(len(bonds)):
        index1=np.where(np.array(atomtypes)==types[int(bonds[i][0]-1)])[0][0]
        index2=np.where(np.array(atomtypes)==types[int(bonds[i][1]-1)])[0][0]
        X[int(bonds[i][0]-1),index2+num_types]+=1
        X[int(bonds[i][1]-1),index1+num_types]+=1

if len(angles)>0:
    for i in range(len(angles)):
        index1=np.where(np.array(atomtypes)==types[int(angles[i][0]-1)])[0][0]
        index2=np.where(np.array(atomtypes)==types[int(angles[i][-1]-1)])[0][0]
        X[int(angles[i][0]-1),index2+num_types*2]+=1
        X[int(angles[i][-1]-1),index1+num_types*2]+=1
if len(dihedrals)>0:
    for i in range(len(dihedrals)):
        index1=np.where(np.array(atomtypes)==types[int(dihedrals[i][0]-1)])[0][0]
        index2=np.where(np.array(atomtypes)==types[int(dihedrals[i][-1]-1)])[0][0]
        X[int(dihedrals[i][0]-1),index2+num_types*3]+=1
        X[int(dihedrals[i][-1]-1),index1+num_types*3]+=1

np.savetxt("temporary_files_predictor/x.tmp",np.c_[X],fmt='%5.0i')
    

scale=0.85
def printer(polarizability,charges,name,types,charge):
    print("")
    hlist=['HGA1','HGA2','HGA3','HGA4','HGA5','HGA6','HGA7','HGAAM0','HGAAM1','HGAAM2','HGP1','HGP2','HGP3','HGP4','HGP5','HGPAM1','HGPAM2','HGPAM3','HGR51','HGR52','HGR53','HGR61','HGR62','HGR63','HGR71']
    polarizability_h=np.copy(polarizability)
    for i in range(len(name)):
        if types[i] in hlist:
            index=[x for x, v in enumerate(bonds) if v[0] == i+1 or v[1] == i+1][0]
            non_h_atom=[x for x in bonds[index] if x!=i+1][0]
            polarizability_h[non_h_atom-1]+=polarizability[i]
    print("%12s %12s %12s %12s %12s %12s %12s" %("Name","Type","Char.","corr. Char.","Pol.", "Pol.H", str(scale)+" Pol.H"))
    print("       ------------------------------------------------------------------------------------")
    if manual == False:
        if abs(round(charges.sum())-charge) >= 0.5:
            print("DEVIATION! RESULTS MIGHT BE WRONG")
            print("Total charge (column Char.): "+str(charges.sum()))
            print("Total corrected charge (column corr.Char.): "+str(round(charges.sum())))
            print("Charge from str file: "+str(charge))
            print("Check wether the predicted total charge, or the str file charge is correct")
            print("Run again with manual charge set, by ./predictor.sh <str/mol2> <file> <correct_charge>")
            sys.exit()
        for i in range(len(name)):
            if types[i] in hlist:
                print("%12s %12s %12.3f %12.3f %12.3f" %(name[i],types[i],charges[i],charges[i]-(charges.sum()-round(charges.sum()))/len(name),polarizability[i]),"","")
            else:
                print("%12s %12s %12.3f %12.3f %12.3f %12.3f %12.3f" %(name[i],types[i],charges[i],charges[i]-(charges.sum()-round(charges.sum()))/len(name),polarizability[i],polarizability_h[i],polarizability_h[i]*scale))
        print("       ------------------------------------------------------------------------------------")
        print("%-30s %12.3f" %("Total charge: ",charges.sum()))
        print("%-30s %12.3f" %("Total corrected charge: ",round(charges.sum())))
        print("%-30s %12.3f" %("Charge from str file: ",charge))
        print("%-30s %12.3f" %("Total polarizability: ",polarizability.sum()))
        print("%-30s %12.3f" %("Total scaled polarizability: ",scale*polarizability.sum()))
    else:
        for i in range(len(name)):
            if types[i] in hlist:
                print("%12s %12s %12.3f %12.3f %12.3f" %(name[i],types[i],charges[i],charges[i]-(charges.sum()-m_charge)/len(name),polarizability[i]),"","")
            else:
                print("%12s %12s %12.3f %12.3f %12.3f %12.3f %12.3f" %(name[i],types[i],charges[i],charges[i]-(charges.sum()-round(charges.sum()))/len(name),polarizability[i],polarizability_h[i],polarizability_h[i]*scale))
        print("       ------------------------------------------------------------------------------------")
        print("%-30s %12.3f" %("Total manual charge: ",charges.sum()))
        print("%-30s %12.3f" %("Total corrected charge: ",round(charges.sum())))
        print("%-30s %12.3f" %("Charge from str file: ",charge))
        print("%-30s %12.3f" %("Total polarizability: ",polarizability.sum()))
        print("%-30s %12.3f" %("Total scaled polarizability: ",scale*polarizability.sum()))


print("Step 3A: Adding up increments - Increments...")
if predict_increment =="yes":
    increments=np.loadtxt(directory+"/setup/increment_pol.dat")
    increments_ch=np.loadtxt(directory+"/setup/increment_charge.dat")
    polarizability=np.zeros(len(name))
    charges=np.zeros(len(name))
    for i in range(len(name)):
        for j in range(num_types*4):
            polarizability[i]+=X[i,j]*increments[j]
            charges[i]+=X[i,j]*increments_ch[j]
    printer(polarizability,charges,name,types,charge)
else:
    print("Not requested")
    
print("")
print("Step 3B: Adding up increments - Neural net...")

if predict_neural_net=="yes":
    if sys.version[0] != '3':
        print("The neural net model cannot be used on any python version but python3 since the pickled model file was created using python3, and is not compatible with other versions. Switch to python3 if you want to use the neural net option.")
    else:
        from sklearn.externals import joblib
        from sklearn.neural_network import MLPRegressor

        regr_charge = joblib.load(directory+"/setup/model_charge.pkl")
        regr_pol = joblib.load(directory+"/setup/model_pol.pkl")
        inp_train=np.copy(X)
        charges = regr_charge.predict(inp_train)
        polarizability = regr_pol.predict(inp_train)
        printer(polarizability,charges,name,types,charge)
else:
    print("Not requested")        

print("")
print("Step 4: Lonepairs")
print("")

if calc_lonepairs!="yes":
    print("Not requested")
    sys.exit()


lp_index=1

nitrogen_warn_list=['NG1T1','NG2D1','NG2S0','NG2S1','NG2S2','NG2S3','NG2O1','NG2P1','NG2R43','NG2R51','NG2R52','NG2R53','NG2R57','NG2R61','NG2R67','NG2RC0','NG301','NG311','NG321','NG331','NG3C51','NG3N1','NG3P0','NG3P1','NG3P2','NG3P3']
warning=[x for x, v in enumerate(types) if v in nitrogen_warn_list]
if warning!=[]:
    print("LPs for nitrogen currently not featured, except for unprotonated nitrogen in rings (NG2R60, NG2R62 or NG2R50),")
    print("as no parametrized reference structures currently available in DrudeFF.")
    

def find_attached(bonds,atom):
    attached_index=[x for x, v in enumerate(bonds) if v[0] == atom or v[1] == atom]
    attached=[]
    for i in range(len(attached_index)):
        attached.append([x for x in bonds[attached_index[i]] if x!=atom][0])

    return attached


halogenlist=['CLGA1','CLGA3','CLGR1','BRGA1','BRGA2','BRGA3','BRGR1','IGR1','FGR1']
nitrogenlist=['NG2R62','NG2R60','NG2R50']
sulfurlist=['SG311','SG2R50','SG301','SG302']
oxygenlist=['OG311','OG302','OG2D2' ,'OD2N1','OG2D1','OG2D3','OG2D4','OG301']
lists=[halogenlist,nitrogenlist,sulfurlist,oxygenlist]

for k in range(len(lists)):
    current_list=[x for x, v in enumerate(types) if v in lists[k]]
    for i in range(len(current_list)):
        attached_1=find_attached(bonds,current_list[i]+1)
        atom1=name[current_list[i]]
        for j in range(len(attached_1)):
            attached_2=find_attached(bonds,attached_1[j])
            attached_2.remove(current_list[i]+1)
        if lists[k]==halogenlist:
            atom2=name[attached_1[0]-1]
            atom3=name[attached_2[0]-1]
            atom4=name[attached_2[1]-1]
            if types[current_list[i]]=='FGR1':
                print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 0.75 A22 1.125")
            elif types[current_list[i]]=='CLGA1':
                print("LONEPAIR COLINEAR LP"+str(lp_index)+"A "+atom1+" "+atom2+" DIST 1.64 SCALE 0.0")
                print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 1.40 A22 0.80")
            elif types[current_list[i]]=='CLGA3':
                print("LONEPAIR COLINEAR LP"+str(lp_index)+"A "+atom1+" "+atom2+" DIST 1.70 SCALE 0.0")
                print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 1.40 A22 0.80")
            elif types[current_list[i]]=='CLGR1':
                print("LONEPAIR COLINEAR LP"+str(lp_index)+"A "+atom1+" "+atom2+" DIST 1.64 SCALE 0.0")
                print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 1.30 A22 0.85")
            elif types[current_list[i]]=='BRGA1':
                print("LONEPAIR COLINEAR LP"+str(lp_index)+"A "+atom1+" "+atom2+" DIST 1.70 SCALE 0.0")
                print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 1.65 A22 0.675")
            elif types[current_list[i]]=='BRGA2':
                print("LONEPAIR COLINEAR LP"+str(lp_index)+"A "+atom1+" "+atom2+" DIST 1.90 SCALE 0.0")
                print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 1.65 A22 0.675")
            elif types[current_list[i]]=='BRGA3':
                print("LONEPAIR COLINEAR LP"+str(lp_index)+"A "+atom1+" "+atom2+" DIST 1.92 SCALE 0.0")
                print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 1.65 A22 0.675")
            elif types[current_list[i]]=='BRGR1':
                print("LONEPAIR COLINEAR LP"+str(lp_index)+"A "+atom1+" "+atom2+" DIST 1.85 SCALE 0.0")
                print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 1.80 A22 0.60")
            elif types[current_list[i]]=='IGR1':
                if "R6" in types[attached_1[0]-1]:
                    print("LONEPAIR COLINEAR LP"+str(lp_index)+"A "+atom1+" "+atom2+" DIST 2.03 SCALE 0.0") # aromatic
                    print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 1.80 A22 0.60")
                else:
                    iodines=1
                    for l in range(len(attached_2)):
                        if types[attached_2[l]-1]=='IGR1':
                            iodines+=1
                    if iodines==3:
                        print("LONEPAIR COLINEAR LP"+str(lp_index)+"A "+atom1+" "+atom2+" DIST 2.01 SCALE 0.0") # 3 iodines
                        print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 1.80 A22 0.60")
                    elif iodines==2:
                        print("LONEPAIR COLINEAR LP"+str(lp_index)+"A "+atom1+" "+atom2+" DIST 2.10 SCALE 0.0") # 2 iodines
                        print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 1.80 A22 0.60")
                    else:
                        print("LONEPAIR COLINEAR LP"+str(lp_index)+"A "+atom1+" "+atom2+" DIST 2.10 SCALE 0.0") # 1 iodine
                        print("ANISOTROPY  "+atom1+" "+atom2+" "+atom3+" "+atom4+" A11 1.72 A22 0.64")
        if lists[k]==nitrogenlist:
            atom2=name[attached_1[0]-1]
            atom3=name[attached_1[1]-1]
            print("LONEPAIR bisector LP"+str(lp_index)+"A "+atom1+" "+atom2+" "+atom3+" distance 0.35 angle 179.99 dihe 179.99")
            if types[current_list[i]]=='NG2R60':
                print("ANISOTROPY  "+atom1+" LP"+str(lp_index)+"A "+atom2+" "+atom3+" A11 1.1611  A22 0.6778") #6-mem ring with 1 N - pyridine
            elif types[current_list[i]]=='NG2R62':
                print("ANISOTROPY  "+atom1+" LP"+str(lp_index)+"A "+atom2+" "+atom3+" A11 1.2376  A22 0.6648") #6-mem ring with 2 N
            elif types[current_list[i]]=='NG2R50':
                print("ANISOTROPY  "+atom1+" LP"+str(lp_index)+"A "+atom2+" "+atom3+" A11 0.808   A22 1.384") # 5-mem ring with 2 N

        if lists[k]==sulfurlist:
            if types[current_list[i]]=='SG311' or types[current_list[i]]=='SG2R50':
                if "HG" in types[attached_1[0]-1]: # Thiol H first
                    atom2=name[attached_1[1]-1]
                    atom3=name[attached_1[0]-1]
                    print("LONEPAIR relative LP"+str(lp_index)+"A "+atom1+" "+atom2+" "+atom3+" distance 0.8 angle 95.0 dihe 100.0")
                    print("LONEPAIR relative LP"+str(lp_index)+"B "+atom1+" "+atom2+" "+atom3+" distance 0.8 angle 95.0 dihe 260.0")
                    print("ANISOTROPY  "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.8800 A22 1.3200")
                elif "HG" in types[attached_1[1]-1]: # Thiol H last
                    atom3=name[attached_1[1]-1]
                    atom2=name[attached_1[0]-1]
                    print("LONEPAIR relative LP"+str(lp_index)+"A "+atom1+" "+atom2+" "+atom3+" distance 0.8 angle 95.0 dihe 100.0")
                    print("LONEPAIR relative LP"+str(lp_index)+"B "+atom1+" "+atom2+" "+atom3+" distance 0.8 angle 95.0 dihe 260.0")
                    print("ANISOTROPY  "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.8800 A22 1.3200")
                else: # Dialkylsulfide
                    atom2=name[attached_1[1]-1]
                    atom3=name[attached_1[0]-1]
                    print("LONEPAIR bisector LP"+str(lp_index)+"A "+atom1+" "+atom2+" "+atom3+" distance 0.70 angle  95.0 dihe 100.0")
                    print("LONEPAIR bisector LP"+str(lp_index)+"B "+atom1+" "+atom3+" "+atom2+" distance 0.70 angle  95.0 dihe 100.0")
                    print("ANISOTROPY  "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.7808 A22 1.3662")
            if types[current_list[i]]=='SG301': # Dialkyldisulfide
                atom2=name[attached_1[1]-1]
                atom3=name[attached_1[0]-1]
                print("LONEPAIR bisector LP"+str(lp_index)+"A "+atom1+" "+atom2+" "+atom3+" distance 0.90 angle 110.0 dihe 90.0")
                print("LONEPAIR bisector LP"+str(lp_index)+"B "+atom1+" "+atom3+" "+atom2+" distance 0.90 angle 110.0 dihe 90.0")
                if "CG" in types[attached_1[1]-1]:
                    print("ANISOTROPY  "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.9208 A22 1.0102")
                else:
                    print("ANISOTROPY  "+atom1+" "+atom3+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.9208 A22 1.0102")
            if types[current_list[i]]=='SG302': #Thiolate
                atom2=name[attached_1[0]-1]
                atom3=name[attached_2[0]-1]
                print("LONEPAIR relative LP"+str(lp_index)+"A "+atom1+" "+atom2+" "+atom3+"  distance 0.45 angle 180.0 dihe 180.0")
        if lists[k]==oxygenlist:
            if types[current_list[i]]=='OG311': # Alcohol
                if 'HG' in types[attached_1[0]-1]:
                    atom3=name[attached_1[0]-1]
                    atom2=name[attached_1[1]-1]
                else:
                    atom3=name[attached_1[1]-1]
                    atom2=name[attached_1[0]-1]
                print("LONEPAIR relative LP"+str(lp_index)+"A "+atom1+" "+atom2+" "+atom3+" distance 0.35 angle 110.9 dihe  91.0")
                print("LONEPAIR relative LP"+str(lp_index)+"B "+atom1+" "+atom2+" "+atom3+" distance 0.35 angle 110.9 dihe 269.0")
                print("ANISOTROPY  "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.8108 A22 1.2162")
            if types[current_list[i]]=='OG302': # Ester
                if "CG2O" in types[attached_1[0]-1]:
                    atom2=name[attached_1[0]-1]
                    atom3=name[attached_1[1]-1]
                else:
                    atom2=name[attached_1[1]-1]
                    atom3=name[attached_1[0]-1]
                print("LONEPAIR relative LP"+str(lp_index)+"A "+atom1+" "+atom2+" "+atom3+" distance 0.35 angle 110.9 dihe  91.0")
                print("LONEPAIR relative LP"+str(lp_index)+"B "+atom1+" "+atom2+" "+atom3+" distance 0.35 angle 110.9 dihe 269.0")
                print("ANISOTROPY  "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.8108 A22 1.2162")
            if types[current_list[i]]=='OG2D2' or types[current_list[i]]=='OD2N1': # deprotonated acid
                atom2=name[attached_1[0]-1]
                if 'CG' in types[attached_2[0]-1]:
                    atom3=name[attached_2[0]-1]
                else:
                    atom3=name[attached_2[1]-1]
                print("LONEPAIR relative LP"+str(lp_index)+"A "+atom1+" "+atom2+" "+atom3+" distance 0.35 angle 110.0 dihe   0.0")
                print("LONEPAIR relative LP"+str(lp_index)+"B "+atom1+" "+atom2+" "+atom3+" distance 0.35 angle 110.0 dihe 180.0")
                print("ANISOTROPY "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.7229 A22 1.265")
            if types[current_list[i]]=='OG2D1' or types[current_list[i]]=='OG2D3' or types[current_list[i]]=='OG2D4': #carbonyl O
                atom2=name[attached_1[0]-1]
                atom3=name[attached_2[0]-1]
                print("LONEPAIR relative LP"+str(lp_index)+"A "+atom1+" "+atom2+" "+atom3+" distance 0.30 angle 91.0 dihe   0.0")
                print("LONEPAIR relative LP"+str(lp_index)+"B "+atom1+" "+atom2+" "+atom3+" distance 0.30 angle 91.0 dihe 180.0")
                if 'HG' in types[attached_2[0]-1] or 'HG' in types[attached_2[1]-1]:
                    if 'CG' in types[attached_2[0]-1] or 'CG' in types[attached_2[1]-1]: #Aldehyde
                        print("ANISOTROPY "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.85956 A22 1.13122")
                    elif 'NG' in types[attached_2[0]-1] or 'NG' in types[attached_2[1]-1]: # Formamide
                        print("ANISOTROPY "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.819 A22 1.183")
                elif 'NG' in types[attached_2[0]-1] or 'NG' in types[attached_2[1]-1]: #Amide
                        print("ANISOTROPY "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.82322 A22 1.14332")
                elif 'CG' in types[attached_2[0]-1] and 'CG' in types[attached_2[1]-1]: #Ketone
                    print("ANISOTROPY "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.86650 A22 1.07246")
                elif 'OG' in  types[attached_2[0]-1] or 'OG' in types[attached_2[1]-1]: #Acid or ester
                    print("ANISOTROPY "+atom1+" "+atom2+" LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.6968  A22 1.2194")

            if types[current_list[i]]=='OG301' or types[current_list[i]]=='OG2R50' or types[current_list[i]]=='OG3R60' or types[current_list[i]]=='OG3C31' or types[current_list[i]]=='OG3C51' or types[current_list[i]]=='OG3C61': #Ether
                atom2=name[attached_1[0]-1]
                atom3=name[attached_1[1]-1]
                print("LONEPAIR bisector LP"+str(lp_index)+"A "+atom1+" "+atom2+" "+atom3+" distance 0.35 angle 110.0 dihe  90.0")
                print("LONEPAIR bisector LP"+str(lp_index)+"B "+atom1+" "+atom2+" "+atom3+" distance 0.35 angle 110.0 dihe 270.0")
                print("LONEPAIR bisector LP"+str(lp_index)+"X "+atom1+" "+atom2+" "+atom3+" distance 0.10 angle   0.0 dihe   0.0")
                print("!LP"+str(lp_index)+"X is a dummy LP for anisotropy.")
                print("ANISOTROPY "+atom1+" LP"+str(lp_index)+"X LP"+str(lp_index)+"A LP"+str(lp_index)+"B A11 0.8889 A22 1.2222")
        lp_index+=1

        
