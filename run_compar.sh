#!/bin/bash

drudeDir=drude_pdbs
predlp=predlp
initpdb=$(for i in ${drudeDir}/*.pdb;do echo $i | cut -f2 -d'/' | cut -f1 -d'_';done)
#echo $psflist
for pdb in $initpdb;do
  echo " processing $pdb"
  python compare.py ${drudeDir}/${pdb}_start.pdb ${predlp}/${pdb}_predlp.pdb 
done
