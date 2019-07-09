#!/bin/bash

drudeDir=drude_pdbs
psflist=$(for i in ${drudeDir}/*.psf;do echo $i | cut -f1 -d'.';done)
#echo $psflist
for psf in $psflist;do
  echo " processing $psf"
  python create_lp.py ${psf}_start.pdb $psf.psf 
done
