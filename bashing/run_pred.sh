#!/bin/bash

drudeDir=/home/pchatter/payal/es_predictor/drude_pdbs
psflist=$(for i in ${drudeDir}/*.psf;do echo $i | cut -f1 -d'.';done)
#echo $psflist
for psf in $psflist;do
  name=$(echo $psf | cut -f7 -d'/')
  echo " processing $name"
  python predictor.py $psf.psf d2c.dict $name  
done
