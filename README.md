# respwithlp

This is python script for calculating resp charges on atoms along with its lone pairs.
It takes in mol2 file or pdb with associated psf file. 
Associates lone pairs to relevant atoms.
Gets resp charges on atomic position along with the lone pair positions as determined by script.
It uses python3 and psi4. (Install both of them in a conda env and activate it before running the script)

How to run the script

python respwlp.py -h

python respwlp.py -mol2 /path/to/mol2.file -c charge -m multiplicity -lot scf -basis 6-31G -dir /path/to/outputfiles 
