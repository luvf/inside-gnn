#!/bin/bash
f[0]="ba2_activation"
f[1]="Aids_activation"
f[2]="Mutag_activation"
f[3]="Bbbp_activation"

q=3
for fichier in ${f[*]}
do
    echo $fichier
    for i in `seq 0 $q`;
    do
	echo $i
	python3 pretraitement.py -i $fichier".csv" -l $i
    done

    for i in `seq 0 $q`;
    do
	echo $i
	python3 si_activation_pattern.py -i $fichier"_pretraite_"$i".csv" -l $i -k 1 -s 1 -m 0
	python3 si_activation_pattern.py -i $fichier"_pretraite_"$i".csv" -l $i -k 5 -s 1 -m 1
    done

    m=$(($q+1))
    python3 posttraitement.py -i $fichier".csv" -l $m
    python3 encodage.py -i $fichier".csv" -j $fichier"_motifs.txt"

done
