#!/bin/bash
## declare an array variable
declare -a arr=(0.52 0.54 0.56 0.58 0.6)

## now loop through the above array
for i in "${arr[@]}"
do
   python uncertainty_for_one.py "$i"
done

cp -r results/ ../../../../srv/scratch/z5195063/