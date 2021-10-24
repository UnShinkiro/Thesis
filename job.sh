#!/bin/bash
## declare an array variable
declare -a arr=(20 15 10)

## now loop through the above array
for i in "${arr[@]}"
do
   python uncertainty_for_one.py "$i"
done

cp -r results/ ../../../../srv/scratch/z5195063/