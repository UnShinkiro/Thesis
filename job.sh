#!/bin/bash
## declare an array variable
declare -a arr=(25 20 15 10)

## now loop through the above array
for i in "${arr[@]}"
do
   python speaker_enrollment.py "$i"
done

cp -r d-vector/ ../../../../srv/scratch/z5195063/