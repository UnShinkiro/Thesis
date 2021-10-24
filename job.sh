#!/bin/bash
## declare an array variable
declare -a arr=(0.7 0.6 0.5 0.4 0.3 0.2)
declare -a loop=(20 15 10)

## now loop through the above array
for n in "${loop[@]}"
do
echo "Uncertainty extraction for $n utterances model"
   for i in "${arr[@]}"
   do
      python uncertainty_extraction.py "$i" "$n"
   done
done

cp -r results/ ../../../../srv/scratch/z5195063/