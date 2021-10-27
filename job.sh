#!/bin/bash
## declare an array variable
#declare -a arr=(0.7 0.6 0.5 0.4 0.3 0.2)
declare -a loop=(9 8 7 6 5)

## now loop through the above array
for n in "${loop[@]}"
do
echo "extracting uncertainties from $n utterances enrolled speaker models"
python uncertainty_for_one.py "$n"
#   for i in "${arr[@]}"
#   do
#      python uncertainty_extraction.py "$i" "$n"
#   done
done

cp -r results/ ../../../../srv/scratch/z5195063/