#!/bin/bash
## declare an array variable
declare -a arr=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7)

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
done