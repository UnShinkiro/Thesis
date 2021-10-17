#!/bin/bash
#cd ../../../srv/scratch/z5195063/
#cp -r saved_model/ $TMPDIR
#cp -r results/ $TMPDIR
#cp -r d-vector/ $TMPDIR
#cp test_utterance.pkl $TMPDIR
#cp vox/vox1_test_wav.tar $TMPDIR
#cd $TMPDIR
#cp -r ~/Thesis/ .
#tar xf vox1_test_wav.tar
#mkdir Thesis/vox/
#mv saved_model/ Thesis/
#mv results/ Thesis/
#mv d-vector/ Thesis/
#mv test_utterance.pkl Thesis/
#mv vox1_test_wav/ Thesis/vox/
#cd Thesis
#source ~/.venvs/Thesis/bin/activate

## declare an array variable
declare -a arr=(25 20 15 10)

## now loop through the above array
for i in "${arr[@]}"
do
   python speaker_enrollment.py "$i"
done

cp -r d-vector/ ../../../../srv/scratch/z5195063/