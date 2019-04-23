#!/bin/bash



src="./src/training.py"
training="./data/icwb2-data/training/pku_training.utf8"
result="../pku_utf8.model"
time=100


time python $src $training $result $time


src="./src/testing.py"
model="../pku_utf8.model"
testing="./data/icwb2-data/testing/pku_test.utf8"
result="../pku_utf8.result"

time python $src $model $testing $result 

time python ./src/Viterbi.py -model ./data/train_01.txt ./data/test.txt
time python ./src/Viterbi.py -test ./data/test.txt ./data/pku_utf8.result ./data/result.txt ./data/tree.model

