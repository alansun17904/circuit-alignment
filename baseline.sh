#!/bin/bash

models=(gpt2-small gpt2-medium gpt2-large gpt2-xl)

for mname in "${models[@]}"
do
	python3 baseline2.py $mname gpt2 "data/align-benchmark/$mname-lt.pkl" --batch_size 64
done
