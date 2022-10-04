#!/bin/bash

for seed in "$@"
do
  python explore/training.py --seed=$seed
done
