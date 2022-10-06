#!/bin/bash

for seed in "$@"
do
  python explore/training.py --seed=$seed --loss='AnscombeMSE' --note=mse
done
