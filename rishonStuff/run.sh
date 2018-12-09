#!/bin/bash
#SBATCH -N 1 # number of minimum nodes
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --job-name="simonTheBro"
#SBATCH -o slurm.%N.%j.out # stdout goes here
#SBATCH -e slurm.%N.%j.out # stderr goes here
source activate cucuEnv
python3 /home/simon.lousky/Mask_RCNN/cucu_train/cucuTrain.py
