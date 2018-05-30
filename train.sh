#!/bin/bash

# written by www.github/GustavZ

echo "infinite training"
export ROOT_DIR=$(pwd)
while [ true ]; do
    gnome-terminal -x sh -c "tensorboard --logdir=${ROOT_DIR}"
    python train_coco.py
    sleep 30
    killall python
    killal /usr/bin/python
    sleep 30
done
