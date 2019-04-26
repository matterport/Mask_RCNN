#!/bin/bash
#run this after starting the vm
sudo mkdir -p /az-ml-container
sudo mkdir -p /mnt/blobfusetmp
sudo chown -R ryan /mnt/blobfusetmp/
sudo chown -R ryan /az-ml-container/
blobfuse /az-ml-container --tmp-path=/mnt/blobfusetmp -o big_writes -o max_read=131072 -o max_write=131072 -o attr_timeout=240 -o fsname=blobfuse -o entry_timeout=240 -o negative_timeout=120 --config-file=/home/ryan/work/blobfuse.cfg

