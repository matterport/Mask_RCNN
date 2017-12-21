#!/bin/bash

for((i=1;i<=10;i++));
do
echo "experments: "$i >> result
python gen_train_val_test.py
mkdir experments/${i} | cp val.txt experments/${i} | cp train.txt experments/${i}
python train_us.py

for((j=1;j<=10;j++));
do
python test_us.py ${j} >> result
done

done
