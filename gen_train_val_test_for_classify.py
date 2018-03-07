import os.path
import os
import random

train = 0.666
val = 0.0
test = 1 - train

label = '/home/liuml/maskrcnn/data/label.txt'  # 标签数据
rootdir = '/home/liuml/maskrcnn/data/images/'  # 指明被遍历的文件夹
maskdir = '/home/liuml/maskrcnn/data/masks/'  # 指明被遍历的文件夹
train_output = 'train.txt'
val_output = 'val.txt'
test_output = 'test.txt'

# # gen the train test val file list
with open(label, encoding="gbk") as lf:
    lines = list(lf.readlines())
    random.shuffle(lines)
    print("gen train file for %f of the total file count %d" %(train, len(lines)))
    with open(train_output, 'w') as train_file:
        for filename in lines[0:round(len(lines) * train)]:
            # print(filename)
            train_file.write(filename)
    print("gen val file for %f of the total file count %d" %(val, len(lines)))
    with open(val_output, 'w') as val_file:
        for filename in lines[round(len(lines) * train):round(len(lines) * (train + val))]:
            val_file.write(filename)
    print("gen test file for %f of the total file count %d" %(val, len(lines)))
    with open(test_output, 'w') as test_file:
        for filename in lines[round(len(lines) * (train + val)): -1]:
            test_file.write(filename)
