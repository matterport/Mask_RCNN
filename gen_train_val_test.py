import os.path
import os
import random

train = 0.5
val = 0.5

train_output = 'train.txt'
val_output = 'val.txt'
rootdir = 'D:\RCNN\matlab_for_preprocess\image&mask_delete_the_border\\images\\'  # 指明被遍历的文件夹
maskdir = 'D:\RCNN\matlab_for_preprocess\image&mask_delete_the_border\\masks\\'  # 指明被遍历的文件夹


# # rename the chinese file to index number for general
# for _, _, filenames in os.walk(rootdir):
#     count = 1
#     for filename in filenames:
#         os.rename(rootdir + filename, rootdir + '%d.bmp'%count)
#         os.rename(maskdir + filename, maskdir + '%d.bmp'%count)
#         count = count + 1


# # gen the train test val file list
for _, _, filenames in os.walk(rootdir):  # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
    random.shuffle(filenames)
    print("gen train file for %f of the total file count %d" %(train, len(filenames)))
    with open(train_output, 'w') as train_file:
        for filename in filenames[0:round(len(filenames) * train)]:
            train_file.write(filename + '\n')
    print("gen val file for %f of the total file count %d" %(val, len(filenames)))
    with open(val_output, 'w') as val_file:
        for filename in filenames[round(len(filenames) * train):-1]:
            val_file.write(filename + '\n')
