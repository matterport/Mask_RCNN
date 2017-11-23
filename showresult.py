import visualize
import pickle
import os
import skimage.io
import skimage.transform
mask_path = 'D:\RCNN\matlab_for_preprocess\image&mask_delete_the_border\masks'
for parent, dir, filenames in os.walk('result'):
    for filename in filenames:
        print(filename)
        mask_name = filename.split(".")[0] + '.bmp'
        ori_mask_image = skimage.io.imread(os.path.join(mask_path, mask_name))
        ori_mask_image = skimage.transform.resize(ori_mask_image, [1024, 1024])
        with open('result/%s'%filename, 'rb') as f:
            image, r, class_names = pickle.load(f)
            class_names = ['bg', 'mass']
            print(image.shape)
            visualize.display_instances(ori_mask_image, image, r['rois'], r['masks'], r['class_ids'],
                                         class_names, r['scores'])


# for i in range(240):
#     try:
#     except Exception:
#         pass