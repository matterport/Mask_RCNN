import visualize
import pickle
f = open('H:\maskrcnn\data\images\\74.dat', 'rb')
image, r, class_names = pickle.load(f)
print(image.shape)
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
pass