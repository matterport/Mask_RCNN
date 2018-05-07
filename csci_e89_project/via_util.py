"""
Mask R-CNN
Common utility functions for handling via generated annotations.
Supports merging separate json files and copying those images with defined
regions.

"""

import json
import collections
import shutil


def sort_annotations(annotations):
    ordered_annotations = collections.OrderedDict(sorted(annotations.items()))
    valid_annotations = {}
    filenames = []
    for k, a in ordered_annotations.items():
        if a['regions']:
            filenames.append(a['filename'])
            valid_annotations[k] = a
    return valid_annotations, filenames


def merge_annotations(s1, s2, m):
    a1 = json.load(open(s1))
    a2 = json.load(open(s2))

    z = { **a1, **a2 }
    z, filenames = sort_annotations(z)
    with open(m, 'w') as outfile:
        json.dump(z, outfile)

    return filenames

# filenames = merge_annotations('/tmp/annotations.json', '/tmp/wolf_11.json', '/tmp/merged.json')

source_dir = '../images/imgnet_n02114100/images/'
target_dir = '../images/imgnet_n02114100/train/'

def merge_images( original_via, new_via, source_image_dir, target_image_dir):

    filenames = merge_annotations(original_via, new_via, '/tmp/merged.json')

    # Copy all the files from source to the target, note it will
    # overwrite any that exist in the target.
    for f in filenames:
        shutil.copy(source_dir+f, target_dir+f)