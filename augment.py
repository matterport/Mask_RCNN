### See https://github.com/aleju/imgaug and http://imgaug.readthedocs.io/en/latest/index.html
from imgaug import augmenters as iaa
import numpy as np
import imageio
import os
import argparse

# python3.6 augment.py -id /path/to/img/folder -od /path_to_output_dir > ./augmentation.log &
#
parser = argparse.ArgumentParser(description='Create an augmented data set.')
parser.add_argument('-id', '--input_dir', type=str, help='Base directory of the images to be augmented.', required=True)
parser.add_argument('-od', '--output_dir', type=str, help='Base directory of the output.', required=True)
args = parser.parse_args()

# store images and related masks as dictionary
img_dict = {}

for image_name in os.listdir(args.input_dir):

    # get abs path to image
    img = "{}.png".format(os.path.join(os.path.abspath(args.input_dir), image_name, "images", image_name))

   # create list of mask files for image
    mask_file_list = []
    mask_folder = os.path.join(os.path.abspath(args.input_dir), image_name, "masks")
    for mask in os.listdir(mask_folder):
        mask_path = os.path.join(os.path.abspath(mask_folder), mask)
        mask_file_list.append(mask_path)

    img_dict[image_name] = {
        "image":img,
        "masks":mask_file_list
    }


# create the augmentation sequences
rotators = iaa.SomeOf((1,3), [
    iaa.Fliplr(1),  # flip horiozontally
    iaa.Flipud(1),  # Flip/mirror input images vertically
    iaa.Rot90(1)
    ], random_order=True)

transformers = iaa.SomeOf((1, 3), [
    iaa.Superpixels(p_replace=0.5, n_segments=64),  # create superpixel representation
    iaa.GaussianBlur(sigma=(0.0, 5.0)),
    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
    iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25), # distort pixels
    # either change the brightness of the whole image (sometimes
    # per channel) or change the brightness of subareas
    iaa.OneOf([
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        iaa.FrequencyNoiseAlpha(
            exponent=(-4, 0),
            first=iaa.Multiply((0.5, 1.5), per_channel=True),
            second=iaa.ContrastNormalization((0.5, 2.0))
        )
    ])
], random_order=True)

# go through all the images and create a set of augmented images and masks for each
for key, val in img_dict.items():

    img = val["image"]
    masks = val["masks"]
    base_masks = []

    # read in images and masks
    base_image = np.array(imageio.imread(img).astype(np.uint8))
    for mask in masks:
        out_mask = imageio.imread(mask).astype(np.uint8)
        base_masks.append(out_mask)

    # loop through and rotate each image three times
    counter = 0
    while counter < 4:

        # create transforms to image and masks
        rotator = rotators.to_deterministic()  # set same random augmentation for img and masks
        images_aug = rotator.augment_image(base_image) # rotate image
        masks_aug = rotator.augment_images(base_masks) # rotate matching masks

        # update counter
        counter += 1

        # create output directories
        output_dir = "{}/{}_rot{}".format(args.output_dir, key, counter)
        out_img_dir = "{}/{}".format(output_dir, "images")
        out_mask_dir = "{}/{}".format(output_dir, "masks")

        # Create results directories
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

        if not os.path.exists(out_mask_dir):
            os.makedirs(out_mask_dir)

        # save augmented image
        img_out = "{}/{}_rot{}.png".format(out_img_dir, key, counter)
        imageio.imwrite(img_out, images_aug)

        for i, mask in enumerate(masks_aug):
            mask_out = "{}/{}_rot{}_{}.png".format(out_mask_dir, key, counter, i)
            imageio.imwrite(mask_out, masks_aug[i])

    # loop through and augment each image three times, leaving masks untouched
    counter2 = 0
    while counter2 < 4:

        # create transforms to image and masks
        augmentor = transformers.to_deterministic()  # set same random augmentation for img
        images_aug = augmentor.augment_image(base_image)

        # update counter
        counter2 += 1

        # create output directories
        output_dir2 = "{}/{}_aug{}".format(args.output_dir, key, counter2)
        out_img_dir2 = "{}/{}".format(output_dir2, "images")
        out_mask_dir2 = "{}/{}".format(output_dir2, "masks")

        # Create results directories
        if not os.path.exists(output_dir2):
            os.makedirs(output_dir2)

        if not os.path.exists(out_img_dir2):
            os.makedirs(out_img_dir2)

        if not os.path.exists(out_mask_dir2):
            os.makedirs(out_mask_dir2)

        # save augmented image
        img_out = "{}/{}_aug{}.png".format(out_img_dir2, key, counter2)
        imageio.imwrite(img_out, images_aug)

        for i, mask in enumerate(base_masks):
            mask_out = "{}/{}_aug{}_{}.png".format(out_mask_dir2, key, counter2, i)
            imageio.imwrite(mask_out, base_masks[i])



