### See https://github.com/aleju/imgaug and http://imgaug.readthedocs.io/en/latest/index.html
import imgaug
from imgaug import augmenters as iaa
import numpy as np
import glob
import imageio
import os
import argparse
import shutil
import re

NUMBER_OF_IMAGE_CHANNELS = 3
NUMBER_OF_MASK_CHANNELS = 1

MASK_PIXEL_THRESHOLD = 0.8  # at least this proportion of the mask must be preserved in the augmentation

#
# source activate py35
# nohup python -u augmentation.py -id /data/dkpun-data/augmentor/input -od /data/vein/augmented/12-jun-ratio-20-1 -na 20 > ../augmentation.log &
#
parser = argparse.ArgumentParser(description='Create an augmented data set.')
parser.add_argument('-id', '--input_dir', type=str, help='Base directory of the images to be augmented.', required=True)
parser.add_argument('-od', '--output_dir', type=str, help='Base directory of the output.', required=True)
parser.add_argument('-na', '--number_of_augmented_images_per_original', type=int, default=1,
                    help='Minimum number of augmented image/mask pairs to produce for each input image/mask pair.',
                    required=False)
parser.add_argument('--augment_colour', dest='augment_colour', action='store_true', help='Apply colour augmentation')
parser.add_argument('--no-augment_colour', dest='augment_colour', action='store_false',
                    help='Do not apply colour augmentation')
parser.set_defaults(augment_colour=True)
args = parser.parse_args()

# pull out list of image folder names

image_file_list = []
mask_file_list = []
for x in os.listdir(args.input_dir):
    img_folder = os.path.join(os.path.abspath(args.input_dir), x, "images")
    for file in os.listdir(img_folder):
        if file.endswith(".png"):
            image_file_list.append(file)
    mask_folder = os.path.join(os.path.abspath(args.input_dir), x, "masks")
    for file in os.listdir(mask_folder):
        if file.endswith(".png"):
            mask_file_list.append(file)

# counter to make sure we have at least x augs per image
total_images_output = 0

# create the augmentation sequences
affine_seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Flipud(0.5),
    iaa.Affine(rotate=(-45, 45))  # rotate images between -45 and +45 degrees
], random_order=True)

no_rotation_affine_seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Flipud(0.5)
], random_order=True)

colour_seq = iaa.Sequential([
    iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5),
    # normalize contrast by a factor of 0.5 to 1.5, sampled randomly per image and for 50% of all images also independently per channel
    iaa.Multiply((0.9, 1.1), per_channel=0.5),
    # multiply 50% of all images with a random value between 0.5 and 1.5 and multiply the remaining 50% channel-wise, i.e. sample one multiplier independently per channel
    iaa.Add((-5, 5), per_channel=0.5)
    # add random values between -40 and 40 to images. In 50% of all images the values differ per channel (3 sampled value). In the other 50% of all images the value is the same for all channels
], random_order=True)

# go through all the images and create a set of augmented images and masks for each
for idx in image_file_list:

    augmented_images = []
    augmented_masks = []

    base_name = idx.split(".")[0]

    base_image = imageio.imread(base_name).astype(np.uint8)
    raise SystemExit("winning")
    base_mask = imageio.imread(mask_file_list[idx]).astype(np.uint8)

    base_mask_pixel_count = np.count_nonzero(base_mask)

    number_of_augmented_images_per_original = args.number_of_augmented_images_per_original
    if re.search('class_Pure_Quartz_Carbonate', base_name):
        number_of_augmented_images_per_original = number_of_augmented_images_per_original * 2

    images_list = []
    masks_list = []
    for i in range(number_of_augmented_images_per_original):
        images_list.append(base_image)
        masks_list.append(base_mask)

    # convert the image lists to an array of images as expected by imgaug
    images = np.stack(images_list, axis=0)
    masks = np.stack(masks_list, axis=0)

    # write out the un-augmented image/mask pair
    print("writing out the un-augmented image/mask pair")
    output_base_name = "{}_orig{}".format(os.path.splitext(base_name)[0], os.path.splitext(base_name)[1])
    imageio.imwrite("{}/{}".format(augmented_images_directory, output_base_name), base_image)
    imageio.imwrite("{}/{}".format(augmented_masks_directory, output_base_name), base_mask)
    total_images_output += 1

    number_of_augmentations_for_this_image = 0
    number_of_retries = 0

    while number_of_augmentations_for_this_image < number_of_augmented_images_per_original:
        # Convert the stochastic sequence of augmenters to a deterministic one.
        # The deterministic sequence will always apply the exactly same effects to the images.
        if number_of_retries == 0:
            affine_det = affine_seq.to_deterministic()  # call this for each batch again, NOT only once at the start
            images_aug = affine_det.augment_images(images)
            masks_aug = affine_det.augment_images(masks)
        else:
            affine_det = no_rotation_affine_seq.to_deterministic()  # call this for each batch again, NOT only once at the start
            images_aug = affine_det.augment_images(images)
            masks_aug = affine_det.augment_images(masks)

        # apply the colour augmentations to the images but not the masks
        if args.augment_colour == True:
            images_aug = colour_seq.augment_images(images_aug)

        # now write out the augmented image/mask pair
        print("writing out the augmented image/mask pair")
        for i in range(len(images_aug)):
            if np.count_nonzero(masks_aug[i]) > (base_mask_pixel_count * MASK_PIXEL_THRESHOLD):
                output_base_name = "{}_augm_{}{}".format(os.path.splitext(base_name)[0],
                                                         number_of_augmentations_for_this_image,
                                                         os.path.splitext(base_name)[1])
                imageio.imwrite("{}/{}".format(augmented_images_directory, output_base_name), images_aug[i])
                imageio.imwrite("{}/{}".format(augmented_masks_directory, output_base_name), masks_aug[i])
                number_of_augmentations_for_this_image += 1
                if number_of_augmentations_for_this_image == number_of_augmented_images_per_original:
                    break
            else:
                print("discarding image/mask pair {} - insufficient label".format(i + 1))
        print("completed {} augmentations for this image.".format(number_of_augmentations_for_this_image))
        number_of_retries += 1
    total_images_output += number_of_augmentations_for_this_image

print("augmented set of {} images generated from {} input images".format(total_images_output, len(image_file_list)))