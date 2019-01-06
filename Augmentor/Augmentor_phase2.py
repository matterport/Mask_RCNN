import sys

sys.path.append("/Users/orshemesh/Desktop/project repo/alteredAugmentor")  # To find local version of the library

import Augmentor

# this dir must to contains input_dir (the name does not mather) + ground_truth dir
path = "/Users/orshemesh/Desktop/Project/Leaves_augmentor/"

# output dir will be in <path>/output
p = Augmentor.Pipeline(path+'leaves_images/', output_directory="output/", save_format="PNG")

p.ground_truth(path+'ground_truth_directory')

p.flip_left_right(probability=0.4)
p.flip_top_bottom(probability=0.2)
p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
p.random_distortion(probability=1, grid_height=7, grid_width=7, magnitude=7)
p.gaussian_distortion(probability=0.2, grid_width = 7, grid_height=7, magnitude= 7, corner= "bell", method="in")
p.random_erasing(probability=0.2, rectangle_area=250)
p.random_contrast(probability=0.2, min_factor=0.2, max_factor=0.6)
p.random_color(probability=0.2, min_factor=0.25, max_factor=0.8)
p.random_brightness(probability=0.2, min_factor=0.25, max_factor=0.8)
p.black_and_white(probability=0.015)
p.greyscale(probability=0.015)

p.set_save_format(save_format="PNG")
# p.sample(1, multi_threaded=True)
p.process()
