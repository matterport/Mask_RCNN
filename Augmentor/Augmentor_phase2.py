import sys
import Augmentor as alteredAugmentor # pip install git+https://github.com/simonlousky/alteredAugmentor.git
# sys.path.append("/Users/orshemesh/Desktop/project repo/alteredAugmentor")  # To find local version of the library
# import Augmentor
# path = "/Users/orshemesh/Desktop/Project/augmented_leaves/"
# this dir must to contains input_dir (the name does not mather) + ground_truth dir

def augment_perform_pipe(src_path, ground_truth_path, destination_path):

    # output dir will be in <path>/output
    p = alteredAugmentor.Pipeline(src_path, output_directory=destination_path, save_format="JPEG")

    p.ground_truth(ground_truth_path)

    # p.flip_left_right(probability=0.4)
    # p.flip_top_bottom(probability=0.2)
    # p.rotate(probability=0.3, max_left_rotation=25, max_right_rotation=25)
    # p.random_color(probability=0.3, min_factor=0.7, max_factor=1.3)
    # p.random_contrast(probability=0.4, min_factor=0.7, max_factor=1.3)
    # p.random_brightness(probability=0.3, min_factor=0.7, max_factor=1.3)
    # p.skew(probability=0.1, magnitude=0.1)
    # p.shear(probability=0.2, max_shear_left=3, max_shear_right=3)
    p.resize(probability=1.0, width=1536, height=1024)

    # p.gaussian_distortion(probability=0.4, grid_width = 700, grid_height=700, magnitude= 8, corner= "ul", method="in") very easy with big grid!!!!
    # p.random_distortion(probability=1.0, grid_height=400, grid_width=600, magnitude=7)
    # p.sample(5000)
    # p.set_save_format(save_format="PNG")

    p.process()
