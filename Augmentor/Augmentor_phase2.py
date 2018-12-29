import Augmentor

# this dir must to contains input_dir (the name does not mather) + ground_truth dir
path = "/Users/orshemesh/Desktop/Project/Leaves_augmentor/"

# output dir will be in <path>/output
p = Augmentor.Pipeline(path, output_directory="leaves_after_phase2/")

p.ground_truth(path+"leaves_after_phase1/")

p.flip_left_right(probability=0.4)
# p.zoom_random(probability=0.2, percentage_area=0.85) # big percentage_area -> less zoom
# p.crop_random(probability=0.2, percentage_area=0.85) # big percentage_area -> less crop
p.flip_top_bottom(probability=0.2)
p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
# p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
# p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
# p.random_distortion(probability=1, grid_height=250, grid_width=250, magnitude=8)
# p.random_brightness(probability=0.5, min_factor=0.95, max_factor=1.05)
# p.resize(probability=1.0, width=512, height=512)
# p.gaussian_distortion(probability=0.2,grid_width = , grid_height= , magnitude= , corner= , method= )

p.set_save_format(save_format="PNG")
p.sample(1, multi_threaded=False)
p.process()
