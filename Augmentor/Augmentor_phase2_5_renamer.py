import os

# dir_path = '/Users/orshemesh/Desktop/Project/augmented_cucumbers/origin/output_phase2/'

# rename all originals and masks to simple names 
def augment_batch_rename(dir_path):

    files_in_dir = os.listdir(dir_path)
    augmented_image_names = [file for file in files_in_dir if file.find('original_IMG') != -1]

    ground_truth_images = [file for file in files_in_dir if file.find('groundtruth') != -1]
    i = 0
    for g_img in ground_truth_images:
        g_img_orig = "IMG"+g_img.split("IMG")[1]
        g_base_name = g_img_orig.split(".")[0]
        f = True
        for a_img in augmented_image_names:
            a_img_orig = "IMG" + a_img.split("IMG")[1]
            if g_base_name in a_img_orig:
                i += 1
                a_new_name = a_img_orig.split('.')[0] + '_' + str(i) + '.' + a_img_orig.split('.')[2]
                g_new_name = 'ground_truth_' + g_img_orig.split('.')[0] + '_' + str(i) + '.' + g_img_orig.split('.')[2]
                print(a_new_name)
                print(g_new_name)
                os.rename(os.path.join(dir_path, g_img), os.path.join(dir_path, g_new_name))
                os.rename(os.path.join(dir_path, a_img), os.path.join(dir_path, a_new_name))

                f = False
        if f:
            print("doesn't find match image to: {}".format(g_img))