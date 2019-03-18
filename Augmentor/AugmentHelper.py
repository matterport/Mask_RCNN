from Augmentor_phase1 import *
from Augmentor_phase2 import *
from Augmentor_phase2_5_renamer import *
from Augmentor_phase3 import *

class AugmentHelper(object):
    def __init__(self, src_dataset_path, output_dataset_path):
        self.src_dataset_path = src_dataset_path
        self.output_dataset_path = output_dataset_path
    
    def configure(self):
        pass

    def generate(self):
        # simon - todo: finish implementation
        gt_path = "/home/simon/Mask_RCNN/cucu_train/project_dataset/train_ground_truth/" #self.src_dataset_path + "../train_ground_truth/"
        augmented_output_path = self.src_dataset_path + "../output_phase2/"
        augment_create_mask_files(self.src_dataset_path, self.src_dataset_path + "annotations.json", gt_path)
        augment_perform_pipe(self.src_dataset_path, gt_path)
        augment_batch_rename(augmented_output_path)
        augment_reconstruct_json(augmented_output_path, self.src_dataset_path + "annotations.json")

augmentor = AugmentHelper("/home/simon/Mask_RCNN/cucu_train/project_dataset/real_train_data/", "")
augmentor.generate()


