import os
import json
import argparse
import datetime
import copy

RESULT_FILE = '/'.join([
    "..",
    "data",
    "annotations.json"
])


def create_merge_annotation(first_anno, first_folder):
    merged_anno = copy.deepcopy(first_anno)

    merged_anno['info']['duplicate_filenames'] = []
    merged_anno['info']['missing_annotations'] = []

    #  Sets of the image that has an annotation
    first_anno_annotations = set([int(i['image_id']) for i in first_anno['annotations']])
    # Sets of filename of annotated images
    first_anno_filenames = set(['/'.join([first_folder, i['file_name']]) for i in first_anno['images']])

    for img in merged_anno['images']:
        img['file_name'] = '/'.join([first_folder, img['file_name']])

        if int(img['id']) not in first_anno_annotations:
            merged_anno['info']['missing_annotations'].append(img['file_name'])

    return merged_anno, first_anno_filenames


def merge_annotation(merged_anno, merged_original_filenames, second_anno, second_folder):
    merged_anno_len_img = max(set([int(i['image_id']) for i in merged_anno['annotations']])) + 1
    merged_anno_len_anno = max(set([int(i['id']) for i in merged_anno['annotations']])) + 1

    merged_anno_mapping_categories = {
        i['name']: i['id'] for i in merged_anno['categories']
    }
    second_anno_mapping = {
        i['id']: i['name'] for i in second_anno['categories']
    }

    # Parse second annotations images
    second_anno_annotations = set([int(i['image_id']) for i in second_anno['annotations']])
    for img in second_anno['images']:

        # Change filename to include folder
        img['file_name'] = '/'.join([second_folder, img['file_name']])

        old_img_id = int(img['id'])
        img['id'] = int(img['id']) + merged_anno_len_img
        merged_anno['images'].append(img)

        # Check if filename was already present for possible duplicates
        # else add the filename to the set
        if img['file_name'] in merged_original_filenames:
            merged_anno['info']['duplicate_filenames'].append(img['file_name'])
        else:
            merged_original_filenames.add(img['file_name'])

        # Check if annotation is missing
        if int(old_img_id) not in second_anno_annotations:
            merged_anno['info']['missing_annotations'].append(img['file_name'])

        for anno in second_anno['annotations']:
            if anno['image_id'] == old_img_id:
                # Increment IDs
                anno['id'] = int(anno['id']) + merged_anno_len_anno
                anno['image_id'] = img['id']

                # Map category name to same ID
                anno['category_id'] = merged_anno_mapping_categories[second_anno_mapping[anno['category_id']]]
                merged_anno['annotations'].append(anno)

    return merged_anno, merged_original_filenames


def load_annotations(anno_path):
    return json.load(open(anno_path)), os.path.split(os.path.dirname(anno_path))[-1]


def remove_images_without_anno(merged_anno):
    merged_anno['images'] = [i for i in merged_anno['images'] \
                             if i['file_name'] not in merged_anno['info']['missing_annotations']]
    return merged_anno


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Merge annotation files')

    parser.add_argument('--annotations', required=True,
                        metavar="/path/to/first/annotations.json, /path/to/second/anno.json",
                        help="Comma separated list of paths to annotations")

    parser.add_argument('--result_file', required=False,
                        default=RESULT_FILE,
                        metavar="annotations.json",
                        help="Result file name")

    args = parser.parse_args()
    print("Annotations: ", args.annotations)
    print("Results: ", args.result_file)

    anno_paths = args.annotations.split(",")
    print("Current Annotation: " + anno_paths[0])

    first_anno, first_folder = load_annotations(anno_paths[0])
    merged_anno, merged_original_filenames = create_merge_annotation(first_anno, first_folder)

    for a in anno_paths[1:]:
        print("Current Annotation: " + a)
        second_anno, second_folder = load_annotations(a)
        merged_anno, merged_original_filenames = merge_annotation(merged_anno, merged_original_filenames, second_anno,
                                                                  second_folder)

    json.dump(remove_images_without_anno(merged_anno), open(args.result_file, "w+"))
