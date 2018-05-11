import json
import os
import cv2


def preprocess(dataset_dir, src_file, dst_file='/tmp/annotations.json', validate=True):
    """
    Scans a VIA generated annotations file and inserts height and width.
    :param dataset_dir: Directory containing the data set and annotation file.
    :param src_file: Annotation file name.
    :param dst_file: File to write the updates to.
    :param validate: If trued runs validation checks on the annotation file.
    """

    annotations_filepath = os.path.join(dataset_dir, src_file)

    with open(annotations_filepath, 'r') as f:
        annotations = json.load(f)

    for k, a in annotations.items():

        # Set shape so we don't have during training.
        image_path = os.path.join(dataset_dir, a['filename'])
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError("Unable to read: ", image_path)

        print(image_path, type(image))
        height, width = image.shape[:2]

        a['height'] = height
        a['width'] = width

        # Perform some sanity checking and corrections.
        if validate:
            # Ensure polygon points fall within the image shape
            for r_idx, r in a['regions'].items():

                # Make sure x points are in bound
                all_points_x = r['shape_attributes']['all_points_x']
                for x_idx, x in enumerate(all_points_x):
                    if x > width:
                        print('Correcting all_points_x for ', a['filename'])
                        all_points_x[x_idx] = width - 1

                # Make sure y points are in bound
                all_points_y = r['shape_attributes']['all_points_y']
                for y_idx, y in enumerate(all_points_y):
                    if y > height:
                        print('Correcting all_points_x for ', a['filename'])
                        all_points_y[y_idx] = height - 1

    with open(dst_file, 'w') as f:
        json.dump(annotations, f)


def main():

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Pre-processing via annotation file.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--annotations', required=True,
                        metavar="annotations.json",
                        default="annotations.json",
                        help="annotations file name")

    args = parser.parse_args()
    print("Annotations: ", args.annotations)
    print("Dataset: ", args.dataset)

    preprocess(args.dataset, args.annotations)


if __name__ == "__main__":
    main()
