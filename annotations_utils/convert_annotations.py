import os
import json
import argparse

import mapping

RESULT_FILE = "annotations_converted.json"


def convert_annotations(old_anno, mapping_name="TACO"):
    mapping_function = getattr(mapping, '{}_category_mapping'.format(mapping_name))

    for c in old_anno['categories']:
        c['name'] = mapping_function(c['name'])

    return old_anno


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert annotations')

    parser.add_argument('--old_annotations', required=True,
                        metavar="/path/to/annotations.json",
                        help="Annotations to convert")

    parser.add_argument('--result_file', required=False,
                        default=RESULT_FILE,
                        metavar="annotations.json",
                        help="Result file name")

    parser.add_argument('--mapping_name', required=False,
                        default="TACO",
                        metavar="TACO",
                        help="Mapping function")

    args = parser.parse_args()
    print("Annotations: ", args.old_annotations)
    print("Results: ", args.result_file)
    print("Mapping name: ", args.mapping_name)

    old_anno = json.load(open(args.old_annotations))
    new_anno = convert_annotations(old_anno, args.mapping_name)

    result_path = os.path.join(os.path.dirname(args.old_annotations), args.result_file)
    json.dump(new_anno, open(result_path, "w"))
