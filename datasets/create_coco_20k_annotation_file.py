import os
import sys

# change working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# add project root to path
sys.path.insert(0, os.getcwd())


import argparse
from detectron2.data.datasets.coco import load_coco_json, convert_to_coco_json
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create COCO 20k annotation file")
    parser.add_argument("--coco-path", type=str, default="datasets/coco")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--not-class-agnostic", action="store_true")

    args = parser.parse_args()

    # Load COCO 2014 train
    json_file = os.path.join(args.coco_path, "annotations/instances_train2014.json")
    image_root = os.path.join(args.coco_path, "train2014")
    dataset_name = "coco_2014_train"

    # load coco 2014 train dataset
    print("Loading COCO 2014 train dataset from {}...".format(json_file))
    coco_dataset = load_coco_json(json_file, image_root, dataset_name)
    # load coco 20k filenames
    print("Loading COCO 20K files list...")
    with open("datasets/coco_20k_filenames.txt", "r") as f:
        # read lines and remove newline character
        coco_20k_filenames = [line.rstrip().split('/')[-1] for line in f.readlines()]
    # filter coco 2014 train annotations that are in coco 20k
    coco_20k_dataset = []
    for sample in tqdm(coco_dataset, desc="Filtering COCO_20K from COCO_2014_train"):
        image_file_name = sample["file_name"].split('/')[-1]
        if image_file_name in coco_20k_filenames:
            if not args.not_class_agnostic:
                for annotation in sample['annotations']:
                    # we use 1 as the category id for foreground
                    annotation['category_id'] = 1
            coco_20k_dataset.append(sample)

    # register coco 20k dataset
    if args.not_class_agnostic:
        coco_20k_name = "coco_20k"
    else:
        coco_20k_name = "coco_20k_CAD"
    def get_coco_20k_dicts():
        return coco_20k_dataset
    DatasetCatalog.register(coco_20k_name, get_coco_20k_dicts)
    MetadataCatalog.get(coco_20k_name).set(thing_classes=MetadataCatalog.get("coco_2014_train").thing_classes)

    # write coco 20k annotations to json file
    if args.output_path == "":
        output_path = os.path.join(args.coco_path, f"annotations/instances_{coco_20k_name}.json")
    else:
        output_path = args.output_path

    print("Writing COCO 20K annotations to {}...".format(output_path))
    convert_to_coco_json(coco_20k_name, output_path)

    # make sure coco path is in datasets project folder
    if args.coco_path != "datasets/coco":
        print("Creating symlink to coco dataset in datasets/coco project folder...")
        os.symlink(args.coco_path, "datasets/coco")

    print("Done!")
