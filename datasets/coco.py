import os
from datasets.registable_dataset import RegisterableDataset
from detectron2.data.datasets.coco import load_coco_json

YELLOW_RGB = (255,211,67)
BLUE_RGB = (0,0,139)
class CocoClassAgnostic(RegisterableDataset):
    def __init__(self, name, json_ann, image_root, coco_path="datasets/coco"):
        self.coco_path = coco_path
        self.name_ = name
        self.json_file_ = f"{coco_path}/annotations/{json_ann}"
        self.image_root_ = f"{coco_path}/{image_root}"
    def get_dataset_dicts(self):
        data = load_coco_json(self.json_file_, self.image_root_)
        for sample in data:
            for annotation in sample['annotations']:
                # we use 1 as the category id for foreground
                annotation['category_id'] = 1
        return data
    @property
    def name(self):
        return self.name_

    @property
    def json_file(self):
        return self.json_file_

    @property
    def image_root(self):
        return self.image_root_

    @property
    def evaluator_type(self):
        return 'coco'

    @property
    def thing_colors(self):
        return [list(BLUE_RGB), list(YELLOW_RGB)]

    @property
    def thing_classes(self):
        return ['foreground']

    @property
    def thing_dataset_id_to_contiguous_id(self):
        return {0: 0}



