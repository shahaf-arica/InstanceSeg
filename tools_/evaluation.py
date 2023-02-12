import os
import sys

# change working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# add project root to path
sys.path.insert(0, os.getcwd())

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tools_.registering import register_all_datasets
from detectron2.data import DatasetCatalog

import argparse

from detectron2.evaluation import COCOEvaluator, inference_on_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create pseudo labels mask coco annotation file")
    parser.add_argument("--ann-file", type=str, default="dataset/coco/annotations/instances_coco_20k.json", help="json file with coco annotations")
    parser.add_argument("--results-file", type=str, default="pseudo_labels/seg_masks.json", help="json file with detections")
    parser.add_argument("--iou_type", type=str, default="segm", help="type of coco iou evaluation")
    args = parser.parse_args()

    register_all_datasets()
    coco_eval = COCOEvaluator("coco_20k_class_agnostic")
    dataset_dicts = DatasetCatalog.get("coco_20k_class_agnostic")
    coco_gt = [dataset_dicts[0]]

    iou_type = args.iou_type
    coco = COCO()
    coco_det = coco.loadRes(args.results_file)

    """
        detection format:
        [{
        "image_id": int,
        "category_id": int,
        "segmentation": RLE,
        "score": float,
        }]
    """

    cocoEval = COCOeval(coco_gt, coco_det, iou_type)
    cocoEval.params.useCats = 0
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()



    prefix = 'instances'

    dataDir='datasets/coco/'
    dataType='val2017'
    annFile = '{}/annotations/{}_{}.json'.format(dataDir,prefix,dataType)

    resFile = 'training_dir/FreeSOLO_pl/inference/coco_instances_results.json'
    #resFile = 'demo/instances_val2017_densecl_r101.json'


