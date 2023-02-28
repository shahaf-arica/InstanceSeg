import os
import sys

import matplotlib.pyplot as plt

# change working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# add project root to path
sys.path.insert(0, os.getcwd())

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create pseudo labels mask coco annotation file")
    parser.add_argument("--ann-file", type=str, default="dataset/coco/annotations/instances_coco_20k_CAD.json", help="json file with coco annotations")
    parser.add_argument("--results-file", type=str, default="results/results_coco_20k_class_agnostic_pseudo_labels.json", help="json file with detections")
    parser.add_argument("--iou_type", type=str, default="segm", help="type of coco iou evaluation")
    args = parser.parse_args()

    iou_type = args.iou_type
    coco = COCO(args.ann_file)
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
    # from detectron2.structures.masks import polygons_to_bitmask
    # import matplotlib.pyplot as plt
    # ann_ids = coco.getAnnIds()
    # for id in ann_ids:
    #     ann = coco.loadAnns(id)[0]
    #     img_id = ann['image_id']
    #     img_inf = coco.loadImgs(img_id)[0]
    #     mask = polygons_to_bitmask(ann["segmentation"], img_inf["height"], img_inf["width"])
    #     print(f"gt mask shape: {mask.shape}")
    #     img = plt.imread(img_inf['file_name'])
    #     print(f"image shape: {img.shape}")
    #     break
    #     # plt.imshow(mask)
    #     # plt.show()
    #
    # import pycocotools.mask as mask_util
    # res_ann_ids = coco_det.getAnnIds()
    # for id in ann_ids:
    #     ann = coco_det.loadAnns(id)[0]
    #     img_id = ann['image_id']
    #     img_inf = coco_det.loadImgs(img_id)[0]
    #     mask_ = mask_util.decode(ann["segmentation"])
    #     print(mask_.shape)
    #     break
    #     # plt.imshow(mask_)
    #     # plt.show()

    cocoEval = COCOeval(coco, coco_det, iou_type)
    cocoEval.params.useCats = 0
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()



