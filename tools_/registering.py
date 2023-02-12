from detectron2.data import DatasetCatalog, MetadataCatalog
from datasets.coco import CocoClassAgnostic
from datasets.registable_dataset import RegisterableDataset
from detectron2.data.datasets.coco import register_coco_instances


def register_dataset(dataset:RegisterableDataset):
    DatasetCatalog.register(dataset.name, dataset.get_dataset_dicts)
    MetadataCatalog.get(dataset.name).set(thing_classes=dataset.thing_classes,
                                          evaluator_type=dataset.evaluator_type,
                                          thing_colors=dataset.thing_colors)


def register_all_datasets():
    coco_path = "datasets/coco"
    register_coco_instances("coco_20k", {}, f"{coco_path}/annotations/instances_coco_20k.json", f"{coco_path}/train2014")
    register_coco_instances("coco_20k_class_agnostic", {}, f"{coco_path}/annotations/instances_coco_20k_CAD.json", f"{coco_path}/train2014")

    # register_dataset(CocoClassAgnostic(name="coco_20k_class_agnostic", json_ann="instances_coco_20k.json", image_root="train2014"))
    # register_dataset(CocoClassAgnostic(name="coco_2017_val_class_agnostic", json_ann="instances_val2017.json", image_root="val2017"))


def register_all_models():
    import modeling.backbone.fpn
