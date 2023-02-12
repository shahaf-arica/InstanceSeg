import os
import sys

# change working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# add project root to path
sys.path.insert(0, os.getcwd())

import argparse
from tools_.registering import register_all_datasets
from detectron2.data import DatasetCatalog, MetadataCatalog

from tools_.vit_feature_extraction import ViTFeatureExtractor
from tools_.tokencut import tokencut_bipartition
from tools_.watershed import watershed_instances

from detectron2.structures.masks import polygons_to_bitmask

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
from matplotlib.pyplot import cm

import pycocotools.mask as mask_util
from detectron2.data.datasets.coco import convert_to_coco_json
from tqdm import tqdm

import scipy.ndimage as ndimage
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
# Image transformation applied to all images
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def pad_img(img, patch_size):
    # Padding the image with zeros to fit multiple of patch-size
    size_im = (
        img.shape[0],
        int(np.ceil(img.shape[1] / patch_size) * patch_size),
        int(np.ceil(img.shape[2] / patch_size) * patch_size),
    )
    padded = torch.zeros(size_im)
    padded[:, : img.shape[1], : img.shape[2]] = img
    img = padded
    patches_shape = img.shape[1] // patch_size, img.shape[2] // patch_size
    return img, patches_shape

def reshape_mask_vec_into_patches_mask(vec, subset_mask) -> np.ndarray:
    if type(subset_mask) == torch.Tensor:
        subset_mask = subset_mask.numpy()
    if type(vec) == torch.Tensor:
        vec = vec.numpy()
    mask = np.zeros(subset_mask.shape).astype(bool)
    mask[subset_mask] = vec
    return mask

def foreground_background_split(bipartition, subset_mask):
    bipartition_mask_1 = reshape_mask_vec_into_patches_mask(bipartition, subset_mask)
    bipartition_mask_2 = reshape_mask_vec_into_patches_mask(~bipartition, subset_mask)

    # get bbox of bipartition_mask_1
    rle_1 = mask_util.encode(np.asfortranarray(bipartition_mask_1.astype(np.uint8)))
    # bbox format is [x, y, width, height]
    bbox_1 = mask_util.toBbox(rle_1)

    # get bbox of bipartition_mask_2
    rle_2 = mask_util.encode(np.asfortranarray(bipartition_mask_2.astype(np.uint8)))
    # bbox format is [x, y, width, height]
    bbox_2 = mask_util.toBbox(rle_2)

    # compare the area of the two bboxes
    if bbox_1[2] * bbox_1[3] < bbox_2[2] * bbox_2[3]:
        foreground_segmentation = bipartition_mask_1
        background_segmentation = bipartition_mask_2
    else:
        foreground_segmentation = bipartition_mask_2
        background_segmentation = bipartition_mask_1

    return foreground_segmentation, background_segmentation


def recursive_bipartite(dino_model, img, subset_mask, out_dict, node_name, depth, max_depth):

    dino_feat_extractor = ViTFeatureExtractor(model=dino_model)

    features = dino_feat_extractor.get_features(img, subset_mask)

    feature_k = features["block 11"]["k"]

    bipartition, second_smallest_vec = tokencut_bipartition(feature_k, feature_k)

    aggregate_att_map = torch.zeros(features["block 11"]["attn"].shape)
    for feat in features.values():
        aggregate_att_map += feat["attn"]
    # avg foreground prior
    aggregate_att_map /= len(features)
    cls_token_att_map = aggregate_att_map[0,1:]
    # normalize to [0,1]
    cls_token_att_map = (cls_token_att_map - cls_token_att_map.min()) / (cls_token_att_map.max() - cls_token_att_map.min())
    # take cls token attention map as foreground prior
    # foreground_prior = np.zeros(subset_mask.shape)
    # foreground_prior[subset_mask.numpy()] = cls_token_att_map
    # apply median filter to smooth the foreground prior
    # foreground_prior = ndimage.median_filter(foreground_prior, size=5)
    # plt.imshow(foreground_prior)
    # plt.show()
    # plt.hist(cls_token_att_map)
    # plt.show()
    # plt.hist(foreground_prior[subset_mask.numpy()].flatten())
    # plt.show()
    percentage_cls_token_above_0_5 = torch.sum(cls_token_att_map > 0.5)/len(cls_token_att_map)
    if percentage_cls_token_above_0_5 < 0.01 and node_name != "root":
        return
    # print(f"percentage cls token above 0.5: {torch.sum(cls_token_att_map > 0.5)/len(cls_token_att_map)}")
    if bipartition is not None and second_smallest_vec is not None:
        foreground_segmentation, background_segmentation = foreground_background_split(bipartition, subset_mask)
        # apply median filter to smooth the foreground segmentation
        # foreground_segmentation = ndimage.median_filter(foreground_segmentation, size=3)
        forg_labels_watershed = watershed_instances(foreground_segmentation)
        # plt.imshow(foreground_segmentation)
        # plt.show()
        # plt.imshow(background_segmentation)
        # plt.show()
        out_dict[node_name] = {
            'bipartition': bipartition,
            'second_smallest_vec': second_smallest_vec,
            'subset_mask': subset_mask,
            'foreground_segmentation': foreground_segmentation,
            'background_segmentation': background_segmentation
        }
        next_subset_mask = torch.Tensor(background_segmentation).type(torch.bool)
        if depth + 1 < max_depth:
            recursive_bipartite(dino_model, img, next_subset_mask, out_dict, node_name + "-0", depth + 1, max_depth)
        else:
            print("max depth reached")
        # subset_mask_1 = subset_dino_mask_1
        # subset_mask_2 = subset_dino_mask_2
        # # recursive call
        # if depth + 1 < max_depth:
        #     recursive_bipartite(dino_model, mae_model, img, subset_mask_1, out_dict, node_name + "-0", depth + 1, max_depth, which_features, which_mae_model)
        #     recursive_bipartite(dino_model, mae_model, img, subset_mask_2, out_dict, node_name + "-1", depth + 1, max_depth, which_features, which_mae_model)


def get_dino_vit(patch_size, vit_size):
    if patch_size == 8:
        if vit_size == "small":
            from modeling.dino.hubconf import dino_vits8
            return dino_vits8()
        elif vit_size == "big":
            from modeling.dino.hubconf import dino_vitb8
            return dino_vitb8()
        else:
            raise ValueError("Vit size must be small or big")
    elif patch_size == 16:
        if vit_size == "small":
            from modeling.dino.hubconf import dino_vits16
            return dino_vits16()
        elif vit_size == "big":
            from modeling.dino.hubconf import dino_vitb16
            return dino_vitb16()
        else:
            raise ValueError("Vit size must be small or big")
    else:
        raise ValueError("Patch size must be 8 or 16")

def rgb_color(color):
    c = np.asarray(to_rgb(color)) * 255
    return np.asarray(c, dtype=np.uint8)
def add_segmentation_to_image(image, seg_mask, color):
    # put green where mask is False
    image[seg_mask] = rgb_color(color)
    return image

def plot_discovered_instances(image_file, gt_annotations, discovered_annotations):
    fig, axs = plt.subplots(1,3, figsize=(14, 10))
    # plot image
    img = plt.imread(image_file)
    axs[0].imshow(img)
    axs[0].set_title("Image")
    # plot ground truth
    n = max(len(gt_annotations), len(discovered_annotations))
    colors = cm.rainbow(np.linspace(0, 1, n))
    img_gt = img.copy()
    for i, ann in enumerate(gt_annotations):
        bbox = ann["bbox"]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=colors[i], facecolor='none')
        axs[1].add_patch(rect)
        mask = polygons_to_bitmask(ann["segmentation"], img.shape[0], img.shape[1])
        # convert color from matplotlib cm to rgb
        img_gt[mask] = rgb_color(colors[i])
    axs[1].imshow(img_gt)
    axs[1].set_title("Ground truth")
    # plot discovered
    img_discovered = img.copy()
    for i, ann in enumerate(discovered_annotations):
        bbox = ann["bbox"]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=colors[i], facecolor='none')
        axs[2].add_patch(rect)
        mask = mask_util.decode(ann["segmentation"]).astype(bool)
        # mask = ann["original_mask"]
        img_discovered[mask] = rgb_color(colors[i])
    axs[2].imshow(img_discovered)
    axs[2].set_title("Discovered")
    return fig



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create pseudo labels mask coco annotation file")
    parser.add_argument("--dataset", type=str, default="coco_20k_class_agnostic")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size for DINO", choices=[8, 16])
    parser.add_argument("--dino-vit-size", type=str, default="small", help="DINO ViT model size", choices=["small", "big"])
    parser.add_argument("--coco-path", type=str, default="datasets/coco")
    parser.add_argument("--output-path", type=str, default="", help="Output directory for json annotation file")
    args = parser.parse_args()

    register_all_datasets()
    dataset_dicts = DatasetCatalog.get(args.dataset)

    dino_model = get_dino_vit(args.patch_size, args.dino_vit_size)

    dataset_pseudo_labels_name = args.dataset + "_pseudo_labels"
    dataset_pseudo_labels_dicts = []

    from pathlib import Path
    debug_dir = "/data/home/ssaricha/InstanceSeg/debug_visualiztions/version_1"
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    for sample in tqdm(dataset_dicts, desc="Creating pseudo labels"):
        img = transform(Image.open(sample["file_name"]))
        img, patches_shape = pad_img(img, args.patch_size)
        mask_all = torch.ones(patches_shape).type(torch.bool)
        out_dict = {}
        recursive_bipartite(dino_model, img, mask_all, out_dict, "root", 0, 5)
        instances_masks = []
        for node_name, node in out_dict.items():
            instances_masks.append(node["foreground_segmentation"])
        annotations = []
        for i, mask in enumerate(instances_masks):
            soft_mask = F.interpolate(torch.from_numpy(mask)[None, None, :, :].float(), size=(sample["height"], sample["width"]), mode='bilinear', align_corners=False)[0]
            mask = (soft_mask >= 0.5).float()[0]
            # rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle = mask_util.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            annotations.append({
                "iscrowd": sample["annotations"][i]["iscrowd"],
                "bbox": mask_util.toBbox(rle),
                "category_id": 1, # pseudo labels are class agnostic, hence, we set the category id to 1
                "segmentation": rle,
                "bbox_mode": sample["annotations"][i]["bbox_mode"],
                # "original_mask": mask.numpy().astype(bool)
            })
        fig = plot_discovered_instances(image_file=sample["file_name"],
                                        gt_annotations=sample["annotations"],
                                        discovered_annotations=annotations)
        fig.savefig(os.path.join(debug_dir, os.path.basename(sample["file_name"])))
        # fig.show()
        plt.close(fig)
        dataset_pseudo_labels_dicts.append({
            "file_name": sample["file_name"],
            "image_id": sample["image_id"],
            "height": sample["height"],
            "width": sample["width"],
            "annotations": annotations
        })
        # TODO: for debugging delete later
        break

    # register the new pseudo labels dataset to detectron2
    def get_dataset_pseudo_labels_dicts_dicts():
        return dataset_pseudo_labels_dicts
    DatasetCatalog.register(dataset_pseudo_labels_name, get_dataset_pseudo_labels_dicts_dicts)
    MetadataCatalog.get(dataset_pseudo_labels_name).set(thing_classes=MetadataCatalog.get(args.dataset).thing_classes)

    # write coco 20k annotations to json file
    if args.output_path == "":
        output_path = os.path.join(args.coco_path, f"annotations/instances_{dataset_pseudo_labels_name}.json")
    else:
        output_path = args.output_path
    print(f"Writing {dataset_pseudo_labels_name} annotations to {output_path}...")
    convert_to_coco_json(dataset_pseudo_labels_name, output_path)

    print("Done!")
