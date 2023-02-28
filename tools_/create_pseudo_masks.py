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
from tools_.tokencut import tokencut_bipartition, tokencut
from tools_.watershed import watershed_instances

from detectron2.structures.masks import polygons_to_bitmask
from detectron2.structures.boxes import BoxMode

import json
from pathlib import Path

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
from torchvision.transforms.functional import resize, InterpolationMode
from PIL import Image

from sklearn.cluster import KMeans
import cv2

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

def min_max_normalization(vec, max_val=255):
    vec = (vec - vec.min()) / (
            vec.max() - vec.min())
    vec = vec * max_val
    return vec

def reshape_vec_2_patches(vec, patch_mask):
    if type(patch_mask) == torch.Tensor:
        patch_mask = patch_mask.numpy()
    reshaped_vec = np.zeros(patch_mask.shape)
    reshaped_vec[patch_mask] = vec
    return reshaped_vec


def num_corners_on_border(bbox, height, width):
    tl = bbox[:2]
    bl = tl + [0, bbox[3]]
    tr = tl + [bbox[2], 0]
    br = tl + bbox[2:]
    # check if there is an overlap between the bbox and at list 2 image borders
    num_of_corners_on_border = 0
    if tl[0] == 0 and tl[1] == 0:
        num_of_corners_on_border += 1
    if bl[0] == 0 and bl[1] == height:
        num_of_corners_on_border += 1
    if tr[0] == width and tr[1] == 0:
        num_of_corners_on_border += 1
    if br[0] == width and br[1] == height:
        num_of_corners_on_border += 1
    return num_of_corners_on_border

def get_instances_masks(K, vector_groups, subset_mask, dims):
    kmeans_labels = []
    instance_masks = []
    for group_name, eig_vec_group in vector_groups.items():
        if eig_vec_group is None:
            continue
        for i in range(eig_vec_group.shape[1]):
            v = min_max_normalization(eig_vec_group[:, i])
            v = reshape_vec_2_patches(v, subset_mask)
            v = np.stack([v, np.arange(dims[0]).reshape(dims[0], 1).repeat(dims[1], axis=1),
                          np.arange(dims[1]).reshape(1, dims[1]).repeat(dims[0], axis=0)], axis=2)
            samples = v.reshape(dims[0] * dims[1], 3)
            samples = samples[subset_mask.numpy().flatten()]
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(samples)
                m = np.zeros(dims) - 1
                m = m.astype(np.uint8)
                m[subset_mask.numpy()] = kmeans.labels_
                kmeans_labels.append({
                    'mask': m,
                    'group_name': group_name,
                    'eig_vec': i+1,
                    'k': k
                })
                semantic_labels = np.unique(kmeans.labels_)
                for s_label in semantic_labels:
                    semantic_mask = np.zeros(dims)
                    semantic_mask[subset_mask.numpy()] = kmeans.labels_ == s_label
                    semantic_mask = semantic_mask.astype(np.uint8)

                    # get RLE encoding of the mask
                    rle = mask_util.encode(np.asfortranarray(semantic_mask))
                    # bbox format is [x, y, width, height]
                    bbox = mask_util.toBbox(rle)
                    if num_corners_on_border(bbox, dims[0], dims[1]) >= 2:
                        continue
                    # break the mask into connected components
                    components = cv2.connectedComponents(semantic_mask * 255, connectivity=4)[1]
                    # put -1 on the background
                    components[semantic_mask == 0] = -1
                    # take on non-background connected components
                    instance_labels = np.unique(components)[1:]
                    for i_label in instance_labels:
                        instance_mask = np.zeros(dims)
                        instance_mask[components == i_label] = 1
                        instance_masks.append(instance_mask)
    return instance_masks, kmeans_labels


def iou_between_masks(masks):
    masks_flat = masks.reshape(masks.shape[0], -1)
    masks = torch.Tensor(masks_flat)
    union_intersection_diff_area = torch.cdist(masks, masks, p=1.0)
    intersection_area = masks @ masks.T
    union_area = union_intersection_diff_area + intersection_area
    iou = intersection_area / union_area
    return iou.numpy()


def cluster_mask_by_iou(masks, threshold=0.6, pivot_iter=5):
    ious = iou_between_masks(masks)
    # put zeros on the diagonal since we don't want to compare a mask with itself
    np.fill_diagonal(ious, 0)
    max_iou_indices = np.where(ious > threshold)[0]
    # get the common indices that are above the threshold
    indices, occurrences = np.unique(max_iou_indices, return_counts=True)
    # sort the indices by the number of occurrences in descending order
    occurrences_sorted = np.argsort(occurrences)[::-1]
    indices = indices[occurrences_sorted]

    ind_list = indices.tolist()
    indices_to_keep = []
    selected_mask = []
    used_indices = set()
    for j in range(len(indices)):
        if len(ind_list) == 0:
            break
        pivot_index = ind_list.pop(0)
        segment_proposals_indices = np.array([pivot_index]).astype(int)
        for iter in range(pivot_iter):
            prev_ind_num = len(segment_proposals_indices)
            above_threshold = np.array(np.where(ious[pivot_index] > threshold)).flatten()
            segment_proposals_indices = np.unique(np.concatenate((segment_proposals_indices, above_threshold)))
            if prev_ind_num == len(segment_proposals_indices):
                break
            ious_between_proposals = ious[:, segment_proposals_indices]
            ious_between_proposals = ious_between_proposals[segment_proposals_indices, :]
            # sum over the columns
            ious_sum = np.sum(ious_between_proposals, axis=1)
            pivot_index = segment_proposals_indices[np.argmax(ious_sum)]
        if pivot_index not in used_indices:
            # get rid of all indices in segment_proposals_indices that are also in used_indices
            segment_proposals_indices = np.array([i for i in segment_proposals_indices if i not in used_indices])
            pivot_mask = masks[pivot_index]
            # get RLE encoding of the pivot mask
            rle = mask_util.encode(np.asfortranarray(pivot_mask.astype(np.uint8)))
            # bbox format is [x, y, width, height]
            pivot_bbox = mask_util.toBbox(rle)
            selected_mask.append({
                'pivot_index': pivot_index,
                'pivot_mask': pivot_mask,
                'pivot_bbox': pivot_bbox,
                'segment_proposals_indices': segment_proposals_indices,
                'proposals_count': len(segment_proposals_indices),
                'is_small': pivot_bbox[2] <= 2 and pivot_bbox[3] <= 2
            })
            indices_to_keep.append(pivot_index)
        used_indices.update(segment_proposals_indices)
        # remove the indices of the current mask from the list
        ind_list = [i for i in ind_list if i not in used_indices]
    return selected_mask



def instance_mask_inference(dino_model, mae_enc_model, img, subset_mask, dims, image=None):

    dino_feat_extractor = ViTFeatureExtractor(model=dino_model, blocks_to_extract=[-1])
    mae_enc_feat_extractor = ViTFeatureExtractor(model=mae_enc_model, blocks_to_extract=[-1])

    dino_features = dino_feat_extractor.get_features(img, subset_mask)
    mae_enc_features = mae_enc_feat_extractor.get_features(img, subset_mask)

    feature_k_dino = dino_features["block -1"]["k"]
    feature_k_mae_enc = mae_enc_features["block -1"]["k"]

    eigenvectors_dino_k, eigenvalues_dino_k = tokencut(feature_k_dino, feature_k_dino, tau=0.2, no_binary_graph=False,eig_vecs=3)
    eigenvectors_mae_enc_k, eigenvalues_mae_enc_k = tokencut(feature_k_mae_enc, feature_k_mae_enc, tau=0.0, no_binary_graph=True, eig_vecs=3)

    K = [2, 3, 4, 5]
    vector_groups = {
        'dino_k': eigenvectors_dino_k,
        'mae_enc_k': eigenvectors_mae_enc_k,
    }
    instance_masks, kmeans_labels = get_instances_masks(K, vector_groups, subset_mask, dims)
    # create numpy array from the masks list of numpy arrays
    masks = np.array(instance_masks)

    original_mask_clusters = cluster_mask_by_iou(masks, threshold=0.6)

    proposals_counts = np.array([c['proposals_count'] for c in original_mask_clusters])

    counts_range = np.max(proposals_counts)-np.min(proposals_counts)
    thresh_percentage = 0.7
    thresh = thresh_percentage*counts_range + np.min(proposals_counts)

    image_masks = []
    # Note: image_rgb is PIL object with size [x,y], not [y,x]
    original_image_shape = [image_rgb.size[1], image_rgb.size[0]]
    for c in original_mask_clusters:
        if c['proposals_count'] > thresh:
            cluster_masks = masks[c['segment_proposals_indices']]
            cluster_masks = resize(torch.Tensor(cluster_masks), original_image_shape, interpolation=InterpolationMode.NEAREST).numpy()
            final_mask = (np.sum(cluster_masks, axis=0)/cluster_masks.shape[0] > 0.5).astype(np.uint8)
            score = (c['proposals_count']-np.min(proposals_counts))/counts_range
            image_masks.append({
                "mask": final_mask,
                "score": score
            })
            # print(c['proposals_count'])

    return image_masks

def get_dino_vit(patch_size, vit_size):
    if patch_size == 8:
        if vit_size == "small":
            from modeling.dino.hubconf import dino_vits8
            return dino_vits8()
        elif vit_size == "base":
            from modeling.dino.hubconf import dino_vitb8
            return dino_vitb8()
        else:
            raise ValueError("Vit size must be 'small' or 'base'")
    elif patch_size == 16:
        if vit_size == "small":
            from modeling.dino.hubconf import dino_vits16
            return dino_vits16()
        elif vit_size == "big":
            from modeling.dino.hubconf import dino_vitb16
            return dino_vitb16()
        else:
            raise ValueError("Vit size must be 'small' or 'base'")
    else:
        raise ValueError("Patch size must be 8 or 16")

def get_mae_dec_vit(vit_size):
    if vit_size == "base":
        from modeling.mae.mae import mae_enc_vitb16
        return mae_enc_vitb16()
    elif vit_size == "large":
        raise NotImplementedError
    else:
        raise ValueError("Vit size must be 'base' or 'large'")


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
    parser.add_argument("--mae-vit-size", type=str, default="base", help="DINO ViT model size",choices=["base"])
    parser.add_argument("--coco-path", type=str, default="datasets/coco")
    parser.add_argument("--ann-path", type=str, default="", help="Output directory for json annotation file")
    parser.add_argument("--res-path", type=str, default="", help="Output directory for json results evaluation file")
    parser.add_argument("--device", type=str, default="cpu", help="computation device", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)

    DEBUG = True
    if DEBUG:
        from detectron2.data.datasets.coco import register_coco_instances
        coco_path = "datasets/coco"
        register_coco_instances("coco_20k_class_agnostic", {}, f"{coco_path}/annotations/instances_coco_20k_CAD_DEBUG.json", "")
        dataset_dicts = DatasetCatalog.get("coco_20k_class_agnostic")

    else:
        register_all_datasets()
        dataset_dicts = DatasetCatalog.get(args.dataset)


    dino_model = get_dino_vit(args.patch_size, args.dino_vit_size).to(device)
    mae_enc_model = get_mae_dec_vit(args.mae_vit_size).to(device)

    dataset_pseudo_labels_name = args.dataset + "_pseudo_labels"
    dataset_pseudo_labels_dicts = []
    results_objects = []


    from pathlib import Path
    debug_dir = "/data/home/ssaricha/InstanceSeg/debug_visualiztions/version_1"
    Path(debug_dir).mkdir(parents=True, exist_ok=True)

    ann_id = 0
    for sample in tqdm(dataset_dicts, desc="Creating pseudo labels"):
        image_rgb = Image.open(sample["file_name"])
        img = transform(image_rgb).to(device)
        img, patches_shape = pad_img(img, args.patch_size)
        mask_all = torch.ones(patches_shape).type(torch.bool)
        out_dict = {}

        image_masks = instance_mask_inference(dino_model, mae_enc_model, img, mask_all, mask_all.shape, image=image_rgb)

        annotations = []
        for i, mask_data in enumerate(image_masks):
            rle = mask_util.encode(np.asfortranarray(mask_data["mask"]))
            rle["counts"] = rle["counts"].decode("ascii")
            annotations.append({
                "iscrowd": 0,  # TODO: is this should be changed?
                "bbox": mask_util.toBbox(rle),
                "category_id": 1,  # pseudo labels are class agnostic, hence, we set the category id to 1
                "segmentation": rle,
                "bbox_mode": BoxMode.XYWH_ABS,
                "id": ann_id,
            })
            res_obj = {
                "image_id": sample["image_id"],
                "category_id": 1,
                "segmentation": rle,
                "score": mask_data["score"]
            }
            results_objects.append(res_obj)

        dataset_pseudo_labels_dicts.append({
            "file_name": sample["file_name"],
            "image_id": sample["image_id"],
            "height": sample["height"],
            "width": sample["width"],
            "annotations": annotations
        })


    # register the new pseudo labels dataset to detectron2
    def get_dataset_pseudo_labels_dicts_dicts():
        return dataset_pseudo_labels_dicts
    DatasetCatalog.register(dataset_pseudo_labels_name, get_dataset_pseudo_labels_dicts_dicts)
    MetadataCatalog.get(dataset_pseudo_labels_name).set(thing_classes=MetadataCatalog.get(args.dataset).thing_classes)

    # write annotations to json file
    if args.ann_path == "":
        ann_path = os.path.join(args.coco_path, f"annotations/instances_{dataset_pseudo_labels_name}.json")
    else:
        ann_path = args.ann_path

    print(f"Writing {dataset_pseudo_labels_name} annotations to {ann_path}...")
    convert_to_coco_json(dataset_pseudo_labels_name, ann_path)
    # check if lock file exists and delete it
    if os.path.exists(ann_path + ".lock"):
        os.remove(ann_path + ".lock")

    # write results to json file
    if args.res_path == "":
        Path("results").mkdir(parents=True, exist_ok=True)
        res_path = os.path.join("results", f"results_{dataset_pseudo_labels_name}.json")
    else:
        res_path = args.res_path

    print(f"Writing {dataset_pseudo_labels_name} as a results file to {res_path}...")
    with open(res_path, 'w') as f:
        json.dump(results_objects, f)

    print("Done!")
