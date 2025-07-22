"""
Builds an MMDetection config file for a given model architecture.
"""

import argparse
import mmdet
import json
import matplotlib.pyplot as plt
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env
import albumentations
import pickle
import os
import re

def load_env_file(filepath):
    """Load environment variables from a shell-style .env file"""
    env_vars = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            # Parse KEY=VALUE or KEY='VALUE' or KEY="VALUE"
            match = re.match(r'^([A-Z_][A-Z0-9_]*)=(.*)$', line)
            if match:
                key, value = match.groups()
                # Remove quotes if present
                value = value.strip('\'"')
                env_vars[key] = value
    return env_vars

def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build an MMDetect config file for a given Faster R-CNN model"
    )
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument(
        "patched_dataset_src", type=str, help="Path to the patched dataset dir"
    )
    parser.add_argument("patch_size", type=str, help="Patch size used for training")
    parser.add_argument("out_dir", type=str, help="Output directory to store files")
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Initialise from MSCOCO weights",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default=None,
        help='Augmentation set to apply. Options: "pixel", "spatial", "both". Default: None',
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="faster_rcnn",
        help="Architecture to use. Default: faster_rcnn",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Set to True for distributed training",
    )

    args = parser.parse_args()
    return args


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info["MMDetection"] = f"{mmdet.__version__}+{get_git_hash()[:7]}"
    return env_info


for key, value in collect_env().items():
    print(f"{key}: {value}")


def main():
    args = parse_args()

    # Load environment variables
    env_vars = load_env_file("./paths.env")
    REPOSITORY_PATH = env_vars["REPOSITORY_PATH"]
    MMDETECTION_ROOT = REPOSITORY_PATH + "/mmdetection"
    WEIGHTS_ROOT = REPOSITORY_PATH + "/openmmlab_pretrained_weights"

    dataset_src = args.patched_dataset_src

    train_src = dataset_src + "/dataset_train_patches.json"
    val_src = dataset_src + "/dataset_val_patches.json"
    test_src = dataset_src + "/dataset_test_patches.json"
    data_root = dataset_src.replace("annotations", "")
    data_prefix = data_root + "images/"

    # Change number of epochs in configs/_base_/schedules/schedule_1x.py
    BATCH_SIZE = args.batch_size
    IMG_SIZE = tuple(
        map(int, args.patch_size.replace("(", "").replace(")", "").split(","))
    )

    # Get classes from COCO
    with open(train_src, "r") as f:
        train_data = json.load(f)
        f.close()
    classes = [x["name"] for x in train_data["categories"]]

    ## Get an RGB palette 0-255 for each class
    def get_rgb_palette(n):
        cmap = plt.cm.get_cmap("hsv", n)
        palette = [cmap(i) for i in range(n)]
        return [(int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)) for x in palette]

    palette = get_rgb_palette(len(classes))

    keymap = {"img": "image", "gt_bboxes": "bboxes"}

    data = {
        "train": {"classes": classes},
        "val": {"classes": classes},
        "test": {"classes": classes},
    }

    if args.augmentation:
        if args.augmentation == "pixel":
            albu_transforms = [
                dict(type="MotionBlur", p=0.5),
                dict(type="RandomBrightnessContrast", p=0.5),
                dict(type="RandomShadow", p=0.5),
            ]
        elif args.augmentation == "spatial":
            albu_transforms = [
                dict(type="HorizontalFlip", p=0.5),
                dict(type="VerticalFlip", p=0.5),
                dict(type="PixelDropout", p=0.5),
                dict(
                    type="RandomSizedBBoxSafeCrop",
                    p=0.5,
                    height=IMG_SIZE[0],
                    width=IMG_SIZE[1],
                ),
            ]
        elif args.augmentation == "both":
            albu_transforms = [
                dict(type="MotionBlur", p=0.5),
                dict(type="RandomBrightnessContrast", p=0.5),
                dict(type="RandomShadow", p=0.5),
                dict(type="HorizontalFlip", p=0.5),
                dict(type="VerticalFlip", p=0.5),
                dict(type="PixelDropout", p=0.5),
                dict(
                    type="RandomSizedBBoxSafeCrop",
                    p=0.5,
                    height=IMG_SIZE[0],
                    width=IMG_SIZE[1],
                ),
            ]
        else:
            raise ValueError(f"Invalid augmentation option: {args.augmentation}")
    else:
        albu_transforms = []

    test_evaluator = dict(
        type="CocoMetric",
        metric="bbox",
        format_only=False,
        ann_file=test_src,
        outfile_prefix=args.out_dir + "/test/" + args.model_name + "_patches",
    )

    data_loader_config = f"""
    
#dataset settings
dataset_type = 'CocoDataset'
classes = {classes}
data_root = '{data_root}'

h, w = {IMG_SIZE}

backend_args = None

albu_transforms = {albu_transforms}

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True),
    # Redundant resize, but necessary to add scale_factor to img_meta (https://github.com/open-mmlab/mmdetection/issues/11164)
    dict(
        type='Resize',
        scale=(h, w),
        keep_ratio=True),
    dict(type = 'Albu',
        transforms = albu_transforms,
        bbox_params = dict(
            type='BboxParams',
            format='pascal_voc', # fixes error https://github.com/albumentations-team/albumentations/issues/459#issuecomment-919113188
            label_fields=['gt_bboxes_labels'],
            min_visibility=0.1,
            min_area = 0.1,
            filter_lost_elements=True),
            keymap={keymap},
        skip_img_without_anno=True 
    ),
    dict(type='PackDetInputs',meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True
        ),
    # Redundant resize, but necessary to add scale_factor to img_meta (https://github.com/open-mmlab/mmdetection/issues/11164)
    dict(
        type='Resize',
        scale=(h, w),
        keep_ratio=True),
    dict(type='PackDetInputs',meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True
        ),
    # Redundant resize, but necessary to add scale_factor to img_meta (https://github.com/open-mmlab/mmdetection/issues/11164)
    dict(
        type='Resize',
        scale=(h, w),
        keep_ratio=True),
    dict(type='PackDetInputs',meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))
]

metainfo = dict(classes = {classes}, palette = {palette})

train_dataloader = dict(
    batch_size={BATCH_SIZE},
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='{train_src}',
        data_prefix=dict(img='{data_prefix}'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args)) 

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='{val_src}',
        data_prefix=dict(img='{data_prefix}'),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file='{val_src}',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

#inference on test dataset and
#format the output results for submission.

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='{test_src}',
        data_prefix=dict(img='{data_prefix}'),
        test_mode=True,
        pipeline=test_pipeline))

test_evaluator = {test_evaluator}
    """

    if args.distributed:
        data_loader_config = (
            data_loader_config
            + f"""
data=dict(samples_per_gpu=1, workers_per_gpu=1, train=dict(classes=classes),  val=dict(classes=classes),  test=dict(classes=classes))
        """
        )
    else:
        data_loader_config = (
            data_loader_config
            + f"""
data=dict(train=dict(classes=classes),  val=dict(classes=classes),  test=dict(classes=classes))
        """
        )

    data_loader_src = os.path.join(args.out_dir, f"{args.model_name}_data_loader.py")
    print(f"Data loader to be saved at {data_loader_src}")
    with open(data_loader_src, "w") as f:
        f.write(data_loader_config)
        f.close()

    model_architectures = {
        "faster_rcnn": "faster-rcnn_r50_fpn_25_class_benthic.py",
        "retinanet": "retinanet_r50_fpn_25_class_benthic.py",
        "cascade_rcnn": "cascade-rcnn_r50_fpn_25_class_benthic.py",
        "deformable-detr": "deformable-detr_r50_16xb2-50e_25_class_benthic.py",
        "dino_r50": "dino-4scale_r50_8xb2-12e_25_class_benthic.py",
        "codetr": "conditional-detr_r50_8xb2-50e_25_class_benthic.py",
    }

    model_config = f"""
_base_ = [
    '{MMDETECTION_ROOT}/configs/_base_/models/{model_architectures[args.architecture]}',
    '{data_loader_src}',
    '{MMDETECTION_ROOT}/configs/_base_/schedules/schedule_1x_{args.architecture}.py',
    '{MMDETECTION_ROOT}/configs/_base_/default_runtime.py'
]
    """

    model_pretrained_weights = {
        "faster_rcnn": "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        "retinanet": "retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
        "cascade_rcnn": "cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth",
        "deformable-detr": "deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth",
        "dino_r50": "dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth",
        "codetr": "conditional-detr_r50_8xb2-50e_coco_20221121_180202-c83a1dc0.pth",
    }

    if args.pretrained:
        model_config = (
            model_config
            + f"""\n\n
# load model from COCO
load_from = '{WEIGHTS_ROOT}/{model_pretrained_weights[args.architecture]}'
        """
        )

    config_src = os.path.join(args.out_dir, f"{args.model_name}_config.py")
    print(f"Model config to be saved at {config_src}")
    with open(config_src, "w") as f:
        f.write(model_config)
        f.close()


if __name__ == "__main__":
    main()
