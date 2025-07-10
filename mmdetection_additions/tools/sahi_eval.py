"""
Evaluate a trained patch-based MMDetection model on whole images using SAHI.
"""
import argparse
import glob
import json
import torch
import pickle
import numpy as np
import os

from mmengine import Config
from mmdet.utils import replace_cfg_vals, update_data_root
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS

from sahi import AutoDetectionModel
from sahi.predict import predict
from sahi.scripts.coco_evaluation import evaluate

from confusion_matrix_abundance_ordered import calculate_confusion_matrix, plot_confusion_matrix

def parse_args():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate model on whole images')
    parser.add_argument('config_src', help='MMDetection model config')
    parser.add_argument('checkpoint_src', help='MMDetection model checkpoint')
    parser.add_argument('img_src', help='Path to the test whole images')
    parser.add_argument('json_src', help='Path to the test json')
    parser.add_argument('patch_size', help='Patch size used for training')
    parser.add_argument('overlap', type=float, help='Overlap used for training')
    parser.add_argument('output_folder_name', help='Name of the output folder. Stored in the model directory.')
    parser.add_argument('--bbox-conf-threshold', type=float, default=0.5, help='Bounding box confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for matching')
    parser.add_argument('--abundance-ordering', type=str, help='Path to the abundance ordering pkl')
    parser.add_argument('--postprocess-type', type=str, default='NMM', help='Postprocessing type')
    parser.add_argument('--match-threshold', type=float, default=0.5, help='Match threshold for SAHI')
    
    args = parser.parse_args()
    return args

def group_detections_by_image_id(json_data):
    """
    Groups detections by image ID.
    
    Args:
        json_data (list): List of detection results in JSON format.
        
    Returns:
        dict: Detections grouped by image ID.
    """
    grouped = {}
    for i, item in enumerate(json_data):
        image_id = item['image_id']
        if image_id not in grouped:
            grouped[image_id] = []
        grouped[image_id].append(item)        
    return grouped

def build_whole_image_dataset(cfg_src, whole_image_test_dataset_json_path):
    """
    Builds the dataset configuration for the whole image test dataset.
    
    Args:
        cfg_src (str): Path to the configuration file.
        whole_image_test_dataset_json_path (str): Path to the whole image test dataset JSON file.
        
    Returns:
        Dataset: The constructed dataset.
    """
    cfg = Config.fromfile(cfg_src)
    # Modify the test dataloader to point to the whole image dataset rather than the patched dataset
    cfg.test_dataloader.dataset.ann_file = whole_image_test_dataset_json_path
    cfg.test_dataloader.dataset.data_root = whole_image_test_dataset_json_path.split('annotations')[0]
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    return dataset

def conf_mat(dataset, results, score_thr=0, tp_iou_thr=0.5):
    """
    Calculates the confusion matrix.
    
    Args:
        dataset (Dataset): The dataset.
        results (list): List of detection results.
        score_thr (float): Score threshold for detections.
        tp_iou_thr (float): IoU threshold for true positive detections.
        
    Returns:
        np.ndarray: The calculated confusion matrix.
    """
    confusion_matrix = calculate_confusion_matrix(dataset, results, score_thr=score_thr, tp_iou_thr=tp_iou_thr)
    return confusion_matrix

def load_model(config_src, checkpoint_src, bbox_conf_threshold):
    """
    Loads the model from configuration and checkpoint files.
    
    Args:
        config_src (str): Path to the configuration file.
        checkpoint_src (str): Path to the checkpoint file.
        bbox_conf_threshold (float): Bounding box confidence threshold.
        
    Returns:
        AutoDetectionModel: The loaded model.
    """
    return AutoDetectionModel.from_pretrained(
        model_type='mmdet',
        model_path=checkpoint_src,
        config_path=config_src,
        confidence_threshold=bbox_conf_threshold,
        device='cuda'
    )

def run_prediction(model, img_src, patch_size, overlap, json_src, output_folder, postprocess_type='NMM', match_threshold=0.5):
    """
    Runs the prediction on the test images.
    
    Args:
        model (AutoDetectionModel): The detection model.
        img_src (str): Path to the test images.
        patch_size (tuple): Patch size used for training.
        overlap (float): Overlap used for training.
        json_src (str): Path to the test JSON file.
        output_folder (str): Name of the output folder.
        postprocess_type (str): Postprocessing type.
        match_threshold (float): Match threshold for SAHI, default is 0.5.
    """
    
    predict(
        detection_model=model,
        source=img_src,
        slice_height=patch_size[0],
        slice_width=patch_size[1],
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type=postprocess_type,
        postprocess_match_metric='IOU',
        postprocess_match_threshold= match_threshold,
        dataset_json_path=json_src,
        visual_bbox_thickness=2,
        project=output_folder,
        name=output_folder,
        novisual=False,
    )

def load_results(results_json_src):
    """
    Loads results from a JSON file.
    
    Args:
        results_json_src (str): Path to the results JSON file.
        
    Returns:
        list: List of detection results.
    """
    with open(results_json_src, 'r') as f:
        results_json = json.load(f)
    return results_json

def prepare_results_for_mmdet(results_grouped):
    """
    Prepares results for MMDetection evaluation.
    
    Args:
        results_grouped (dict): Detections grouped by image ID.
        
    Returns:
        list: List of results formatted for MMDetection.
    """
    results_grouped_mmdet = []
    for per_img_res in results_grouped:
        per_img_res_mmdet = {}
        per_img_res_mmdet['image_id'] = per_img_res
        per_img_res_mmdet['pred_instances'] = {}
        
        # SAHI outputs results in form [x_min, y_min, w, h] but Ground Truth expects [x_min, y_min, x_max, y_max]
        new_bbox = []
        for item in results_grouped[per_img_res]:
            new_bbox.append([item['bbox'][0], item['bbox'][1], item['bbox'][0] + item['bbox'][2], item['bbox'][1] + item['bbox'][3]])
        per_img_res_mmdet['pred_instances']['bboxes'] = torch.tensor(new_bbox)
        
        per_img_res_mmdet['pred_instances']['scores'] = torch.tensor([item['score'] for item in results_grouped[per_img_res]])
        per_img_res_mmdet['pred_instances']['labels'] = torch.tensor([item['category_id'] for item in results_grouped[per_img_res]])
        
        results_grouped_mmdet.append(per_img_res_mmdet)
    return results_grouped_mmdet

def reorder_confusion_matrix(conf, dataset, abundance_ordering):
    """
    Reorders the confusion matrix based on abundance ordering.
    
    Args:
        conf (np.ndarray): The confusion matrix.
        dataset (Dataset): The dataset.
        abundance_ordering (str): Path to the abundance ordering pickle file.
        
    Returns:
        tuple: Reordered confusion matrix and class ordering.
    """
    with open(abundance_ordering, 'rb') as f:
        abundance_ordering = pickle.load(f)

    class_ordering = {}
    for class_name in dataset.metainfo['classes']:
        class_ordering[class_name] = list(abundance_ordering.keys()).index(class_name)

    # Some classes may not be present in the dataset
    for key in list(class_ordering.keys()):
        if key not in dataset.metainfo['classes']:
            del class_ordering[key]
    # Order and make sequential
    class_ordering = dict(sorted(class_ordering.items(), key=lambda item: item[1]))
    class_ordering = {key: i for i, key in enumerate(class_ordering.keys())}
    class_ordering['background'] = len(class_ordering)

    current_class_ordering = {key: i for i, key in enumerate(dataset.metainfo['classes'] + ['background'])}

    confusion_matrix_reordered = np.zeros(shape=[len(class_ordering), len(class_ordering)])
    for key1, i in class_ordering.items():
        for key2, j in class_ordering.items():
            confusion_matrix_reordered[i, j] = conf[current_class_ordering[key1], current_class_ordering[key2]]

    return confusion_matrix_reordered, class_ordering

def plot_conf_matrix(confusion_matrix, labels, title, out_full, ):
    """
    Plots the confusion matrix.
    
    Args:
        confusion_matrix (np.ndarray): The confusion matrix.
        labels (list): List of class labels.
        title (str): Title of the plot.
        out_full (str): Path to the output folder.
    """
    plot_confusion_matrix(confusion_matrix, labels, show=True, title=title, save_dir=out_full)

def main():
    args = parse_args()
    
    patch_size = tuple(map(int, args.patch_size.replace('(', '').replace(')', '').split(',')))

    model = load_model(args.config_src, args.checkpoint_src, args.bbox_conf_threshold)
    out_full = args.checkpoint_src.split('best')[0] + args.output_folder_name
    
    # If dir exists, predict will run but add a number to the end of the output_name
    # assert the name is unique so that correct results are loaded in
    if os.path.exists(out_full):
        raise ValueError(f'Output folder {out_full} already exists')
    
    run_prediction(model,
                   args.img_src,
                   patch_size,
                   args.overlap,
                   args.json_src,
                   out_full,
                   args.postprocess_type,
                   args.match_threshold)
    
    results_json_src = os.path.join(out_full, 'result.json')
    results_json = load_results(results_json_src)
    results_grouped = group_detections_by_image_id(results_json)
    results_grouped_mmdet = prepare_results_for_mmdet(results_grouped)
    
    dataset = build_whole_image_dataset(args.config_src, args.json_src)
    conf = conf_mat(dataset, results_grouped_mmdet, score_thr=args.bbox_conf_threshold, tp_iou_thr=args.iou_threshold)
    
    if args.abundance_ordering is not None:
        confusion_matrix_reordered, class_ordering = reorder_confusion_matrix(conf, dataset, args.abundance_ordering)
        plot_conf_matrix(confusion_matrix_reordered, class_ordering.keys(), f'Normalised Confusion Matrix\nSAHI processed ({args.postprocess_type}), ordered by abundance', out_full)
    else:
        labels = dataset.metainfo['classes'] + ['background']
        plot_conf_matrix(conf, labels, f'Normalised Confusion Matrix\nSAHI processed ({args.postprocess_type})', out_full)
        
    evaluate(
        dataset_json_path=args.json_src,
        result_json_path=results_json_src,
        out_dir=out_full,
        type='bbox',
        max_detections=10000,
        classwise=True
    )

if __name__ == '__main__':
    main()
