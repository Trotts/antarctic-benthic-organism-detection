"""
Evaluate a trained patch-based MMDetection model on whole images without SAHI.
"""

import glob
import mmcv
import numpy as np
import torch
import argparse
import os
import json
from mmdet.apis import DetInferencer
import struct
import zlib
from typing import Optional
import time
import copy
import cv2
import matplotlib.pyplot as plt

from sahi_eval import load_results, build_whole_image_dataset, conf_mat, reorder_confusion_matrix
from sahi.scripts.coco_evaluation import evaluate
from sahi.utils.cv import Colors, apply_color_mask
from sahi.prediction import ObjectPrediction
from sahi.utils.file import Path

from confusion_matrix_abundance_ordered import plot_confusion_matrix

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
    
    args = parser.parse_args()
    return args

def crop_to_patch(img, patch_bbox):
    """
    Crops an image to a patch given a bounding box in xyxy format.
    
    Args:
        img (np.ndarray): Image to crop.
        patch_bbox (list): Bounding box in xyxy format.
        
    Returns:
        np.ndarray: Cropped image.
    """
    x_min, y_min, x_max, y_max = patch_bbox
    return img[y_min:y_max, x_min:x_max]

def threshold_results(results, threshold):
    """
    Given a dictionary of results, filters out the predictions that are below a certain threshold.
    
    Args:
        results (dict): Dictionary of results.
        threshold (float): Threshold to filter out predictions.
        
    Returns:
        dict: Filtered dictionary of results.
    """
    to_keep = []
    
    for i, score in enumerate(results['predictions'][0]['scores']):
        if score >= threshold:
            to_keep.append(i)
            
    ## Keep only the predictions that are above the threshold
    results = {k: [v[i] for i in to_keep] for k, v in results['predictions'][0].items()}
    return results

def calculate_patch_bboxes(
        image_height: int, 
        image_width: int,
        patch_height: int = 512, 
        patch_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2
):
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping patches according to the height and width provided. These patches are returned
    as bounding boxes in xyxy format.

    Args:
        image_height (int): Height of the image.
        image_width (int): Width of the image.
        patch_height (int): Height of the patch.
        patch_width (int): Width of the patch.
        overlap_height_ratio (float): Ratio of overlap between patches in the height dimension.
        overlap_width_ratio (float): Ratio of overlap between patches in the width dimension.
        
    Returns:
        list: List of bounding boxes in xyxy format.
    """

    patch_bboxes = []
    y_max = y_min = 0
    y_overlap = int(patch_height * overlap_height_ratio)
    x_overlap = int(patch_width * overlap_width_ratio)

    while y_max < image_height:
        x_max = x_min = 0
        y_max = y_min + patch_height

        while x_max < image_width:
            x_max = x_min + patch_width

            if y_max > image_height or x_max > image_width:
                x_max = min(x_max, image_width)
                y_max = min(y_max, image_height)
                x_min = max(x_max - patch_width, 0)
                y_min = max(y_max - patch_height, 0)
                patch_bboxes.append([x_min, y_min, x_max, y_max])
            else:
                patch_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return patch_bboxes

def get_whole_image_filepaths(whole_images_src, dataset_json):
    """
    Returns the filepaths of the whole images that are in the dataset.
    
    Args:
        whole_images_src (str): Path to the directory containing the whole images.
        dataset_json (dict): Dictionary containing the dataset information.
        
    Returns:
        list: List of filepaths to the whole images that are in the dataset.
    """
    whole_images_srcs = glob.glob(whole_images_src + '/*.JPG')
    dataset_image_filepaths = [whole_images_src + '/' + image['file_name'] for image in dataset_json['images']]
    whole_images_in_dataset = [img for img in dataset_image_filepaths]
    whole_images_in_dataset = list(set(whole_images_in_dataset))
    return [img for img in whole_images_srcs if img in whole_images_in_dataset]

def prepare_results_for_mmdet(results_per_image_by_id):
    """
    Prepares results for MMDetection evaluation.
    
    Args:
        results_grouped (dict): Detections grouped by image ID.
        
    Returns:
        list: List of results formatted for MMDetection.
    """
    results_mmdet = []
    for per_img_results in results_per_image_by_id:
        per_img_results_mmdet = {}
        per_img_results_mmdet['image_id'] = per_img_results
        per_img_results_mmdet['pred_instances'] = {}
        per_img_results_mmdet['pred_instances']['bboxes'] = []
        per_img_results_mmdet['pred_instances']['labels'] = []
        per_img_results_mmdet['pred_instances']['scores'] = []
        
        for patch_results in results_per_image_by_id[per_img_results]:
            if len(patch_results['bboxes']) > 0:
                for i, bbox in enumerate(patch_results['bboxes']):
                    per_img_results_mmdet['pred_instances']['bboxes'].append(bbox)
                    per_img_results_mmdet['pred_instances']['labels'].append(int(patch_results['labels'][i]))
                    per_img_results_mmdet['pred_instances']['scores'].append(float(patch_results['scores'][i]))
                    
        # Make bboxes, labels, scores a torch tensor
        per_img_results_mmdet['pred_instances']['bboxes'] = torch.tensor(per_img_results_mmdet['pred_instances']['bboxes'])
        per_img_results_mmdet['pred_instances']['labels'] = torch.tensor(per_img_results_mmdet['pred_instances']['labels'])
        per_img_results_mmdet['pred_instances']['scores'] = torch.tensor(per_img_results_mmdet['pred_instances']['scores'])
        
        results_mmdet.append(per_img_results_mmdet)
        
    return results_mmdet

def output_results_for_eval(dataset, results_per_image_by_id, out_full):
    """
    Converts results to COCO format for evaluation.
    
    Args:
        dataset (Dataset): Whole image dataset loaded by MMDetection.
        results_per_image_by_id (dict): Results grouped by image ID.
        out_full (str): Path to the output directory.
    
    """
    results_out = []

    for image_id, results in results_per_image_by_id.items():
        for result in results:
            for i, bbox in enumerate(result['bboxes']):
                # Eval script expects COCO format
                bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                
                results_out.append({
                    'image_id': image_id,
                    'bbox': bbox_xywh,
                    'category_id': result['labels'][i],
                    'score': result['scores'][i],
                    'segmentation': [],
                    'iscrowd': 0,
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    'category_name': dataset.metainfo['classes'][result['labels'][i]]
                })
                
    dst = out_full + '/result.json'

    with open(dst, 'w') as f:
        json.dump(results_out, f, indent=4)
        
# https://stackoverflow.com/a/57555123
def writePNGwithdpi(im, filename, dpi=(300,300)):
   """Save the image as PNG with embedded dpi"""

   # Encode as PNG into memory
   retval, buffer = cv2.imencode(".png", im)
   s = buffer.tobytes()

   # Find start of IDAT chunk
   IDAToffset = s.find(b'IDAT') - 4

   # Create our lovely new pHYs chunk - https://www.w3.org/TR/2003/REC-PNG-20031110/#11pHYs
   pHYs = b'pHYs' + struct.pack('!IIc',int(dpi[0]/0.0254),int(dpi[1]/0.0254),b"\x01" ) 
   pHYs = struct.pack('!I',9) + pHYs + struct.pack('!I',zlib.crc32(pHYs))

   # Open output filename and write...
   # ... stuff preceding IDAT as created by OpenCV
   # ... new pHYs as created by us above
   # ... IDAT onwards as created by OpenCV
   with open(filename, "wb") as out:
      out.write(buffer[0:IDAToffset])
      out.write(pHYs)
      out.write(buffer[IDAToffset:])

# Modified from sahi.utils.cv.visualize_object_predictions
def visualize_object_predictions(
    image: np.array,
    object_prediction_list,
    rect_th: int = None,
    text_size: float = None,
    text_th: float = None,
    color: Colors = None,
    hide_labels: bool = False,
    hide_conf: bool = False,
    output_dir: Optional[str] = None,
    file_name: str = "prediction_visual",
    export_format: str = "png",
    text_colour: tuple = (255, 255, 255),
):
    """
    Visualizes prediction category names, bounding boxes over the source image
    and exports it to output folder.

    Args:
        image: source image
        object_prediction_list: a list of prediction.ObjectPrediction
        rect_th: rectangle thickness
        text_size: size of the category name over box
        text_th: text thickness
        color: a sahi Color object
        hide_labels: hide labels
        hide_conf: hide confidence
        output_dir: directory for resulting visualization to be exported
        file_name: exported file will be saved as: output_dir+file_name+".png"
        export_format: can be specified as 'jpg' or 'png'
        text_colour: color of the text
    """
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select predefined classwise color palette if not specified
    if color is None:
        colors = Colors()
    else:
        colors = color
        
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.003), 2)
    # set text_th for category names
    text_th = text_th or max(rect_th - 1, 1)
    # set text_size for category names
    text_size = text_size or rect_th / 3

    # add masks to image if present
    for object_prediction in object_prediction_list:
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()
        # visualize masks if present
        if object_prediction.mask is not None:
            # deepcopy mask so that original is not altered
            mask = object_prediction.mask.bool_mask
            # set color
            if colors is not None:
                color = colors(object_prediction.category.id)
            # draw mask
            rgb_mask = apply_color_mask(mask, color)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.6, 0)

    # add bboxes to image if present
    for object_prediction in object_prediction_list:
        # deepcopy object_prediction_list so that original is not altered
        object_prediction = object_prediction.deepcopy()

        bbox = object_prediction.bbox.to_xyxy()
        category_name = object_prediction.category.name
        score = object_prediction.score.value

        # set color
        if colors is not None:
            color = colors(object_prediction.category.id)
            
        # set bbox points
        point1, point2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        # visualize boxes
        cv2.rectangle(
            image,
            point1,
            point2,
            color=color,
            thickness=rect_th,
        )
        
        label = ""

        if not hide_labels:
            # arange bounding box text location
            label = f"{category_name}"

        if not hide_conf:
            label += f" {score:.2f}"

        box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[
            0
        ]  # label width, height
        outside = point1[1] - box_height - 3 >= 0  # label fits outside box
        point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
        # add bounding box text
        cv2.rectangle(image, point1, point2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
            0,
            text_size,
            text_colour,
            thickness=text_th,
        )

    # export if output_dir is present
    if output_dir is not None:
        # export image with predictions
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # save inference result
        save_path = str(Path(output_dir) / (file_name + "." + export_format))
        #cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        writePNGwithdpi(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), save_path, dpi=(300,300))
        
    elapsed_time = time.time() - elapsed_time
    return {"image": image, "elapsed_time": elapsed_time}


def visualise_results(whole_images_srcs,
                      results_per_image_by_id,
                      out_full,
                      image_ids,
                      dataset,
                      hide_labels = False,
                      hide_conf = False):
    
     # Generate colour map based on the whole dataset classes
    colour_cycle = plt.cm.tab20.colors
    colours = []
    # > 20 classes 
    for i, class_lables in enumerate(dataset.metainfo['classes']):
        colours.append(colour_cycle[i % len(colour_cycle)])
    # Make colour map RGB and tuples as required by SAHI Colors object
    colours = [(int(255 * colour[0]), int(255 * colour[1]), int(255 * colour[2])) for colour in colours]
    c = Colors()
    c.palette = colours
    
    # For each image...
    for image in whole_images_srcs:
        filename = image.split('/')[-1]
        image_id = image_ids[filename]
        stem = filename.split('.')[0]
        predictions = []
        # ... get the detections
        for result in results_per_image_by_id[image_id]:
            for i, bbox in enumerate(result['bboxes']):
                bbox = [int(x) for x in bbox]
                label = result['labels'][i]
                score = result['scores'][i]
                category_name = dataset.metainfo['classes'][label]
                # ... convert to SAHI ObjectPrediction
                prediction = ObjectPrediction(
                    bbox = bbox,
                    category_id = label,
                    score = score,
                    category_name = category_name
                )
                predictions.append(prediction)
            
        visualize_object_predictions(
            image = mmcv.imread(image, channel_order='rgb'),
            object_prediction_list = predictions,
            output_dir= out_full + '/visuals/',
            file_name = stem,
            export_format = 'png',
            color = c,
            rect_th = 1,
            text_size = 0.5,
            text_th = 1,
            text_colour = (255, 255, 255), 
            hide_labels= hide_labels,
            hide_conf = hide_conf
        )
        
def get_img_ids_from_json(coco_json):
    """
    Returns the image IDs from a COCO JSON.
    
    Args:
        coco_json (string) COCO JSON.
        
    Returns:
        dict: Dictionary of image IDs.
    """
    img_ids = {}
    for img in coco_json['images']:
        img_ids[img['file_name']] = img['id']
        
    return img_ids

def filename_key_to_id(results_per_image, image_ids):
    """
    Converts the key in results_per_image from filename to image ID.
    
    Args:
        results_per_image (dict): Dictionary of results.
        image_ids (dict): Dictionary of image IDs.
        
    Returns:
        dict: Dictionary of results with image IDs as keys.
    """
    results_per_image_by_id = {}
    for filename, results in results_per_image.items():
        image_id = image_ids[filename]
        results_per_image_by_id[image_id] = results
        
    return results_per_image_by_id

def get_per_patch_results(patch_bboxes, img, inference_model, bbox_conf_threshold):
    """
    Returns the results for each patch of the image.
    
    Args:
        patch_bboxes (list): List of bounding boxes for each patch.
        img (np.ndarray): Image to get detections for.
        inference_model (DetInferencer): MMDetection inference model.
        bbox_conf_threshold (float): Bounding box confidence threshold.
        
    Returns:
        list: List of results for each patch.
    """
    results_per_patch = []
    for patch_bbox in patch_bboxes:
        patch = crop_to_patch(img, patch_bbox)
        results = inference_model(patch)
        thresholded_results = threshold_results(results, bbox_conf_threshold)
        # Add the patch bounding box to the results dict so we can keep track of which patch the results came from
        thresholded_results['patch_bbox'] = patch_bbox
        results_per_patch.append(thresholded_results)
        
    return results_per_patch

def rescale_results_to_image(results_per_patch, patch_bboxes):
    """
    Rescales the bounding boxes to fit onto the whole image.
    
    Args:
        results_per_patch (list): List of results for each patch.
        patch_bboxes (list): List of bounding boxes for each patch.
    
    Returns:
        list: List of results with bounding boxes rescaled to the whole image.
    """
    for i, result in enumerate(results_per_patch):
            patch_bbox = patch_bboxes[i]
            x_min, y_min, x_max, y_max = patch_bbox
            
            if len(result['bboxes']) > 0:
                rescaled_bboxes = result['bboxes'] + np.array([x_min, y_min, x_min, y_min])
                result['bboxes'] = rescaled_bboxes.tolist()
    
    return results_per_patch


def main():
    args = parse_args()
    patch_size = tuple(map(int, args.patch_size.replace('(', '').replace(')', '').split(',')))

    test_json = load_results(args.json_src)
    whole_image_filepaths = get_whole_image_filepaths(args.img_src, test_json)
    
    # Get detections for each image in the dataset
    # using MMDetection thus boxes as xyxy
    inference_model = DetInferencer(args.config_src,
                                    args.checkpoint_src,
                                    device = 'cuda',
                                    show_progress= False)
    
    results_per_image = {}
    for image_src in whole_image_filepaths:
        filename = image_src.split('/')[-1]
        # Load image as BGR
        img = mmcv.imread(image_src)
        image_height, image_width = img.shape[:2]

        patch_bboxes = calculate_patch_bboxes(
            image_height=image_height,
            image_width=image_width,
            patch_height=patch_size[0],
            patch_width=patch_size[1],
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap
        )
    
        # Get detections for each patch of image
        results_per_patch = get_per_patch_results(patch_bboxes, img, inference_model, args.bbox_conf_threshold)

        # Rescale detections to fit onto the image
        results_per_patch = rescale_results_to_image(results_per_patch, patch_bboxes)
        
        results_per_image[filename] = results_per_patch
    
    # Get the image IDS for each image in results_per_image
    image_ids = get_img_ids_from_json(test_json)
    
    # Convert the key from filename to image_id
    results_per_image_by_id = filename_key_to_id(results_per_image, image_ids)
            
    results_mmdet = prepare_results_for_mmdet(results_per_image_by_id)
    
    dataset = build_whole_image_dataset(args.config_src, args.json_src)

    confusion_matrix = conf_mat(dataset, results_mmdet, score_thr=args.bbox_conf_threshold, tp_iou_thr=0.5)

    confusion_matrix, class_ordering = reorder_confusion_matrix(confusion_matrix, dataset, args.abundance_ordering)
    
    class_ordering['background'] = len(class_ordering)
    
    out_full = args.checkpoint_src.split('best')[0] + args.output_folder_name
    
    if not os.path.exists(out_full):
        os.makedirs(out_full)
    
    plot_confusion_matrix(confusion_matrix, class_ordering.keys(), show = False, 
                          title = 'Normalised Confusion Matrix\nNon-SAHI processed, ordered by abundance',
                          save_dir = out_full)
    
    output_results_for_eval(dataset, results_per_image_by_id, out_full)
    evaluate(args.json_src,
            out_full + '/result.json',
            out_full,
            type = 'bbox',
            max_detections=10000,
            classwise = True)
    
    visualise_results(whole_image_filepaths,
                      results_per_image_by_id,
                      out_full,
                      image_ids,
                      dataset)
    
if __name__ == '__main__':
    main()
