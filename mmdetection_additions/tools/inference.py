"""
Loads an MMDetection model  plus trained weights and performs inference on images using SAHI.
Exports results in COCO format and optionally visualises them.
"""

import argparse
from sahi_eval import load_model
from sahi.predict import get_sliced_prediction
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as patheffects


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="MMDetection SAHI Inference")
    parser.add_argument("config", help="Path to the MMDetection config file")
    parser.add_argument("checkpoint", help="Path to the model checkpoint file")
    parser.add_argument("input", help="Path to the input image file/directory")
    parser.add_argument(
        "output",
        type=str,
        default="output",
        help="Directory to save results and optional visualisations",
    )
    parser.add_argument("patch_size", help="Patch size used for training")
    parser.add_argument(
        "--bbox-conf-threshold",
        type=float,
        default=0.6,
        help="Bounding box confidence threshold for filtering predictions",
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5, help="Overlap ratio for patching images"
    )
    parser.add_argument(
        "--post-process-match-threshold",
        type=float,
        default=0.2,
        help="Threshold for post-processing matching",
    )
    parser.add_argument(
        "--post-process-algo",
        type=str,
        default="NMM",
        help="Post-processing algorithm to use",
    )
    parser.add_argument(
        "--visualise", action="store_true", help="Whether to visualise the results"
    )

    return parser.parse_args()


def run_inference(
    model,
    input_path,
    patch_height,
    patch_width,
    overlap,
    post_process_type,
    post_process_match_threshold,
):
    """
    Runs inference on the input image.

    Args:
        model (AutoDetectionModel): The SAHI detection model.
        input_path (str): Path to the input image file or directory.
        patch_height (int): Height of each patch for SAHI inference.
        patch_width (int): Width of each patch for SAHI inference.
        overlap (float): Overlap ratio for patching images.
        post_process_type (str): Type of post-processing algorithm to use.
        post_process_match_threshold (float): Threshold for post-processing matching.
    Returns:
        list: List of results in COCO format.
    """

    results_sahi = get_sliced_prediction(
        input_path,
        model,
        slice_height=patch_height,
        slice_width=patch_width,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type=post_process_type,
        postprocess_match_threshold=post_process_match_threshold,
    )

    results_sahi_coco = results_sahi.to_coco_annotations()

    return results_sahi_coco


def export_coco_visualisations(
    results_sahi,
    file_path,
    ids_to_class,
    palette,
    output_dir,
    figsize=(30, 30),
    fontsize=12,
):
    """
    Exports visualisations of the results in COCO format.
    Args:
        results_sahi (list): List of results in COCO format.
        file_path (str): Path to the input image file.
        ids_to_class (dict): Mapping from class IDs to class names.
        palette (list): List of RGB tuples for class colours.
        output_dir (str): Directory to save the visualisations.
        figsize (tuple): Size of the figure for visualisation.
        fontsize (int): Font size for class labels in the visualisation.
    """
    img = Image.open(file_path)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img)
    for result in results_sahi:
        x_min, y_min, width, height = result["bbox"]
        label_id = result["category_id"]
        colour = palette[label_id]
        rect = Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=1,
            edgecolor=colour,
            facecolor="none",
        )
        ax.add_patch(rect)
        class_label = ids_to_class[label_id]
        if "score" in result:
            score = f'{result["score"]:.2f}'
            out = class_label + ": " + score
        else:
            out = class_label
        ax.text(
            x_min,
            y_min - 2,
            out,
            color=colour,
            path_effects=[
                patheffects.withStroke(
                    linewidth=2, foreground="black", capstyle="round"
                )
            ],
            fontsize=fontsize,
        )
    ax.axis("off")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_visualised.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def extract_classes_and_palette(config_src):
    """
    Extracts the classes and palette lists from the config content string.
    Args:
        config_src (str): The path of the config file.
    Returns:
        tuple: (classes_to_ids, palette)
    """
    with open(config_src, "r") as f:
        config_content = f.read()

    metainfo_start = config_content.find("metainfo = ")
    if metainfo_start == -1:
        raise ValueError("metainfo not found in config content")
    metainfo_end = config_content.find("])", metainfo_start) + 2
    metainfo_content = config_content[metainfo_start:metainfo_end]
    metainfo_content = metainfo_content.replace("metainfo = ", "")
    data = eval(metainfo_content)
    classes = data["classes"]
    ids_to_classes = {i: cls for i, cls in enumerate(classes)}

    palette = data["palette"]
    palette = [[c / 255.0 for c in color] for color in palette]

    return ids_to_classes, palette


def export_results_to_json(results, output_dir):
    """
    Exports the results to a JSON file.
    Args:
        results (list): List of results in COCO format.
        output_dir (str): Directory to save the JSON file.
    """
    output_json_path = os.path.join(output_dir, "results.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results exported to {output_json_path}")


def main():

    args = parse_args()
    patch_size = tuple(
        map(int, args.patch_size.replace("(", "").replace(")", "").split(","))
    )

    model = load_model(args.config, args.checkpoint, args.bbox_conf_threshold)

    results_sahi_coco = run_inference(
        model,
        args.input,
        patch_size[0],
        patch_size[1],
        args.overlap,
        args.post_process_algo,
        args.post_process_match_threshold,
    )

    export_results_to_json(results_sahi_coco, args.output)

    if args.visualise:
        config_classes_to_ids, config_palette = extract_classes_and_palette(args.config)
        export_coco_visualisations(
            results_sahi_coco,
            args.input,
            config_classes_to_ids,
            config_palette,
            args.output,
        )


if __name__ == "__main__":
    main()
