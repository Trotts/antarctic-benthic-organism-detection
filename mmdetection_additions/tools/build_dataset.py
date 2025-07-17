"""
Builds a patched dataset for model training/eval.
"""

from pathlib import Path
import pandas as pd
from multiprocessing import Pool
from functools import partial
import albumentations as A
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import json
import argparse

CC_LICENSE = {
    "id": 1,
    "name": "Creative Commons Attribution 4.0 International License",
    "url": "https://creativecommons.org/licenses/by/4.0/",
}


def load_coco_annotations_as_df(annotations_path):
    """
    Loads COCO annotations as a DataFrame.

    Args:
        annotations_path (str): Path to COCO annotations file.

    Returns:
        df (pd.DataFrame): DataFrame containing annotations.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    classes_df = pd.DataFrame(coco["categories"])
    annotations_df = pd.DataFrame(coco["annotations"])
    images_df = pd.DataFrame(coco["images"])

    split_bboxes_df = pd.DataFrame(
        annotations_df["bbox"].tolist(), columns=["xmin", "ymin", "w", "h"]
    )
    annotations_df = pd.concat([annotations_df, split_bboxes_df], axis=1)

    df = pd.merge(annotations_df, images_df, left_on="image_id", right_on="id")
    df = pd.merge(df, classes_df, left_on="category_id", right_on="id")
    df = df[["file_name", "xmin", "ymin", "w", "h", "name"]]

    df["xmax"] = df["xmin"] + df["w"]
    df["ymax"] = df["ymin"] + df["h"]

    name_to_id_lookup = {n: i for i, n in enumerate(df.name.unique())}
    df["category_id"] = df["name"].apply(lambda n: name_to_id_lookup[n])

    return df


def get_image_size(file_name, images_path):
    """
    Load an image and get its size.

    Args:
        file_name (str): Name of image file.
        images_path (str): Path to directory containing images.

    Returns:
        dict: Dictionary containing image size information.
    """

    image = Image.open(images_path / file_name)
    return {
        "file_name": file_name,
        "image_height": image.height,
        "image_width": image.width,
    }


def get_image_sizes_df(images_path, file_names):
    """
    Gets the size of each image in a directory

    Args:
        images_path (str): Path to directory containing images.
        file_names (list): List of file names to get sizes for

    Returns:
        sizes_df (pd.DataFrame): DataFrame containing image sizes.
    """
    pool = Pool(processes=len(os.sched_getaffinity(0)))
    image_sizes = pool.map(partial(get_image_size, images_path=images_path), file_names)
    sizes_df = pd.DataFrame(image_sizes)
    return sizes_df


def calculate_patch_bboxes(
    image_height: int,
    image_width: int,
    patch_height: int = 512,
    patch_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
):
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping patches according to the height and width provided.

    Args:
        image_height (int): Height of image.
        image_width (int): Width of image.
        patch_height (int): Height of patch.
        patch_width (int): Width of patch.
        overlap_height_ratio (float): Ratio of overlap between patches in the height dimension.
        overlap_width_ratio (float): Ratio of overlap between patches in the width dimension.

    Returns:
        patch_bboxes (list[list[int]]): List of patches as bounding boxes in xyxy format.
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


def contains_object(annotations_df, image_patch_row, min_visibility=0.1):
    """
    Checks if a patch contains an object.

    Args:
        annotations_df (pd.DataFrame): DataFrame containing annotations.
        image_patch_row (pd.Series): Row of DataFrame containing patch information.
        min_visibility (float): Minimum visibility of bounding box to keep when patched.
        min_area (float): Minimum area of bounding box to keep when patched.

    Returns:
        bool: Whether the patch contains an object.
    """
    xyxy_bboxes = annotations_df.query("file_name == @image_patch_row.file_name")[
        ["xmin", "ymin", "xmax", "ymax"]
    ].values
    patch_bbox = image_patch_row[["xmin", "ymin", "xmax", "ymax"]].values

    transforms = A.Compose(
        [A.Crop(*patch_bbox)],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_visibility=min_visibility
        ),
    )

    transformed = transforms(
        image=np.ones((image_patch_row.image_height, image_patch_row.image_width, 3)),
        bboxes=xyxy_bboxes,
        labels=np.ones(len(xyxy_bboxes)),
    )

    transformed_bboxes = transformed["bboxes"]

    return len(transformed_bboxes) > 0


def create_image_patches_df(
    images_path,
    annotations_df,
    patch_height: int = 512,
    patch_width: int = 512,
    overlap_height_ratio: float = 0.0,
    overlap_width_ratio: float = 0.0,
    min_bbox_visibility: float = 0.1,
):
    """
    Creates a DataFrame containing locations of image patches.

    Args:
        images_path (str): Path to directory containing images.
        annotations_df (pd.DataFrame): DataFrame containing annotations.
        patch_height (int): Height of patch.
        patch_width (int): Width of patch.
        overlap_height_ratio (float): Ratio of overlap between patches in the height dimension.
        overlap_width_ratio (float): Ratio of overlap between patches in the width dimension.
        min_bbox_visibility (float): Minimum visibility of bounding box to keep when patched as ratio.
    """
    sizes_df = get_image_sizes_df(images_path, annotations_df.file_name.unique())
    sizes_df["patches"] = sizes_df.apply(
        lambda row: calculate_patch_bboxes(
            row.image_height,
            row.image_width,
            patch_height,
            patch_width,
            overlap_height_ratio,
            overlap_width_ratio,
        ),
        axis=1,
    )

    patches_row_df = (
        sizes_df[["file_name", "patches"]]
        .explode("patches")
        .rename(columns={"patches": "patch"})
    )
    patches_row_df = pd.DataFrame(
        patches_row_df.patch.tolist(),
        columns=["xmin", "ymin", "xmax", "ymax"],
        index=patches_row_df.file_name,
    ).reset_index()

    image_patches_df = pd.merge(
        patches_row_df,
        sizes_df[["file_name", "image_height", "image_width"]],
        on="file_name",
        how="inner",
    )

    image_patches_df["contains_object"] = image_patches_df.apply(
        partial(contains_object, annotations_df, min_visibility=min_bbox_visibility),
        axis=1,
    )

    image_patches_df.reset_index(inplace=True)
    image_patches_df.rename(columns={"index": "patch_id"}, inplace=True)

    return image_patches_df


class DatasetAdaptor:
    """
    Dataset adaptor for loading images and labels.

    Adapted from https://gist.github.com/Chris-hughes10/6736427bbaa45ddffa0095efd867d027#file-dataset_adaptor-py
    """

    def __init__(self, images_dir_path, annotations_dataframe):
        """
        Args:
            images_dir_path (str): Path to directory containing images.
            annotations_dataframe (pd.DataFrame): DataFrame containing annotations.
        """
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe

        self.file_name_to_idx = {
            image_id: idx
            for idx, image_id in enumerate(self.annotations_df.file_name.unique())
        }
        self.idx_to_file_name = {v: k for k, v in self.file_name_to_idx.items()}

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.file_name_to_idx)

    def __getitem__(self, idx: int):
        """
        Loads image and labels for a given index.

        Args:
            idx (int): Index of image to load.

        Returns:
            image (np.array): Image as a numpy array.
            xyxy_bbox (np.array): Bounding boxes in xyxy format (xmin, ymin, xmax, ymax).
            class_labels (np.array): Class labels for each bounding box.
            idx (int): Index of image.
        """
        file_name = self.idx_to_file_name[idx]
        image = Image.open(self.images_dir_path / file_name)

        xyxy_bbox = self.annotations_df.query("file_name == @file_name")[
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = self.annotations_df.query("file_name == @file_name")[
            "category_id"
        ].values

        return np.array(image), xyxy_bbox, class_labels, idx


class ImagePatchDetectionDataset:
    """
    Dataset of image patches for object detection.

    Adapted from https://gist.github.com/Chris-hughes10/4de795a2aff8c2c1944a29263ffef65d#file-slices_dataset_v2-py
    """

    def __init__(
        self,
        ds_adaptor: DatasetAdaptor,
        patches_df: pd.DataFrame,
        as_patch: bool = True,
        pad_patch_to_height: int = 250,
        pad_path_to_width: int = 250,
        sequential_cropping: bool = True,
        transforms=None,
        min_bbox_visibility: float = 0.1,
        scale: tuple = (0.01, 0.03),
    ):
        """
        Args:
            ds_adaptor (DatasetAdaptor): DatasetAdaptor object to load images and labels.
            patches_df (pd.DataFrame): DataFrame containing patch information.
            as_patch (bool): Whether to return the whole image or just the patch.
            pad_patch_to_height (int): Height to pad patch to. Only applies if as_patch is True.
            pad_path_to_width (int): Width to pad patch to. Only applies if as_patch is True.
            sequential_cropping (bool): Whether to crop sequentially (True) or randomly (False). Only applies if as_patch is True.
            transforms (list): List of albumentations transforms to apply to images.
                               If sequential_cropping is True, a RandomResizedCrop transform will be added to the start of the list.
            min_bbox_visibility (float): Minimum visibility of bounding box to keep when patched.
            scale (tuple): Percentage bounds to use for RandomResizedCrop transform. Only applies if sequential_cropping is False.
                           e.g. (0.01, 0.03) means the patch will be between 1% and 3% of the image size.
        """
        self.ds_adaptor = ds_adaptor
        self.patches_df = patches_df
        self.as_patch = as_patch
        self.pad_patch_to_height = pad_patch_to_height
        self.pad_path_to_width = pad_path_to_width
        self.sequential_cropping = sequential_cropping
        self.transforms = transforms
        self.min_bbox_visibility = min_bbox_visibility
        self.scale = scale

    def __len__(self) -> int:
        """Returns the number of patches in the dataset."""
        return len(self.patches_df)

    def _apply_transforms(self, transforms_list, image, bboxes, class_labels):
        """
        Applies a list of transforms to an image and bounding boxes.

        Args:
            transforms_list (list): List of albumentations transforms to apply to images.
            image (np.ndarray): Image to apply transforms to.
            bboxes (np.ndarray): Bounding boxes to apply transforms to.
            class_labels (np.ndarray): Class labels to apply transforms to.

        Returns:
            image (np.ndarray): Transformed image.
            bboxes (np.ndarray): Transformed bounding boxes.
            class_labels (np.ndarray): Transformed class labels.
        """
        transforms = A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_visibility=self.min_bbox_visibility,
            ),
        )

        transformed = transforms(image=image, bboxes=bboxes, labels=class_labels)

        image = transformed["image"]
        bboxes = np.array(transformed["bboxes"])
        class_labels = np.array(transformed["labels"])

        return image, bboxes, class_labels

    def create_sequential_cropping_transform(self, patch_bbox_corners):
        """
        Generates a list of transforms to sequentially crop an image to a patch.
        When sequentially cropping, the patch is generated using the coordinates given by the current row in the patches_df.

        Args:
            patch_bbox_corners (np.ndarray): Coordinates of the patch to crop to.

        Returns:
            transforms (list): List of albumentations transforms to apply to images.
        """
        return [
            A.Crop(*patch_bbox_corners),
            A.PadIfNeeded(
                min_height=self.pad_patch_to_height,
                min_width=self.pad_path_to_width,
                border_mode=0,
            ),
        ]

    def create_random_crop_transform(self):
        """
        Generates a list of transforms to randomly crop an image to a patch.
        When randomly cropping, the patch is generated by randomly selecting an area of the image,
        based on some percentage of the image size (scale). This is then resized to the patch size.

        Args:
            None

        Returns:
            transforms (list): List of albumentations transforms to apply to images.
        """
        return [
            A.RandomResizedCrop(
                self.pad_patch_to_height, self.pad_path_to_width, scale=self.scale
            )
        ]

    def __getitem__(self, idx: int):
        """
        Loads an image/patch and labels for a given index.

        Args:
            idx (int): Loads the image found at this index in the patches_df.

        Returns:
            image (np.array): Image as a numpy array.
            bboxes (list): Bounding boxes in xyxy format (xmin, ymin, xmax, ymax).
            class_labels (list): Class labels for each bounding box.
            file_name (str): Name of image loaded.
        """
        row = self.patches_df.iloc[idx]
        file_name = row.file_name
        adaptor_idx = self.ds_adaptor.file_name_to_idx[file_name]

        image, bboxes, class_labels, _ = self.ds_adaptor[adaptor_idx]

        transforms = []

        if self.as_patch:
            if self.sequential_cropping:
                patch_bbox_corners = row[["xmin", "ymin", "xmax", "ymax"]].values
                transforms.extend(
                    self.create_sequential_cropping_transform(patch_bbox_corners)
                )
            else:
                transforms.extend(self.create_random_crop_transform())

        if self.transforms:
            transforms.extend(self.transforms)

        image, bboxes, class_labels = self._apply_transforms(
            transforms, image, bboxes, class_labels
        )

        return image, bboxes.tolist(), class_labels.tolist(), file_name


class COCOGenerator:
    """
    Generate COCO formatted data from the located bounding boxes.

    Parameters:
    None

    Attributes:
    filename_id_map (dict): A mapping of filenames to filename IDs
    category_id_maps (dict): A mapping of category names to category IDs
    license_info (list): The license information for the dataset. May be multiple licenses.
    merged_coco_data (dict): The merged data for an image in COCO format
    """

    def __init__(self):
        """
        Initialise the COCOGenerator.
        """
        self.filename_id_map = {}
        self.category_id_maps = {}
        self.license_info = {}
        self.merged_coco_data = {
            "info": {
                "description": "Initial Benthic Dataset in COCO Format",
                "url": "",
                "version": "1.0",
                "year": 2024,
                "contributor": "Cameron Trotter (cater@bas.ac.uk)",
                "date_created": "2023/11/21",
            },
            "images": [],
            "annotations": [],
            "categories": [],
            "licenses": [],
        }

    def __str__(self):
        return self.merged_coco_data

    def _convert_to_coco_format(self, image_info, image_boxes):
        """
        Convert the located bounding boxes to COCO format.

        Parameters:
        image_filename (str): The filename of the image
        image_boxes (list): The list of bounding boxes for the image

        Returns:
        dict: The COCO formatted data
        """
        image_filepath, image_width, image_height, license = image_info

        license_id = license["id"]
        split_filepath = image_filepath.split("/")
        image_filename = split_filepath[-1]

        if license_id not in self.license_info:
            self.license_info[license_id] = license
            self.merged_coco_data["licenses"].append(license)

        image_dict = {
            "id": len(self.filename_id_map) + 1,
            "file_name": image_filename,
            "width": image_width,
            "height": image_height,
            "license": license_id,
            "coco_url": "None",
            "date_captured": "None",
        }

        self.filename_id_map[image_filename] = image_dict["id"]

        annotation_id = 0
        annotations = []
        categories = []

        for bbox in image_boxes:
            category_name, top_left_x, top_left_y, box_width, box_height = bbox

            if category_name not in self.category_id_maps:
                category_id = len(self.category_id_maps) + 1
                self.category_id_maps[category_name] = category_id

                categories.append(
                    {"id": self.category_id_maps[category_name], "name": category_name}
                )

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_dict["id"],
                    "category_id": self.category_id_maps[category_name],
                    "segmentation": [],
                    "bbox": [top_left_x, top_left_y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )

            annotation_id = annotation_id + 1

        return {
            "images": [image_dict],
            "annotations": annotations,
            "categories": categories,
        }

    def process_images(self, image_data, export_location=None):
        """
        Process the located bounding boxes for each image.

        Parameters:
        image_data (tuple): The image data, consisting of the image information and the located bounding boxes
        export_location (str): The path to the output file. If None, return the COCO formatted data instead.
        """

        for image_info, image_boxes in image_data:
            coco_data = self._convert_to_coco_format(image_info, image_boxes)

            # Merge the data directly into the combined JSON
            self.merged_coco_data["images"].extend(coco_data["images"])
            self.merged_coco_data["annotations"].extend(coco_data["annotations"])
            self.merged_coco_data["categories"].extend(coco_data["categories"])

        # re-ID the annotations so they are sequential between images
        for i, ann in enumerate(self.merged_coco_data["annotations"]):
            ann["id"] = i

        if export_location:
            self._export(export_location)
        else:
            return self.merged_coco_data

    def _export(self, output_file):
        """
        Write the COCO formatted data to a file.

        Parameters:
        output_file (str): The path to the output file
        """
        with open(output_file, "w") as f:
            json.dump(self.merged_coco_data, f, indent=4)


def class_ids_to_labels(ids_list, convertion_df):
    """
    Converts a list of class ids to class labels.

    Args:
        ids_list (list[int]): List of class ids.
        convertion_df (pd.DataFrame): DataFrame containing class id to label mapping.

    Returns:
        labels_list (list[str]): List of class labels.
    """
    labels_list = [
        convertion_df.query("category_id == @id")["name"].values[0] for id in ids_list
    ]
    return labels_list


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build a patched dataset for model training/eval"
    )
    parser.add_argument(
        "images_path", type=str, help="Path to directory containing whole images"
    )
    parser.add_argument(
        "annotations_dir", type=str, help="Path to whole COCO annotations dir"
    )
    parser.add_argument(
        "output_dir", type=str, help="Path to output parent dir (e.g. the patches dir)"
    )
    parser.add_argument(
        "patch_size", type=str, help="Patch size to be used as tuple string"
    )
    parser.add_argument(
        "overlap", type=float, help="Overlap to be used between patches"
    )
    parser.add_argument(
        "min_bbox_visibility",
        type=float,
        help="Minimum visibility of bounding box to keep when patched as a ratio",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    patch_size = tuple(
        map(int, args.patch_size.replace("(", "").replace(")", "").split(","))
    )

    dataset_name = (
        str(patch_size[0])
        + "_"
        + str(patch_size[1])
        + "_overlap_"
        + str(args.overlap).replace(".", "")
        + "_minviz_"
        + str(args.min_bbox_visibility).replace(".", "")
    )

    dst = os.path.join(args.output_dir, dataset_name)
    os.makedirs(dst, exist_ok=True)

    images_path = Path(args.images_path)
    annotations_path = Path(args.annotations_dir)
    whole_annotations_json = annotations_path / "dataset.json"

    annotations_df = load_coco_annotations_as_df(whole_annotations_json)
    patches_df = create_image_patches_df(
        images_path,
        annotations_df,
        patch_height=patch_size[0],
        patch_width=patch_size[1],
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        min_bbox_visibility=args.min_bbox_visibility,
    )

    ds = DatasetAdaptor(images_path, annotations_df)
    patches = ImagePatchDetectionDataset(ds, patches_df, sequential_cropping=True)

    class_list_df = pd.DataFrame(
        annotations_df[["category_id", "name"]].drop_duplicates()
    )
    class_list_df.reset_index(drop=True, inplace=True)

    image_data = []

    patch_dst = Path(dst) / "images/"
    annotations_dst = Path(dst) / "annotations/"
    os.makedirs(patch_dst, exist_ok=True)
    os.makedirs(annotations_dst, exist_ok=True)

    train_json_src = annotations_path / "dataset_train.json"
    val_json_src = annotations_path / "dataset_val.json"
    test_json_src = annotations_path / "dataset_test.json"

    train_json = json.load(open(train_json_src))
    val_json = json.load(open(val_json_src))
    test_json = json.load(open(test_json_src))

    for i in tqdm(range(len(patches))):
        image, bboxes, labels, filename = patches[i]
        labels = class_ids_to_labels(labels, class_list_df)

        filename_stem, extension = filename.split(".")
        patch_filename = f"{filename_stem}_patch_{i}.{extension}"

        ## Save patch out
        to_save = Image.fromarray(image)
        to_save.save(patch_dst / patch_filename)
        patch_width, patch_height = to_save.size

        ## Combine bboxes and labels into a single list in COCO format
        coco_bboxes = [
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]] for bbox in bboxes
        ]
        labels_with_bboxes = [
            [label, *coco_bboxes[i]] for i, label in enumerate(labels)
        ]

        ## Append for saving to a whole dataset JSON
        image_data.append(
            [
                [patch_filename, patch_width, patch_height, CC_LICENSE],
                labels_with_bboxes,
            ]
        )

    coco_gen = COCOGenerator()
    coco_gen.process_images(
        image_data, os.path.join(annotations_dst, "dataset_patches.json")
    )

    # Split the dataset into train, val, and test based on the original split
    whole_json_src = os.path.join(annotations_dst, "dataset_patches.json")
    with open(whole_json_src, "r") as f:
        whole_json = json.load(f)

    train_json = whole_json.copy()
    val_json = whole_json.copy()
    test_json = whole_json.copy()

    whole_train_json_src = annotations_path / "dataset_train.json"
    whole_val_json_src = annotations_path / "dataset_val.json"
    whole_test_json_src = annotations_path / "dataset_test.json"

    whole_train_json = json.load(open(whole_train_json_src))
    whole_val_json = json.load(open(whole_val_json_src))
    whole_test_json = json.load(open(whole_test_json_src))

    train_stems = [
        Path(image["file_name"]).stem for image in whole_train_json["images"]
    ]
    train_json["images"] = [
        img
        for img in whole_json["images"]
        if Path(img["file_name"]).stem.split("_patch")[0] in train_stems
    ]
    train_json["annotations"] = [
        ann
        for ann in whole_json["annotations"]
        if ann["image_id"] in [img["id"] for img in train_json["images"]]
    ]

    val_stems = [Path(image["file_name"]).stem for image in whole_val_json["images"]]
    val_json["images"] = [
        img
        for img in whole_json["images"]
        if Path(img["file_name"]).stem.split("_patch")[0] in val_stems
    ]
    val_json["annotations"] = [
        ann
        for ann in whole_json["annotations"]
        if ann["image_id"] in [img["id"] for img in val_json["images"]]
    ]

    test_stems = [Path(image["file_name"]).stem for image in whole_test_json["images"]]
    test_json["images"] = [
        img
        for img in whole_json["images"]
        if Path(img["file_name"]).stem.split("_patch")[0] in test_stems
    ]
    test_json["annotations"] = [
        ann
        for ann in whole_json["annotations"]
        if ann["image_id"] in [img["id"] for img in test_json["images"]]
    ]

    train_json_dst = os.path.join(annotations_dst, "dataset_train_patches.json")
    val_json_dst = os.path.join(annotations_dst, "dataset_val_patches.json")
    test_json_dst = os.path.join(annotations_dst, "dataset_test_patches.json")
    all_json_dst = os.path.join(annotations_dst, "dataset_patches.json")

    with open(train_json_dst, "w") as f:
        json.dump(train_json, f, indent=4)
    with open(val_json_dst, "w") as f:
        json.dump(val_json, f, indent=4)
    with open(test_json_dst, "w") as f:
        json.dump(test_json, f, indent=4)
    with open(all_json_dst, "w") as f:
        json.dump(whole_json, f, indent=4)

    # Ensure the IDs are consistent with the whole image dataset
    whole_annotations = json.load(open(whole_annotations_json))
    whole_annotations_cats = whole_annotations["categories"]

    whole_class_list_df = pd.DataFrame(whole_annotations_cats)
    whole_class_list_df.reset_index(drop=True, inplace=True)

    whole_image_class_ordering = whole_class_list_df.to_dict(orient="records")

    patches_json_srcs = [train_json_dst, val_json_dst, test_json_dst, all_json_dst]

    for src in patches_json_srcs:
        filename = Path(src).name
        # Get current annotation classes
        with open(src, "r") as f:
            current_json = json.load(f)

        current_ordering = current_json["categories"]

        # Replace the current IDs with the whole image class ordering IDs
        new_ordering = []
        for i, cls in enumerate(whole_image_class_ordering):
            cls["id"] = i
            new_ordering.append(cls)

        current_json["categories"] = new_ordering

        # Add supercategory key to each category
        for cat in current_json["categories"]:
            cat["supercategory"] = cat["name"]

        # Put id key first in the categories list
        current_json["categories"] = [
            {k: cat[k] for k in ["id", "name", "supercategory"]}
            for cat in current_json["categories"]
        ]

        # Adjust the annotations to match the new class ordering
        for ann in current_json["annotations"]:
            class_name = current_ordering[ann["category_id"] - 1]["name"]
            new_cat_for_ann = [
                cat["id"] for cat in new_ordering if cat["name"] == class_name
            ][0]
            ann["category_id"] = new_cat_for_ann

        # Sort the JSON's categories list
        current_json["categories"] = sorted(
            current_json["categories"], key=lambda x: x["id"]
        )

        # Del the category_id key from the categories list
        for cat in current_json["categories"]:
            if "category_id" in cat:
                del cat["category_id"]

        # Save the updated JSON
        with open(annotations_dst / filename, "w") as f:
            json.dump(current_json, f, indent=4)


if __name__ == "__main__":
    main()
