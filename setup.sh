#!/bin/bash

exec > setup_output.txt 2>&1
set -e 
set -o pipefail 

PROJECT_DIR="."
PYTHON_VERSION="3.8"
ENVIRONMENT="environment.yml"
ENV_NAME="wsbd"
SRC="mmdetection_additions"
DST="mmdetection"

echo "Checking for mamba..."
if ! command -v mamba &> /dev/null; then
    echo "Mamba not found. Please install it first: https://mamba.readthedocs.io/en/latest/"
    exit 1
fi

# Move into the project directory
pushd "$PROJECT_DIR"

echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
mamba create -f $ENVIRONMENT -y
. "$(conda info --base)/etc/profile.d/conda.sh"
. "$(conda info --base)/etc/profile.d/mamba.sh"
mamba activate $ENV_NAME

echo "Installing MMDetection..."
git clone --depth 1 --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e . -v
cd ..

echo "Merge the required additional files created for this project into MMDetection..."
cp -r "$SRC/configs/_base_/models/." "$DST/configs/_base_/models/"
cp -r "$SRC/configs/_base_/schedules/." "$DST/configs/_base_/schedules/"
cp "$SRC/configs/_base_/default_runtime.py" "$DST/configs/_base_/default_runtime.py"
cp -r "$SRC/tools/." "$DST/tools/"

echo "Downloading the Weddell Sea Bethic Dataset from the Polar Data Centre..."
mkdir -p "data"
wget "https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=1ba97e4b-efb7-460b-9f2d-90437e33ce09&output=zip.tree" -O "data/wsbd.zip"
unzip -q "data/wsbd.zip"
rm "data/wsbd.zip"
# wsbd.zip contains a directory, "The Weddell Sea Benthic Dataset: A computer vision-ready object detection dataset for in situ benthic biodiversity monitoring model development",
# which has the annotations and images subfolders needed.
# Move annotations and images to the `data` directory level.
mkdir -p "data/annotations"
mkdir -p "data/images"
mv "The Weddell Sea Benthic Dataset: A computer vision-ready object detection dataset for in situ benthic biodiversity monitoring model development/annotations/"* "data/annotations/"
mv "The Weddell Sea Benthic Dataset: A computer vision-ready object detection dataset for in situ benthic biodiversity monitoring model development/images/"* "data/images/"
rm -rf "The Weddell Sea Benthic Dataset: A computer vision-ready object detection dataset for in situ benthic biodiversity monitoring model development"

echo "Downloading the best model weights for the Weddell Sea Benthic Dataset from the Polar Data Centre..."
mkdir -p "best_weights"
wget https://ramadda.data.bas.ac.uk/repository/entry/get/wsbd_model_weights.pth?entryid=synth%3Ab2874f3f-285d-4ae6-9bb4-6bfe3eacbfff%3AL3dzYmRfbW9kZWxfd2VpZ2h0cy5wdGg%3D -O "best_weights/wsbd_model_weights.pth"

echo "Downloading the required initial model weights from the MMDetection model zoo..."
mkdir -p "openmmlab_pretrained_weights"

# Faster R-CNN: The URL does not have the file name encoded, so rename it:
wget "https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r50_fpn_1x_coco" -O "openmmlab_pretrained_weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

wget -P "openmmlab_pretrained_weights" https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth
wget -P "openmmlab_pretrained_weights" https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth
wget -P "openmmlab_pretrained_weights" https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr_r50_16xb2-50e_coco/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth
wget -P "openmmlab_pretrained_weights" https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_improved_8xb2-12e_coco/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth
wget -P "openmmlab_pretrained_weights" https://download.openmmlab.com/mmdetection/v3.0/conditional_detr/conditional-detr_r50_8xb2-50e_coco/conditional-detr_r50_8xb2-50e_coco_20221121_180202-c83a1dc0.pth

echo "Finished!"
echo "To activate the created environment, run: mamba activate $ENV_NAME"
echo "Run code from the 'mmdetection' directory, not the 'mmdetection_additions' directory."
echo "Before running any code, please make sure to modify the file paths at: /mmdetection/tools/paths.env"

# Return to the original directory
popd