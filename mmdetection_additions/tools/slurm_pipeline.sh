#!/usr/bin/env bash

# Full pipeline for training and evaluating an MMDetection model with SAHI using SLURM.
# Usage: sbatch slurm_pipeline.sh
# Make sure to edit below before running.
# Also edit mmdetection/tools/build_config.py lines 15-16.

#SBATCH --time=14-00:00:00
#SBATCH --partition= # partition name
#SBATCH --account= # account name
#SBATCH --gres= # GPU type and number
#SBATCH --mem-per-gpu= # Require XGB of memory per GPU

#SBATCH --output=job_log/run_pipeline.%j.%N.out
#SBATCH --error=job_log/run_pipeline.%j.%N.err 
#SBATCH --job-name=run_pipeline

# Path to the repository root directory
REPOSITORY_PATH='/PATH/TO/YOUR/REPOSITORY'
# Path to a visible cache for mmdetection, mmengine, etc.
CACHE_PATH='/PATH/TO/YOUR/CACHE'
# Miniforge3 installation path
MINIFORGE_PATH='/PATH/TO/YOUR/MINIFORGE3'

# Path to parent dir for saving models, configs, logs, etc.
work_dir_path="${REPOSITORY_PATH}/output"
# Path to the patches directory
patches_dir="${REPOSITORY_PATH}/patches" # Path to the patches directory
# Path to the whole data directory (images and annotations parent)
whole_data_src="${REPOSITORY_PATH}/data" # Path to the whole data directory (images and annotations parent)
# Path to the abundance ordering file
abundance="${REPOSITORY_PATH}/abundance_ordering/abundance_ordering.pkl" # Path to the abundance ordering file

# Name of the directory to save models, configs, logs, etc. in the work_dir_path
tests_dir_name='pipeline'
# Options: faster_rcnn, retinanet, cascade_rcnn, deformable-detr,
# dino_r50, codetr
architecture='deformable-detr'
# Best params as default
patch_size='(500,500)'
overlap=0.5
min_bbox_visibility=0.25
postprocess_type='NMM'
postprocess_match_threshold=0.2
augs='spatial'
pretrained=True
conf_threshold=0.6
batch_size=12

# set caches so they can be seen by the nodes (defaults @ ~/.config can't)
echo "Setting caches"
export XDG_CACHE_HOME=${CACHE_PATH}
export TORCH_HOME=${CACHE_PATH}
export MMENGINE_HOME=${CACHE_PATH}
export MPLCONFIGDIR=${CACHE_PATH}

# Load openmmlab mamba environment
echo "Loading mamba environment"
env_name='wsbd'
. ${MINIFORGE_PATH}/etc/profile.d/conda.sh
. ${MINIFORGE_PATH}/etc/profile.d/mamba.sh
mamba activate ${env_name}

# Determine the name given to the dataset by build_dataset based on the values above
patch_size_cleaned=$(echo $patch_size | tr -d '()' | tr ',' '_')
# Use it as the model name also
model_name="${patch_size_cleaned}_overlap_${overlap//./}_minviz_${min_bbox_visibility//./}"
# Add on additonal info here as needed
model_name_with_augs="${model_name}_${augs}_${architecture}"

if [ $pretrained = True ]; then
    model_name_with_augs="${model_name_with_augs}_pretrained"
fi

test_dir_src="${work_dir_path}/${tests_dir_name}"
src="${test_dir_src}/${model_name_with_augs}"

if [ ! -d $work_dir_path ]; then
    mkdir -p $work_dir_path
fi

if [ ! -d $test_dir_src ]; then
    mkdir $test_dir_src
fi

if [ ! -d $src ]; then
    mkdir $src
fi

whole_images_src="${whole_data_src}/images" 
whole_annotations_dir="${whole_data_src}/annotations"

############ GENERATE DATASET #############
echo "Generating dataset"

python -u ${REPOSITORY_PATH}/mmdetection/tools/build_dataset.py\
    $whole_images_src\
    $whole_annotations_dir\
    $patches_dir\
    $patch_size\
    $overlap\
    $min_bbox_visibility

patch_dataset_src="${patches_dir}/${model_name}/annotations"

############# GENERATE CONFIG #############
echo "Generating config"
if [ $pretrained = True ]; then
    python -u ${REPOSITORY_PATH}/mmdetection/tools/build_config.py\
        $model_name_with_augs\
        $patch_dataset_src\
        $patch_size\
        $src\
        --batch-size $batch_size\
        --augmentation $augs\
        --architecture $architecture\
        --pretrained
fi
if [ $pretrained = False ]; then
    python -u ${REPOSITORY_PATH}/mmdetection/tools/build_config.py\
        $model_name_with_augs\
        $patch_dataset_src\
        $patch_size\
        $src\
        --batch-size $batch_size\
        --augmentation $augs\
        --architecture $architecture
fi

############# TRAIN MODEL #############
echo "Training model"
config="${src}/${model_name_with_augs}_config.py"

python -u ${REPOSITORY_PATH}/mmdetection/tools/train.py\
    $config\
    --work-dir $src\
    --cfg-options randomness.seed=42

############# TEST MODEL - PATCH FORM #############
echo "Testing model - patch form"
best_full=$(find $src -name "best_coco_bbox_mAP_epoch*.pth" | sort -n | tail -n 1)

python -u ${REPOSITORY_PATH}/mmdetection/tools/test.py\
    $config\
    $best_full\
    --out $src/test/results.pkl

############ VISUALISE PATCH RESULTS #############
echo "Visualising result - patch form"
python -u ${REPOSITORY_PATH}/mmdetection/tools/analysis_tools/analyze_results.py\
    $config\
    $src/test/results.pkl\
    $src/test/visualizations/\
    --show-score-thr $conf_threshold

############# CONFUSION MATRIX - PATCH FORM #############
echo "Confusion matrix - patch form"
python -u ${REPOSITORY_PATH}/mmdetection/tools/confusion_matrix_abundance_ordered.py\
    $config\
    $src/test/results.pkl\
    $src/test/\
    --score-thr $conf_threshold\
    --abundance-ordering $abundance

############# CONFUSION MATRIX AND EVAL METRICS - WHOLE IMAGE FORM WITH SAHI #############
echo "Confusion matrix - whole image form with SAHI"
whole_images_test_json_src="${whole_annotations_dir}/dataset_test.json"
whole_image_test_output_folder_name='test'

sahi_nmm_folder_name="${whole_image_test_output_folder_name}_sahi"

python -u ${REPOSITORY_PATH}/mmdetection/tools/sahi_eval.py\
    $config\
    $best_full\
    $whole_images_src\
    $whole_images_test_json_src\
    $patch_size\
    $overlap\
    $sahi_nmm_folder_name\
    --bbox-conf-threshold $conf_threshold\
    --abundance-ordering $abundance\
    --postprocess-type $postprocess_type\
    --match-threshold $postprocess_match_threshold

############# CONFUSION MATRIX AND EVAL METRICS - WHOLE IMAGE FORM WITHOUT SAHI #############
echo "Confusion matrix - whole image form without SAHI"
non_sahi_folder_name="${whole_image_test_output_folder_name}_non_sahi"

python -u ${REPOSITORY_PATH}/mmdetection/tools/non_sahi_eval.py\
    $config\
    $best_full\
    $whole_images_src\
    $whole_images_test_json_src\
    $patch_size\
    $overlap\
    $non_sahi_folder_name\
    --bbox-conf-threshold $conf_threshold\
    --abundance-ordering $abundance

echo "Finished"