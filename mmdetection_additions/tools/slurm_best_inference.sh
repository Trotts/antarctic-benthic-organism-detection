#!/usr/bin/env bash

# Script file for running inference on a SLURM cluster.
# Usage: sbatch slurm_run_best_weights.sh
# Make sure to edit below before running.

#SBATCH --time=14-00:00:00
#SBATCH --partition= # partition name
#SBATCH --account= # account name
#SBATCH --gres= # GPU type and number
#SBATCH --mem-per-gpu= # Require XGB of memory per GPU

#SBATCH --output=job_log/run_best_inference.%j.%N.out
#SBATCH --error=job_log/run_best_inference.%j.%N.err 
#SBATCH --job-name=run_best_inference

# Path to the repository root directory
REPOSITORY_PATH='/PATH/TO/YOUR/REPOSITORY'
# Path to a visible cache for mmdetection, mmengine, etc.
CACHE_PATH='/PATH/TO/YOUR/CACHE'
# Miniforge3 installation path
MINIFORGE_PATH='/PATH/TO/YOUR/MINIFORGE3'
# Path to an input image for inference
INPUT_IMAGE_PATH='/PATH/TO/YOUR/INPUT/IMAGE'

# Path to the config file for the best model
config_path="${REPOSITORY_PATH}/mmdetection/configs/_base_/models/best_config.py"
weights_path="${REPOSITORY_PATH}/best_weights/wsbd_model_weights.pth"
output_dir="${REPOSITORY_PATH}/output" 

# Best params needed for inference
patch_size='(500,500)'
overlap=0.5
postprocess_type='NMM'
postprocess_match_threshold=0.2
conf_threshold=0.6

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

# Run inference with the best weights
echo "Running inference with the best weights"
python -u ${REPOSITORY_PATH}/mmdetection/tools/inference.py \
    ${config_path} \
    ${weights_path} \
    ${INPUT_IMAGE_PATH} \
    ${output_dir} \
    ${patch_size} \
    --bbox-conf-threshold ${conf_threshold} \
    --overlap ${overlap} \
    --post-process-match-threshold ${postprocess_match_threshold} \
    --post-process-algo ${postprocess_type} \
    --visualise \
    