
# Automated Detection of Antarctic Benthic Organisms in High-Resolution In Situ Imagery to Aid Biodiversity Monitoring


<div align="center">
Integrating MMDetection and SAHI to detect Antarctic Benthic Organisms
  <img src="./assets/framework_overview.png" width="600" alt="Framework Overview (Fig 1 of Trotter et. al, 2025)" />
</div>


## 📑 Overview
This repository contains the code for _Automated Detection of Antarctic Benthic Organisms in High-Resolution In Situ Imagery to Aid Biodiversity Monitoring_ (Trotter et. al, 2025) [TODO: Add link to paper when available].

The code is based on the [MMDetection](https://github.com/open-mmlab/mmdetection) framework and uses the [Slicing Aided Hyper Inference (SAHI)](https://github.com/obss/sahi) library. Whilst MMDetection has some integration with SAHI out of the box, this repo provides further integration, namely **the ability to evaluate an MMDetection model with and without SAHI post-processing**. 

We also provide **a full training and evaluation pipeline for MMDetection models with SAHI post-processing**, for use with SLURM clusters.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone [TODO: Add link to repo]
   cd [TODO: Add repo name]
   ```

2. Run the setup script:
   ```bash
   ./setup.sh
   ```
This will:
- Install the required dependencies through a mamba environment.
- Clone and install the MMDetection repository.
- Merge the required additional files created for this project into MMDetection (see `./mmdetection_additions`).
- Download the Weddell Sea Bethic Dataset.
- Download the optimal model weights for the Weddell Sea Benthic Dataset. [TODO: Add in when available on PDC]
- Download the required model weights from the MMDetection model zoo.

An output log, `setup_output.txt`, will be created in the root directory of the repository. This log contains the output of the setup script, including any errors that may have occurred during installation.

## 📊 Usage

The below instructions assume you are using a SLURM cluster. If you are not using SLURM, you can run the scripts called within the `.sh` files directly in your terminal.

## ⚡ Inference

To run inference using the optimal model trained on the Weddell Sea Benthic Dataset:

1. Edit the `mmdetection/tools/slurm_best_inference.sh` file to set the correct sbatch instructions and paths for your environment (lines 7-24).

2. Submit the job to SLURM:
```bash
sbatch ./mmdetection/tools/slurm_best_inference.sh
```

## 🏋️ Training

To run a full training and evaluation pipeline using the Weddell Sea Benthic Dataset:

1. Edit the `mmdetection/tools/slurm_pipeline.sh` file to set the correct sbatch instructions and paths for your environment (lines 8-23).

2. Set the parameters you wish use for training in the `mmdetection/tools/slurm_pipeline.sh` file (lines 34-48).

3. Edit `mmdetection/tools/build_config.py` to set the path for the project root (line 15).

3. Submit the job to SLURM:
```bash
sbatch ./mmdetection/tools/slurm_pipeline.sh
```
### 🐙 Training on your own dataset

To make use of your own image dataset, ensure this is provided in COCO format, split into train, validation, and test sets. Follow the formatting of the downloaded Weddell Sea Benthic Dataset as a guide. You can then use the `mmdetection/tools/slurm_pipeline.sh` script to generate a patched dataset and model config file required for training and evaluation. The resulting model weights can be used for inference with `mmdetection/tools/slurm_best_inference.sh`.

## 📂 Data Access

Both the Weddell Sea Benthic Dataset and the optimal model weights are available for download independant of the `setup.sh` script via the NERC Polar Data Centre.

The Weddell Sea Benthic Dataset:

Trotter, C., Griffiths, H.J., Khan, T.M., Purser, A., & Whittle, R.J. (2025). The Weddell Sea Benthic Dataset: A computer vision-ready object detection dataset for in situ benthic biodiversity monitoring model development (Version 1.0) [Data set]. NERC EDS UK Polar Data Centre. https://doi.org/10.5285/1ba97e4b-efb7-460b-9f2d-90437e33ce09 

The optimal model weights:

Trotter, C., Griffiths, H. J., Khan, T. M., & Whittle, R. J. (2025). Automated detection of Antarctic benthic organisms in high-resolution in situ imagery to aid biodiversity monitoring: optimal model weights (Version 1.0) [Data set]. NERC EDS UK Polar Data Centre. https://doi.org/10.5285/B2874F3F-285D-4AE6-9BB4-6BFE3EACBFFF


## 📝 Citations

If you use this code, the Weddell Sea Benthic Dataset, or the best model weights in your own research, please cite the following paper:

```bibtex
[TODO: Add bibtex citation when available]
``` 
