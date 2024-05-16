## Code Contribution
- Efran:
  - Most of Efran's work is related to getting the original 3DInAction repo running (this codebase). 
    - Efran attempted to get running the various modules in the original repos. The components with bugs that couldn't be resolved were moved after various attempts to get it up in running. You will notice in the code various functions and imports commented out or slightly modified.
      - Unused models moved to `models\archive`.
      - The scripts used in the `evaluation`, `figures`, `models`, and `util_scripts` had their imports and some of their scripts modified to avoid errors.
  - `run.py` was created by Efran to manage model runs, testing, and eval instead of using run_experiment.sh
  - `train.py`, `test.py`, and `eval.py` was modified from the original files in 3DInAction to work with MSR data
    - All 3 files now had logging capabilities. They also have an accompyning notebook (`run.ipynb`, `test.ipynb`, and `evaluation.ipynb`) which were used for dev and test of code changes
    - `train.py` now has special conditions for MSR data. `test.py` and `evaluation.py` now have a `run_ms`r function to handle msr data. `evaluation.py` now saves graphs of loss for train & test and stores in log\RUN_NAME\results folder. `test.py` catalogs results through training (without using wandb module) and stores in log\RUN_NAME folder
  - Efran created several helper scripts for training & testing. `make_holdout.ipynb` was created to make holdout set for MSR data. `yaml_builder.ipynb` was created to multiple config yamls for grid search. `analyze_grid_search_results.ipynb` was created analyze gridsearch results. `run_grid.py` was created to perform train, test, eval on each gridsearch yaml file to find best hyperparemeters
- Mathew:
  - Most of Mathew's work can be found in the `helper_&_other_scripts` folder. Here he created various scripts to import data, test features, and try to get various parts of the original 3DInAction running. 
    - Some of these notebooks are a part of a bigger file which can be viewed on Google Drive (shared with professor):https://drive.google.com/drive/folders/1Fk7pWL930S8xkWjXMdL5zD6kP5Ci8GWr?usp=drive_link
    - The script `helper_&_other_scripts\Pipeline of 3DInAction Model.ipynb` is where Mathew spent time trying to implement the 3DInAction architecture from scratch. Was not used in final models but a lot of time & effort was put into this to get them to run.
    - The scripts in `helper_&_other_scripts\Visualizing_Point_Clouds` where used to reproduce video of our training data. Was not used in the final project but the intention was to predict frames and compare original video with predicted
    - The script `helper_&_other_scripts\SetTransformers.ipynb` was created by Mathew to modify efran's 3DInAction code to help get MSR data running.
  - After we pivoted from IKEA + 3DInAction to Set Transformer & MSR, Mathew had already created a script `helper_&_other_scripts\MSR-Action3D Parse Depth Map Sequences.ipynb` that packed the MSR dataset into a format for us to test & train on.
  - Mathew added code to get the train.py by function as well as the Set Transformer model working with the MSR dataset. He found a large bug that was caused problems with our training process and fixed it so that attention from the transformers was applied correctly
- Harrison:
  - Harrison helped update and edit the presentation & slides.
  
## Instructions for Setup

### 1. Requirements
The code was tested with python 3.8.16 torch 1.10.1 and CUDA 11.3. 

- Download Cuda 11.3 first:  [Cuda 11.3 Download]([https://www.anu.edu.au/](https://developer.nvidia.com/cuda-11.3.0-download-archive)) 

```
sh
conda create -n tpatches_env python=3.8.16
conda activate tpatches_env
conda install pip #for using pip commands in the conda environments
# Install with instructions from https://pytorch.org/get-started/locally/
# Below is instructions for installation of long term support
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
#Then install all other requirements
pip install faiss-gpu
pip install -r requirements.txt (note, if any of the packages gives you an error, I suggest removing the '==version_number' from it)
```


### 2. Datasets

We evaluate on MSR datasets:
1. [MSR-Action3D FPS](https://drive.google.com/file/d/1ffSQyjbaX32vRs26M9Hhw0nE2HMrUTSV/view?usp=share_link) (200MB) (used for this project)
Optional:

Download the datasets, extract the `.zip` file and update the `dataset_path` in the `.config` file under `DATA`.

### 3. Train, test and evaluate

To train, test and evaluate with the default settings run

```sh run_experiment.sh```

For a customized model, edit the `.config` file.
Examples for different configurations are available in the `configs` directory.


