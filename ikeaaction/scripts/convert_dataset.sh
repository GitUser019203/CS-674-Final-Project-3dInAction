#!/usr/bin/env bash

DATASET_PATH='/data1/datasets/ANU_ikea_dataset_smaller/'
OUTPUT_PATH='/data1/datasets/ANU_ikea_dataset_smaller_clips/'
FPC=32

python convert_dataset_to_clips.py --dataset_path $DATASET_PATH --output_dataset_dir $OUTPUT_PATH --frames_per_clip $FPC
