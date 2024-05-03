#!/usr/bin/env bash

GPU_IDX=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX

IDENTIFIER='set_transformer_dbug'
CONFIG='configs\ikeaasm\config_ikeaasm.yaml'
LOGDIR='./log/'

echo '--------------- starting training'
#python train.py --identifier $IDENTIFIER --config $CONFIG --logdir $LOGDIR --fix_random_seed

echo '--------------- starting testing'
#python test.py --identifier $IDENTIFIER --model_ckpt '000001.pt' --logdir $LOGDIR --fix_random_seed
echo '--------------- starting eval'
python ./evaluate.py --identifier $IDENTIFIER --logdir $LOGDIR