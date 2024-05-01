python efran_test.py
GPU_IDX=0
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX

IDENTIFIER='tpatches_debug'
CONFIG='configs\ikeaasm\config_ikeaasm.yaml'
LOGDIR='./log/'

echo 'hello'

python train.py --identifier $IDENTIFIER --config $CONFIG --logdir $LOGDIR --fix_random_seed