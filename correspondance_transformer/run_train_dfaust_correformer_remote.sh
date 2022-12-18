GPU_IDX=1
NUM_THREADS=96
export OMP_NUM_THREADS=$NUM_THREADS
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES=$GPU_IDX
export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="1-96"

NPOINTS=1024
BATCH_SIZE=128
NHEADS=4
DIM=256
DFF=256
DATASET_PATH='/data1/datasets/dfaust/'


python3 train_dfaust_correformer.py --dataset_path $DATASET_PATH --dim $DIM --n_heads $NHEADS --batch_size $BATCH_SIZE --n_points $NPOINTS --d_feedforward $DFF