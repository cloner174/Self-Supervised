# Set the number of threads for OpenMP. Usually 1 for DNN training to avoid conflicts.
export OMP_NUM_THREADS=1

# Specify which GPUs to make visible to the script.
# Use a single GPU (GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Number of processes per node
# Example: Using 2 GPUs -->> NPROC_PER_NODE=2
#Using 1  here:
NPROC_PER_NODE=1

# Master port for distributed communication
MASTER_PORT=11903

# Path to save checkpoints
CHECKPOINT_PATH="./save_ckpt_pretrain_dist"

# Arguments for pretrain.py
LEARNING_RATE=0.01
BATCH_SIZE=4
TEACHER_T=0.05
STUDENT_T=0.1
TOPK=8192
CONTRAST_T=0.07
CONTRAST_K=16384
SCHEDULE_EPOCHS=100
TOTAL_EPOCHS=150
PRETRAIN_DATASET="SLR"
SKELETON_REPRESENTATION="graph-based"

mkdir -p ${CHECKPOINT_PATH}

echo "Starting pretraining with torch.distributed.launch..."
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Processes per node: ${NPROC_PER_NODE}"
echo "Checkpoint path: ${CHECKPOINT_PATH}"

# The --use_env flag is recommended with torch.distributed.launch if your script
# is designed to primarily pick up RANK, LOCAL_RANK, WORLD_SIZE from env variables.
# However, your pretrain.py defines --local_rank as an argument, which launch.py also passes.
# misc.init_distributed_mode also reads from environment variables.
python -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE} --master_port=${MASTER_PORT} pretrain.py \
    --lr ${LEARNING_RATE} \
    --batch-size ${BATCH_SIZE} \
    --teacher-t ${TEACHER_T} \
    --student-t ${STUDENT_T} \
    --topk ${TOPK} \
    --mlp \
    --contrast-t ${CONTRAST_T} \
    --contrast-k ${CONTRAST_K} \
    --checkpoint-path ${CHECKPOINT_PATH} \
    --schedule ${SCHEDULE_EPOCHS} \
    --epochs ${TOTAL_EPOCHS} \
    --pre-dataset ${PRETRAIN_DATASET} \
    --skeleton-representation ${SKELETON_REPRESENTATION} \
    --inter-dist
    # Add any other arguments for pretrain.py as needed

echo "Pretraining script finished."