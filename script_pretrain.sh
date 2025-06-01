#!/bin/bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch --nproc_per_node=1 --master_port 11903 pretrain.py \
--workers 2 \
--lr 0.01 \
--batch-size 32 \
--teacher-t 0.05 \
--student-t 0.1 \
--topk 8192 \
--mlp \
--contrast-t 0.07 \
--contrast-k 16384 \
--checkpoint-path save_ckpt_pretrain \
--schedule 100 \
--epochs 150· \
--pre-dataset SLR \
--skeleton-representation graph-based \
--inter-dist