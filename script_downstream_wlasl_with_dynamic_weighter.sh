export TF_ENABLE_ONEDNN_OPTS=0

CUDA_VISIBLE_DEVICES=0 python downstream_classification.py \
  --lr 0.01 \
  --workers 16 \
  --batch-size 32 \
  --pretrained 'save_ckpt_pretrain\checkpoint_0150.pth.tar' \
  --finetune-dataset SLR \
  --protocol cross_subject \
  --data-ratio 1.0 \
  --finetune-skeleton-representation graph-based \
  --checkpoint-path save_ckpt_downstream \
  --optim SGD \
  --subset_name WLASL \
  --num_class 100 \
  --input_size 64 \
  --eval_step 1 \
  --view all \
  --save-ckpt \
  --epochs 28 \
  --use_dynamic_weighter