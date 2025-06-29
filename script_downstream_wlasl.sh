export TF_ENABLE_ONEDNN_OPTS=0

CUDA_VISIBLE_DEVICES=0 python downstream_classification.py \
  --lr 0.01 \
  --batch-size 32 \
  --pretrained 'save_ckptsave_ckpt_pretrain_implanted\checkpoint_0150.pth.tar' \
  --finetune-dataset SLR \
  --protocol cross_subject \
  --data-ratio 1.0 \
  --finetune-skeleton-representation graph-based \
  --checkpoint-path save_ckpt_downstream_implanted \
  --optim SGD \
  --subset_name WLASL \
  --num_class 100 \
  --input_size 64 \
  --eval_step 1 \
  --view all \
  --save-ckpt