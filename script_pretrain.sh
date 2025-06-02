export TF_ENABLE_ONEDNN_OPTS=0

python pretrain.py \
--workers 16 \
--lr 0.01 \
--batch-size 16 \
--teacher-t 0.05 \
--student-t 0.1 \
--topk 8192 \
--mlp \
--contrast-t 0.07 \
--contrast-k 16384 \
--checkpoint-path save_ckpt_pretrain \
--schedule 100 \
--epochs 150 \
--pre-dataset SLR \
--skeleton-representation graph-based \
--inter-dist \
--device cuda