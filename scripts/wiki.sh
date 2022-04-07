DATA=wikitext103
AUG=$AUG
MIXUP=1
ALGO=estyle
MODEL=$ALGO-mixup$MIXUP-aug$AUG
GPU=$GPU
CUDA_VISIBLE_DEVICES=$GPU python pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP



SAVE_DIR=/mnt/input/projects/dabs/models/
cp $SAVE_DIR/$DATA-$MODEL/'epoch=10-step=99999.ckpt' $SAVE_DIR/$DATA-$MODEL/epoch_10-step_99999.ckpt
CKPT=$SAVE_DIR/$DATA-$MODEL/epoch_10-step_99999.ckpt
echo Transfer $TRANSFER
echo ckpt $CKPT

DATA2=qnli
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=qqp
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=sst2
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=stsb
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=rte
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=wnli
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=mrpc
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT
