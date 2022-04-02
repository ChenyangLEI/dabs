# python pretrain.py exp.name=<experiment-name> dataset=<dataset> algorithm=<algorithm>

DATA=pamap2
AUG=0
MIXUP=1
GPU=0
ALGO=estyle
MODEL=$ALGO-mixup$MIXUP-aug$AUG
CUDA_VISIBLE_DEVICES=$GPU python pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP



DATA2=pamap2
TRANSFER=$DATA-to-$DATA2-$MODEL
SAVE_DIR=/mnt/input/projects/dabs/models/
cp $SAVE_DIR/$DATA-$MODEL/'epoch=512-step=99999.ckpt' $SAVE_DIR/$DATA-$MODEL/epoch_512-step_99999.ckpt
CKPT=$SAVE_DIR/$DATA-$MODEL/epoch_512-step_99999.ckpt
echo Transfer $TRANSFER
echo ckpt $CKPT

CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT
