DATA=pamap2

if [ $AUG is '']
then
    AUG=1
else
    AUG=$AUG
fi

if [ $GPU is '']
then
    GPU=0
else
    GPU=$GPU
fi

if [ $RATIO is '']
then
    RATIO=1
else
    RATIO=$RATIO
fi

if [ $MIXUP is '']
then
    MIXUP=0
else
    MIXUP=$MIXUP
fi

ALGO=estyle
MODEL=$ALGO-mixup$MIXUP-aug$AUG-ratio$RATIO-log
CUDA_VISIBLE_DEVICES=$GPU python pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP ratio=$RATIO



DATA2=pamap2
TRANSFER=$DATA-to-$DATA2-$MODEL
SAVE_DIR=/mnt/input/projects/dabs/models/
cp $SAVE_DIR/$DATA-$MODEL/'epoch=512-step=99999.ckpt' $SAVE_DIR/$DATA-$MODEL/epoch_512-step_99999.ckpt
CKPT=$SAVE_DIR/$DATA-$MODEL/epoch_512-step_99999.ckpt
echo Transfer $TRANSFER
echo ckpt $CKPT

CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT
