DATA=mc4

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
MODEL=$ALGO-mixup$MIXUP-aug$AUG-ratio$RATIO
echo 'MODEL:'  $MODEL


CUDA_VISIBLE_DEVICES=$GPU python pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP ratio=$RATIO



SAVE_DIR=/mnt/input/projects/dabs/models/
cp $SAVE_DIR/$DATA-$MODEL/'epoch=0-step=99999.ckpt' $SAVE_DIR/$DATA-$MODEL/epoch_0-step_99999.ckpt
CKPT=$SAVE_DIR/$DATA-$MODEL/epoch_0-step_99999.ckpt
echo Transfer $TRANSFER
echo ckpt $CKPT

DATA2=paws_de
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=paws_en
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=paws_es
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=paws_fr
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=paws_ja
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=paws_ko
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=paws_zh
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT
