# python pretrain.py exp.name=<experiment-name> dataset=<dataset> algorithm=<algorithm>
DATA=imagenet

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
MODEL=$ALGO-mixup$MIXUP-aug$AUG-ratio$RATIO-div

CUDA_VISIBLE_DEVICES=$GPU python pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP ratio=$RATIO



SAVE_DIR=/mnt/input/projects/dabs/models/
cp $SAVE_DIR/$DATA-$MODEL/'epoch=4-step=99999.ckpt' $SAVE_DIR/$DATA-$MODEL/epoch_4-step_99999.ckpt
CKPT=$SAVE_DIR/$DATA-$MODEL/epoch_4-step_99999.ckpt
echo Transfer $TRANSFER
echo ckpt $CKPT

DATA2=traffic_sign
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=dtd
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=vgg_flower
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=cifar10
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT


DATA2=aircraft
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

cp -r /mnt/input/data/cu_birds ./DATASETS/natural_images/
DATA2=cu_birds
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT
