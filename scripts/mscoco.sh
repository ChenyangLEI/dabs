if [ $AUG is '']
then
    AUG=1
else
    AUG=$AUG
fi

if [ $MIXUP is '']
then
    MIXUP=0
else
    MIXUP=$MIXUP
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

DATA=mscoco
ALGO=estyle
MODEL=$ALGO-mixup$MIXUP-aug$AUG-ratio$RATIO-log

echo 'MODEL:'  $MODEL

mkdir $PWD/DATASETS
mkdir $PWD/DATASETS/captioned_images
mkdir $PWD/DATASETS/captioned_images/mscoco
cp /mnt/input/data/coco_raw/*zip $PWD/DATASETS/captioned_images/mscoco
cd $PWD/DATASETS/captioned_images/mscoco

unzip test2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
cd /home/aiscuser/dabs



CUDA_VISIBLE_DEVICES=$GPU python pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP ratio=$RATIO


SAVE_DIR=/mnt/input/projects/dabs/models/
cp $SAVE_DIR/$DATA-$MODEL/'epoch=7-step=99999.ckpt' $SAVE_DIR/$DATA-$MODEL/epoch_7-step_99999.ckpt
CKPT=$SAVE_DIR/$DATA-$MODEL/epoch_7-step_99999.ckpt
echo Transfer $TRANSFER
echo ckpt $CKPT

DATA2=vqa
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT
