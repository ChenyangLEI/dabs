AUG=$AUG
GPU=$GPU
DATA=mscoco
MIXUP=1
ALGO=estyle
MODEL=$ALGO-mixup$MIXUP-aug$AUG

mkdir /home/aiscuser/dabs/DATASETS/captioned_images/mscoco
cp /mnt/input/data/coco_raw/*zip /home/aiscuser/dabs/DATASETS/captioned_images/mscoco
cd /home/aiscuser/dabs/DATASETS/captioned_images/mscoco

unzip test2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
cd ~/dabs

CUDA_VISIBLE_DEVICES=$GPU python pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP


SAVE_DIR=/mnt/input/projects/dabs/models/
cp $SAVE_DIR/$DATA-$MODEL/'epoch=7-step=99999.ckpt' $SAVE_DIR/$DATA-$MODEL/epoch_7-step_99999.ckpt
CKPT=$SAVE_DIR/$DATA-$MODEL/epoch_7-step_99999.ckpt
echo Transfer $TRANSFER
echo ckpt $CKPT

DATA2=vqa
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT
