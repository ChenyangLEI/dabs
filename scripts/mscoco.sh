DATA=mscoco
AUG=$AUG
GPU=$GPU
MIXUP=1
ALGO=estyle
MODEL=$ALGO-mixup$MIXUP-aug$AUG
CUDA_VISIBLE_DEVICES=$GPU python pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP



SAVE_DIR=/mnt/input/projects/dabs/models/
cp $SAVE_DIR/$DATA-$MODEL/'epoch=0-step=99999.ckpt' $SAVE_DIR/$DATA-$MODEL/epoch_0-step_99999.ckpt
CKPT=$SAVE_DIR/$DATA-$MODEL/epoch_0-step_99999.ckpt
echo Transfer $TRANSFER
echo ckpt $CKPT

DATA2=paws_de
TRANSFER=$DATA-to-$DATA2-$MODEL
#CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

# python pretrain.py exp.name=<experiment-name> dataset=<dataset> algorithm=<algorithm>
AUG=$AUG
GPU=$GPU
DATA=mscoco
MIXUP=1
ALGO=estyle
MODEL=$ALGO-mixup$MIXUP-aug$AUG

CUDA_VISIBLE_DEVICES=$GPU python pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP

mkdir /home/aiscuser/dabs/DATASETS/captioned_images/mscoco
cp mnt/input/data/coco_raw/*zip /home/aiscuser/dabs/DATASETS/captioned_images/mscoco
cd /home/aiscuser/dabs/DATASETS/captioned_images/mscoco

unzip test2017.zip
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
cd ~/dabs

SAVE_DIR=/mnt/input/projects/dabs/models/
cp $SAVE_DIR/$DATA-$MODEL/'epoch=10-step=99999.ckpt' $SAVE_DIR/$DATA-$MODEL/epoch_10-step_99999.ckpt
CKPT=$SAVE_DIR/$DATA-$MODEL/epoch_10-step_99999.ckpt
echo Transfer $TRANSFER
echo ckpt $CKPT

DATA2=vqa
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT
>>>>>>> d0bbf0e2c91885f6eb5d1ea7717a0c23fff05af4
