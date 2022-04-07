# python pretrain.py exp.name=<experiment-name> dataset=<dataset> algorithm=<algorithm>
AUG=$AUG
GPU=$GPU
DATA=imagenet
<<<<<<< HEAD:scripts/imagenet_mixup1_aug0.7.sh
# DATA=mc4
AUG=$AUG

GPU=$GPU
MIXUP=1
ALGO=estyle
MODEL=$ALGO-mixup$MIXUP-aug$AUG
=======
MIXUP=1
ALGO=estyle
MODEL=$ALGO-mixup$MIXUP-aug$AUG

>>>>>>> d0bbf0e2c91885f6eb5d1ea7717a0c23fff05af4:scripts/imagenet.sh
CUDA_VISIBLE_DEVICES=$GPU python pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP



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

