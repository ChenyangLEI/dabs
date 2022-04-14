DATA=librispeech
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
MODEL=$ALGO-mixup$MIXUP-aug$AUG
CUDA_VISIBLE_DEVICES=$GPU python -W ignore pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP ratio=$RATIO
echo MODEL $MODEL

SAVE_DIR=/mnt/input/projects/dabs/models/
cp $SAVE_DIR/$DATA-$MODEL/'epoch=22-step=99999.ckpt' $SAVE_DIR/$DATA-$MODEL/epoch_22-step_99999.ckpt
CKPT=$SAVE_DIR/$DATA-$MODEL/epoch_22-step_99999.ckpt
echo Transfer $TRANSFER
echo ckpt $CKPT

git clone 'https://github.com/soerenab/AudioMNIST'
mkdir /home/aiscuser/dabs/DATASETS/speech
mv AudioMNIST /home/aiscuser/dabs/DATASETS/speech/AudioMNIST-master
DATA2=audio_mnist
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python -W ignore transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=fluent_speech_action
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python -W ignore transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=fluent_speech_location
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python -W ignore transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=fluent_speech_object
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python -W ignore transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT

DATA2=librispeech
TRANSFER=$DATA-to-$DATA2-$MODEL
CUDA_VISIBLE_DEVICES=$GPU python -W ignore transfer.py exp.name=$TRANSFER dataset=$DATA2 ckpt=$CKPT
