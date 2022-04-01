# python pretrain.py exp.name=<experiment-name> dataset=<dataset> algorithm=<algorithm>

DATA=pamap2
AUG=1.0
MIXUP=0
ALGO=estyle
MODEL=$ALGO_mixup$MIXUP_aug$AUG
python pretrain.py exp.name=$DATA-$MODEL dataset=$DATA algorithm=$ALGO spatialaug=$AUG mixup=$MIXUP
