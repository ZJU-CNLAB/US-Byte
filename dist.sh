#!/bin/bash
dnn="${dnn:-vgg16}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
threshold="${threshold:-0}"
nwpernode=4
nstepsupdate=1
PY=horovodrun
GRADSPATH=./logs

$PY -np $nworkers python dist_trainer_US-Byte.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --saved-dir $GRADSPATH

