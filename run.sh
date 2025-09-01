#!/bin/bash

CODE_ROOT=/root/autodl-tmp/WSI_Prompt
DATA_NAME=ZJ
TASK=IDH_label
COORD_ROOT=/root/autodl-tmp/HE_SN_ALL/SN_${DATA_NAME}/processed_slide/
MODEL_NAME=ResNet50
MODEL_NAME_MIL=CLAM_SB
EXP_CODE=${MODEL_NAME_MIL}_${MODEL_NAME}_${DATA_NAME}_base
SIZE=224

cd $CODE_ROOT
echo "Current Working Directory: $(pwd)"

################### MIL training ###################
###CLAM### --testing \
python3 MIL_train.py \
--ft_model $MODEL_NAME \
--mil_method $MODEL_NAME_MIL \
--n_classes 2 \
--drop_out \
--lr 2e-4 \
--B 8 \
--accumulate_grad_batches 1 \
--label_frac 1.0 \
--task $DATA_NAME \
--test_label $TASK \
--exp_code $EXP_CODE \
--max_epochs 50 \
--bag_loss ce \
--inst_loss svm \
--log_data \
--data_root_dir $COORD_ROOT  \
--test_data_root_dir $COORD_ROOT  \

