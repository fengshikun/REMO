#!/bin/bash
#rlaunch --gpu=1 --cpu=16 --memory=160000 -- 
export PATH=/root/anaconda3/bin:$PATH
source activate mol3d
date=mlabe_0103_test
cd /share/project/tanghan/sharefs-hantang/tanghan/models
python /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/pretrain_rcmasking.py +experiment=pretrain_scratch \
model.output_model_file=model/stage2_no_stop_gradient_test \
wandb.run_name=TEST_without_stop_gradient-mlabes_predict-rc_masking_save