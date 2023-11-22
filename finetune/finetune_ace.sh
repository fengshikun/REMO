#!/bin/bash
# export PATH=/root/anaconda3/bin:$PATH
# source activate mol3d
export model_file=Pretrained_model.path
export model_type=identifi_Scratch_cliff_100
export mask_type=None
export device=0
export seed_list=(0 1 2);
for seed in "${seed_list[@]}"; do
    export log_name=finetune_"$model_type"_"$mask_type"_ActivityCliff_seed"$seed"
    python finetune/finetune_ac.py \
    ++model.input_model_file="$model_file" \
    training_settings.device="$device" \
    training_settings.lr=1e-3 \
    training_settings.eval_train=1 \
    training_settings.runseed="$seed" > ./finetune/logs2/activity_cliff/chemical_reaction_"$log_name".log;

done;