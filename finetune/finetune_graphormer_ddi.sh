#!/bin/bash
export PATH=/root/anaconda3/bin:$PATH
# source activate mol3d
#export model_file=/share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/20_graphormer_mlabes_480kgraphormer_masking_mlabes_rc.pth
export model_type=graphormer
export mask_type=mlabes_RC
export device=0
export seed_list=(0 1 2);
for seed in "${seed_list[@]}"; do
    export log_name=finetune_"$model_type"_"$mask_type"_DrugDrugInteraction_seed"$seed"
    python finetune_graphormer_ddi.py \
    training_settings.device="$device" \
    training_settings.lr=1e-3 \
    training_settings.decay=1e-6 \
    training_settings.epochs=300 \
    training_settings.eval_train=1 \
    training_settings.runseed="$seed" \
    > /home/linbicheng/molecular/Chemical_Reaction_Pretraining/finetune/logs/"$log_name".log;
    # ++model.input_model_file="$model_file" \
done;