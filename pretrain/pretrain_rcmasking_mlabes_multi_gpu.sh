#!/bin/bash
export PATH=/root/anaconda3/bin:$PATH
source activate mol3d
date=1223_3
cd /share/project/tanghan/sharefs-hantang/tanghan/models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 1233 /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/pretrain_rcmasking_mlabes_multi_gpu.py --lr 0.001 --batch_size 64 --epochs 5 --output_model_file stage2_mlabe_no_stop_gradient --run_name without_stop_gradient-mlabes_predict-rc_masking-multi_gpu-save > /share/project/tanghan/sharefs-hantang/tanghan/running_log/chemical_reaction_pretraining_${date}.log
# cd /share/project/tanghan/sharefs-hantang/tanghan/models
# python /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/pretrain_rcmasking_mlabes_multi_gpu.py --lr 0.001 --batch_size 64 --epochs 5 --output_model_file stage2_mlabe_no_stop_gradient --run_name without_stop_gradient-mlabes_predict-rc_masking_save > /share/project/tanghan/sharefs-hantang/tanghan/running_log/chemical_reaction_pretraining_${date}.log