python /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/pretrain_rcmasking.py +experiment=pretrain_NodeEdge \
training_settings.stage_one_model=/share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/10_20_30_40__NodeEdge_rc.pth \
model.output_model_file=10_continue_ \
training_settings.epochs_RC=100 \
training_settings.epochs_1hop=0 \
training_settings.epochs_2hop=0 \
training_settings.epochs_3hop=0 \
wandb.run_name=10_continue_NodeEdge \
training_settings.device=1 