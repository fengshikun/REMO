python /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/pretrain_rcmasking_480k.py +experiment=pretrain_NodeEdge \
training_settings.stage_one_model=/share/project/tanghan/sharefs-hantang/chem/model_gin/masking.pth \
model.output_model_file=100_NodeEdge_480k \
training_settings.epochs_RC=100 \
training_settings.epochs_1hop=0 \
training_settings.epochs_2hop=0 \
training_settings.epochs_3hop=0 \
wandb.run_name=100_NodeEdge_480k \
training_settings.testing_stage=0 \
training_settings.device=0