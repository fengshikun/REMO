python /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/pretrain_identification.py +experiment=pretrain_identification \
training_settings.stage_one_model=/share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/100_rc_identify_2_mlabes_100_reaction_centre_identification.pth \
training_settings.epochs_start=101 \
training_settings.testing_stage=0 \
model.output_model_file=100_rc_identify_2 \
wandb.run_name=100_rc_identify \
training_settings.device=1