python /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/pretrain_identification.py +experiment=pretrain_identification \
training_settings.epochs_RC=100 \
training_settings.testing_stage=0 \
training_settings.focal_loss=1 \
model.output_model_file=100_rc_identify_focal_loss \
wandb.run_name=100_rc_identify_focal_loss \
training_settings.device=1