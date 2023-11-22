# python pretrain/pretrain_indentification_mask_graphormer.py --config-name identifi_mlabel.yaml run_name=gf_mlabel_identi training_settings.mlabel=false training_settings.identi=true

python /share/project/tanghan/sharefs-hantang/tanghan/Chemical_Reaction_Pretraining/pretrain/pretrain_graphormer.py +experiment=pretrain_mlabels_graphormer \
training_settings.device=1 \
training_settings.epochs_RC=20 \
model.target=NodeEdge \
training_settings.testing_stage=0 \
model.output_model_file=20_graphormer_NodeEdge_480k \
wandb.run_name=20_graphormer_NodeEdge_480k