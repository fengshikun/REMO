# Chemical_Reaction_Pretraining

A self-supervised learning framework to exploit information from chemical reaction dataset




# Pretraining(Pretrained by both Masked reaction Reconstruction and Reaction Centre Idntification)

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port 10086 pretrain_indentification_mask_graphormer.py --yaml_file ../conf/indentifi_mlabel.yaml

```


We also provided the pretrained model in this link(https://drive.google.com/file/d/1t62Xo5Akco9Z04El_wtbcNR_9HN0wnnf/view?usp=sharing)




## Fintuning

### finetuning on the MoleculeACE
sh ./finetune/finetune_ace.sh



### finetuning on the moleculenet

python test/finetune_gfmodel_scripts.py --input_model_files pretrained_model --graphormer_config_yaml ./test/assets/graphormer_small.yaml

### finetuning on the drugddi

sh finetune/finetune_graphormer_ddi.sh


### finetuning on the ACNet

The Original ACNet utilizes offline features, which serve as inputs to a Multilayer Perceptron (MLP) for evaluation. Please refer to the code available at https://github.com/DrugAI/ACNet for details on how the evaluation process is implemented.