from easydict import EasyDict as edict


DDI_CONFIG = edict()
DDI_CONFIG.data_dir = '/home/user/molecular/deepddi/data/DrugBank5.0_Approved_drugs'
DDI_CONFIG.label_file = '/home/user/molecular/deepddi/data/DrugBank_known_ddi.txt'
DDI_CONFIG.train_ratio = 0.6