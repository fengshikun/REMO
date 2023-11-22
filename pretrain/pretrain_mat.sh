export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/home/fengshikun/miniconda3/envs/MOLuni/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node 4 --master_port 10086  pretrain_identification_masking_mat.py --yaml_file ../conf/identifi_mlabel_mat.yaml > pretrain_identification_masking_mat.log 2>&1 &
