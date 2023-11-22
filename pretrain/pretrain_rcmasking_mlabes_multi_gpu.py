import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import wandb
# sys.path.append(sys.path[0]+'/..')
import os
# os.chdir("/share/project/Chemical_Reaction_Pretraining/pretrain/")
sys.path.insert(0,'..')
from data_analysis.USPTO_CONFIG import USPTO_CONFIG

from tqdm import tqdm
import numpy as np

from models.gnn.models import GNN

from sklearn.metrics import roc_auc_score

import pandas as pd
from data_processing import ReactionDataset, CustomCollator
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


criterion = nn.CrossEntropyLoss()

import timeit
class ReactantStage2(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self, gnn):
        super(ReactantStage2, self).__init__()
        self.gnn = gnn
        
    def forward(self, batch_input, device):
        batch_graph = batch_input['graph'].to(device)
        print(batch_graph.x.shape, device)
        node_rep = self.gnn(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
        # print(node_rep.shape)
        # print(torch.isnan(node_rep).sum())
        tensor_2_head = torch.tensor([]).to(device)
        
        batch_size = len(batch_input['condition_idx'])
        
        for batch_num in range(batch_size):
            node_rep_single = torch.index_select(node_rep, 0, torch.LongTensor([i for i, x in enumerate(batch_graph.batch) if x == batch_num]).to(device))
            
            cond_rep = torch.index_select(node_rep_single, 0, torch.LongTensor(batch_input['condition_idx'][batch_num]).to(device))
            
            pri_rep = torch.index_select(node_rep_single, 0, torch.LongTensor(batch_input['primary_idx'][batch_num]).to(device))
            # print(torch.isnan(pri_rep).sum(), torch.isnan(cond_rep).sum(), torch.isnan(node_rep_single).sum())
            cond_pool = torch.mean(cond_rep, dim=0, keepdim=True)
            # if torch.isnan(cond_pool).sum() > 0:
            #     print(cond_pool, cond_rep, batch_input['condition_idx'][batch_num])
            # print(torch.isnan(cond_pool).sum())
            if len(batch_input['condition_idx'][batch_num]) > 0:
                new_rep = torch.cat([pri_rep, torch.broadcast_to(cond_pool, pri_rep.shape)], dim=1)
            # print(torch.isnan(new_rep).sum())
            else:
                new_rep = torch.cat([pri_rep, torch.broadcast_to(torch.zeros_like(cond_pool), pri_rep.shape)], dim=1)
            tensor_2_head = torch.cat([tensor_2_head, new_rep, torch.zeros(cond_rep.shape[0], new_rep.shape[1]).to(device)], dim=0)
            # print(torch.isnan(tensor_2_head).sum())
        # print(tensor_2_head.shape)
        
        # print(torch.isnan(tensor_2_head).sum())
        return batch_input['graph'], tensor_2_head, batch_input['mlabes']
    
def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def train(args, model_list, loader, optimizer_list, device, rank, world_size):
    model, linear_pred_atoms, linear_pred_bonds, linear_pred_mlabes = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_mlabes = optimizer_list

    model.train()
    linear_pred_atoms.train()
    linear_pred_bonds.train()
    linear_pred_mlabes.train()
    
    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0
    acc_mlabes_accum = 0
    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        device = model.device
        temp_graph, node_rep, mlabes = model(batch, device)
        
        ## loss for nodes
        # print("batch run through")
        pred_node = linear_pred_mlabes(node_rep[temp_graph.masked_atom_indices])
        tgt = torch.tensor([mlabe for i, mlabe in enumerate(batch["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
        # loss = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
        loss = criterion(pred_node.double(), tgt)
        
        acc_node = compute_accuracy(pred_node, tgt)
        acc_node_accum += acc_node

        optimizer_model.zero_grad()
        optimizer_linear_pred_mlabes.zero_grad()
        # optimizer_linear_pred_bonds.zero_grad()
        
        loss.backward()
        
        optimizer_model.step()
        optimizer_linear_pred_mlabes.step()
        loss_accum += float(loss.cpu().item())
        if step % 100 == 1:
            wandb.log({"loss_accum": loss_accum/step,
                       "accuracy_accum": acc_node_accum/step,
                       "loss": loss,
                       "accuracy": acc_node
                       })
        # if step > 300:
        #     break
    return loss_accum/step, acc_node_accum/step
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    # parser.add_argument('--device', type=int, default=0,
    #                    help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.15,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_edge', type=int, default=0,
                        help='whether to mask edges or not together with atoms')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--run_name', type=str, default = "training", help="running name in WanDB")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    world_size = torch.cuda.device_count()
    print('Start multi gpu training')
    rank = args.local_rank

    if world_size > 1:
        dist.init_process_group('nccl', rank=args.local_rank, world_size=world_size)


    # device_ids = [0, 1, 2, 3, 4, 5]
    # device_ids = [0, 1]
    wandb.login(key='11aa569e656b0bc19ad079b4bdaaa9ea3553694a')
    wandb.init(project="Chemical-Reaction-Pretrainin-Stage2", name=args.run_name)
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    model.load_state_dict(torch.load("/share/chem/model_gin/masking.pth"))
    
    reaction_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset, atom_vocab=USPTO_CONFIG.atom_vocab_file)
    
    collator_rc = CustomCollator(mask_stage='reaction_centre')
    

    collator_1hop = CustomCollator(mask_stage='one_hop')
    collator_2hop = CustomCollator(mask_stage='two_hop')
    collator_3hop = CustomCollator(mask_stage='three_hop')
    if world_size == 1:
        data_loader_rc = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn = collator_rc)
        data_loader_1hop = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn = collator_1hop)
        
        
        data_loader_2hop = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn = collator_2hop)
        
        
        data_loader_3hop = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn = collator_3hop)
    else:
        train_sampler = DistributedSampler(reaction_dataset, num_replicas=world_size,
                                        rank=rank)
        data_loader_rc = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size, num_workers = args.num_workers, collate_fn = collator_rc, sampler=train_sampler)
        data_loader_1hop = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size, num_workers = args.num_workers, collate_fn = collator_1hop, sampler=train_sampler)
        
        
        data_loader_2hop = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size,  num_workers = args.num_workers, collate_fn = collator_2hop, sampler=train_sampler)
        
        
        data_loader_3hop = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size, num_workers = args.num_workers, collate_fn = collator_3hop, sampler=train_sampler)
    
    model = ReactantStage2(model).to(rank)
    linear_pred_atoms = torch.nn.Linear(args.emb_dim * 2, 119).to(rank)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim * 2, 4).to(rank)
    linear_pred_mlabes = torch.nn.Linear(args.emb_dim * 2, 2458).to(rank)
    if torch.cuda.device_count() > 1:
        print("We have available ", torch.cuda.device_count(), "GPUs!")
        # model = nn.DataParallel(model, device_ids = device_ids)
        # linear_pred_atoms = nn.DataParallel(linear_pred_atoms, device_ids=device_ids)
        # linear_pred_bonds = nn.DataParallel(linear_pred_bonds, device_ids=device_ids)
        # linear_pred_mlabes = nn.DataParallel(linear_pred_mlabes,device_ids =device_ids)
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=False) 
        linear_pred_atoms = DistributedDataParallel(linear_pred_atoms, device_ids=[rank], find_unused_parameters=True) 
        linear_pred_bonds = DistributedDataParallel(linear_pred_bonds, device_ids=[rank], find_unused_parameters=True) 
        linear_pred_mlabes = DistributedDataParallel(linear_pred_mlabes, device_ids=[rank], find_unused_parameters=True) 
        
    model_list = [model, linear_pred_atoms, linear_pred_bonds, linear_pred_mlabes]
    
    #set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_mlabes = optim.Adam(linear_pred_mlabes.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_mlabes]
    print("=====Stage2: Phase 1 - original reaction centre masking")
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc_atom = train(args, model_list, data_loader_rc, optimizer_list, device, rank, world_size)
        print(train_loss, train_acc_atom)
    if not args.output_model_file == "":
       torch.save(model.state_dict(), args.output_model_file + "rc.pth")
       
    print("=====Stage2: Phase 2 - One Hop neighbours masking")
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc_atom = train(args, model_list, data_loader_1hop, optimizer_list, device, rank, world_size)
        print(train_loss, train_acc_atom)
    if not args.output_model_file == "":
       torch.save(model.state_dict(), args.output_model_file + "one_hop.pth")
       
    print("=====Stage2: Phase 3 - Two Hop neighbours masking")
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc_atom = train(args, model_list, data_loader_2hop, optimizer_list, device, rank, world_size)
        print(train_loss, train_acc_atom)
    if not args.output_model_file == "":
       torch.save(model.state_dict(), args.output_model_file + "two_hop.pth")
    
    print("=====Stage2: Phase 4 - Three Hop neighbours masking")
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc_atom = train(args, model_list, data_loader_3hop, optimizer_list, device, rank, world_size)
        print(train_loss, train_acc_atom)
    if not args.output_model_file == "":
       torch.save(model.state_dict(), args.output_model_file + "three_hop.pth")
    """
    for step, batch in tqdm(enumerate(data_loader)):
        # batch = batch.to(device)
        temp_graph, node_rep, mlabes = model(batch, device)
        print(temp_graph)
        print(node_rep.shape)
        print(len(mlabes))
        print(temp_graph.masked_atom_indices)
        # print(temp_graph.mask_node_label)
        # print(temp_graph.mask_edge_label)
        # print(mlabes[temp_graph.masked_atom_indices])
        print([mlabe for i, mlabe in enumerate(mlabes) if i in temp_graph.masked_atom_indices])
        print(temp_graph.x[temp_graph.masked_atom_indices])
        if step > 0:
            break
    """
if __name__ == "__main__":
    main()