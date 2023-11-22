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

from test.gcn_utils.datas import MoleculeDataset, DataLoaderMasking, MaskAtom

from data_analysis.USPTO_CONFIG import USPTO_CONFIG

from tqdm import tqdm
import numpy as np

from models.gnn.models import GNN

from sklearn.metrics import roc_auc_score

import pandas as pd
from data_processing import ReactionDataset, CustomCollator

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
        node_rep = self.gnn(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
        # print(node_rep.shape)
        # print(torch.isnan(node_rep).sum())
        tensor_2_head = torch.zeros((node_rep.size(0), node_rep.size(1) * 2), dtype=node_rep.dtype, device=node_rep.device)
        
        batch_size = len(batch_input['condition_idx'])
        
        for batch_num in range(batch_size):
            node_rep_single = node_rep[batch_graph.batch==batch_num]
            
            pri_num = max(batch_input['primary_idx'][batch_num]) + 1
            # cond_rep = node_rep_single
            atom_num = (batch_graph.batch==batch_num).sum().item()
            
            if pri_num < atom_num: # has conditional mol
                cond_rep = node_rep_single[pri_num:]
                cond_pool = torch.mean(cond_rep, dim=0, keepdim=True)
            else: # no conditional mol
                cond_pool = torch.zeros((1, node_rep.size(1)), dtype=node_rep.dtype, device=node_rep.device)
            # print(pri_num, atom_num, batch_input['primary_idx'], node_rep_single.shape)
            # concat
            tensor_2_head[batch_graph.batch==batch_num] = torch.cat([node_rep_single, torch.broadcast_to(cond_pool, node_rep_single.shape)], dim=1)
            # print(torch.isnan(tensor_2_head).sum())
        # print(tensor_2_head.shape)
        
        # print(torch.isnan(tensor_2_head).sum())
        return batch_input['graph'], tensor_2_head, batch_input['mlabes']
    
def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def train(args, model_list, loader, optimizer_list, device, org_dataloader_iterator, data_loader_org):
    model, linear_pred_atoms, linear_pred_bonds, linear_pred_mlabes, org_linear_pred_atoms, org_linear_pred_bonds = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_mlabes, org_optimizer_linear_pred_atoms, org_optimizer_linear_pred_bonds = optimizer_list

    model.train()
    linear_pred_atoms.train()
    linear_pred_bonds.train()
    linear_pred_mlabes.train()

    org_linear_pred_atoms.train()
    org_linear_pred_bonds.train()

    
    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0
    acc_mlabes_accum = 0
    acc_node_accum_org = 0
    

    
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        temp_graph, node_rep, mlabes = model(batch, device)

        # org_mask loss
        try:
            org_batch = next(org_dataloader_iterator)
        except StopIteration:
            org_dataloader_iterator = iter(data_loader_org)
            org_batch = next(org_dataloader_iterator)
        org_batch.to(device)
        node_rep_org = model.gnn(org_batch.x, org_batch.edge_index, org_batch.edge_attr)
        pred_node_org = org_linear_pred_atoms(node_rep_org[org_batch.masked_atom_indices])
        org_loss = criterion(pred_node_org.double(), org_batch.mask_node_label[:,0])

        acc_node_org = compute_accuracy(pred_node_org, org_batch.mask_node_label[:,0])
        acc_node_accum_org += acc_node_org
        ## loss for nodes
        
        pred_node = linear_pred_atoms(node_rep[temp_graph.masked_atom_indices])
        # tgt = torch.tensor([mlabe for i, mlabe in enumerate(batch["mlabes"]) if i in temp_graph.masked_atom_indices]).to(device)
        # loss = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
        loss_node = criterion(pred_node.double(), temp_graph.mask_node_label[:,0])
        
        acc_node = compute_accuracy(pred_node, temp_graph.mask_node_label[:,0])
        acc_node_accum += acc_node
         # Edge prediction
        masked_edge_index = temp_graph.edge_index[:, temp_graph.connected_edge_indices]
        edge_rep = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
        pred_edge = linear_pred_bonds(edge_rep)
        loss_edge = criterion(pred_edge.double(), temp_graph.mask_edge_label[:,0])
        loss = loss_node + loss_edge + org_loss

        acc_edge = compute_accuracy(pred_edge, temp_graph.mask_edge_label[:,0])
        acc_edge_accum += acc_edge
        
        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()

        org_optimizer_linear_pred_atoms.zero_grad()
        # optimizer_linear_pred_bonds.zero_grad()
        
        loss.backward()
        
        optimizer_model.step()
        optimizer_linear_pred_atoms.step()
        optimizer_linear_pred_bonds.step()

        org_optimizer_linear_pred_atoms.step()
        
        loss_accum += float(loss.cpu().item())
        
        if step % 100 == 1:
            wandb.log({"loss_accum": loss_accum/step,
                       "node_accuracy_accum": acc_node_accum/step,
                       "edge_accuracy_accum": acc_edge_accum/step,
                       "stage1_mask_attr": acc_node_accum_org/step,
                       "loss": loss,
                       "node_loss" : loss_node,
                       "edge_loss": loss_edge,
                       "state1_mask_loss": org_loss,
                       "stage1_mask_acc": acc_node_org,
                       "node_accuracy": acc_node,
                       "edge_accurcay": acc_edge,
                       
                       })
        # if step > 300:
        #     break
    return loss_accum/step, acc_node_accum/step, acc_edge_accum/step

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=80,
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

    parser.add_argument('--stage_one_model', type=str, default = "/share/project/sharefs-skfeng/Chemical_Reaction_Pretraining/test/attr_mask/baseline_final.pth", help="the path of stage one model")


    parser.add_argument('--stage_one_head_model', type=str, default = "/share/project/sharefs-skfeng/Chemical_Reaction_Pretraining/test/attr_mask/baseline_linear_pred_atoms_final.pth", help="the path of stage one model")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        
    # wandb.login(key='11aa569e656b0bc19ad079b4bdaaa9ea3553694a')
    wandb.init(project="Chemical-Reaction-Pretrainin-Stage2", name=args.run_name)
    model = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    model.load_state_dict(torch.load(args.stage_one_model))


    # org mask dataset
    root_dataset = '/home/Backup/chem/dataset'
    dataset = MoleculeDataset("{}/".format(root_dataset) + args.dataset, dataset=args.dataset, transform = MaskAtom(num_atom_type = 119, num_edge_type = 5, mask_rate = args.mask_rate, mask_edge=args.mask_edge))
    data_loader_org = DataLoaderMasking(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    
    org_linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    org_linear_pred_atoms.load_state_dict(torch.load(args.stage_one_head_model))
    org_linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)

    org_optimizer_linear_pred_atoms = optim.Adam(org_linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    org_optimizer_linear_pred_bonds = optim.Adam(org_linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)



    reaction_dataset = ReactionDataset(data_path=USPTO_CONFIG.dataset, atom_vocab=USPTO_CONFIG.atom_vocab_file)
    
    collator_rc = CustomCollator(mask_stage='reaction_centre')
    
    collator_1hop = CustomCollator(mask_stage='one_hop')
    collator_2hop = CustomCollator(mask_stage='two_hop')
    collator_3hop = CustomCollator(mask_stage='three_hop')
    
    data_loader_rc = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn = collator_rc)
    data_loader_1hop = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn = collator_1hop)
    data_loader_2hop = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn = collator_2hop)
    data_loader_3hop = torch.utils.data.DataLoader(reaction_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn = collator_3hop)
    
    model_react = ReactantStage2(model).to(device)
    linear_pred_atoms = torch.nn.Linear(args.emb_dim * 2, 119).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim * 2, 4).to(device)
    linear_pred_mlabes = torch.nn.Linear(args.emb_dim * 2, 2458).to(device)

    

    
    model_list = [model_react, linear_pred_atoms, linear_pred_bonds, linear_pred_mlabes, org_linear_pred_atoms, org_linear_pred_bonds]
    
    #set up optimizers
    optimizer_model = optim.Adam(model_react.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_mlabes = optim.Adam(linear_pred_mlabes.parameters(), lr=args.lr, weight_decay=args.decay)

    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_pred_mlabes, org_optimizer_linear_pred_atoms, org_optimizer_linear_pred_bonds]
    
    print("=====Stage2: Phase 1 - original reaction centre masking")
    org_dataloader_iterator = iter(data_loader_org)
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc_atom, train_acc_edge = train(args, model_list, data_loader_rc, optimizer_list, device, org_dataloader_iterator, data_loader_org)
        print(train_loss, train_acc_atom, train_acc_edge)
    if not args.output_model_file == "":
       torch.save(model.state_dict(), args.output_model_file + "rc.pth")
       
    print("=====Stage2: Phase 2 - One Hop neighbours masking")
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc_atom, train_acc_edge = train(args, model_list, data_loader_1hop, optimizer_list, device, org_dataloader_iterator, data_loader_org)
        print(train_loss, train_acc_atom, train_acc_edge)
    if not args.output_model_file == "":
       torch.save(model.state_dict(), args.output_model_file + "one_hop.pth")
       
    print("=====Stage2: Phase 3 - Two Hop neighbours masking")
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc_atom, train_acc_edge = train(args, model_list, data_loader_2hop, optimizer_list, device, org_dataloader_iterator, data_loader_org)
        print(train_loss, train_acc_atom, train_acc_edge)
    if not args.output_model_file == "":
       torch.save(model.state_dict(), args.output_model_file + "two_hop.pth")
    
    print("=====Stage2: Phase 4 - Three Hop neighbours masking")
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc_atom, train_acc_edge = train(args, model_list, data_loader_3hop, optimizer_list, device, org_dataloader_iterator, data_loader_org)
        print(train_loss, train_acc_atom, train_acc_edge)
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