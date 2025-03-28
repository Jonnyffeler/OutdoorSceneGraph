import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

from aligner.networks.gat import MultiGAT
from aligner.networks.pointnet import PointNetfeat
from aligner.networks.pct import NaivePCT
from aligner.networks.hgat import HGN_Attn, HGN_Conv

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super(ProjectionHead, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.l2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.l2(x)
        return x

class MultiModalFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)), requires_grad=self.requires_grad)

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
        joint_emb = torch.cat(embs, dim=1)
        return joint_emb

class MultiModalEncoder(nn.Module):
    def __init__(self, modules, rel_dim, attr_dim, hidden_units=[64, 128, 128], heads = [2, 2], dim_hyperedge_emb=384, emb_dim = 100, pt_out_dim = 256,
                       dropout = 0.0, attn_dropout = 0.0, instance_norm = False, attn_mode="node"):
        super(MultiModalEncoder, self).__init__()
        self.modules = modules
        self.pt_out_dim = pt_out_dim
        self.rel_dim = rel_dim
        self.emb_dim = emb_dim
        self.attr_dim =  attr_dim
        self.hidden_units = hidden_units
        self.heads = heads
        self.dim_hyperedge_emb = dim_hyperedge_emb
        
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.instance_norm = instance_norm
        self.inner_view_num = len(self.modules) # Point Net + Structure Encoder + Meta Encoder
        self.attn_mode = attn_mode

        self.meta_embedding_rel = nn.Linear(self.rel_dim, self.emb_dim)
        self.meta_embedding_attr = nn.Linear(self.attr_dim, self.emb_dim)
        
        if 'point' in self.modules:
            self.object_encoder = PointNetfeat(global_feat=True, batch_norm=True, point_size=3, input_transform=False, feature_transform=False, out_size=self.pt_out_dim)
        elif 'pct' in self.modules:
            self.object_encoder = NaivePCT()
        else:
            print("no point")#raise NotImplementedError
        
        self.object_embedding = nn.Linear(self.pt_out_dim, self.emb_dim)
        
        self.node_embedding = nn.Linear(384, 64)#768
        
        self.structure_encoder = MultiGAT(n_units=self.hidden_units, n_heads=self.heads, dropout=self.dropout)
        self.structure_embedding = nn.Linear(256, self.emb_dim)

        self.hypergraph_encoder = HGN_Attn(n_units=[64, 128, 128], n_heads=self.heads, dim_hyperedge_emb=self.dim_hyperedge_emb, dropout=0.5, attn_mode=self.attn_mode)
        self.hypergraph_embedding = nn.Linear(256, self.emb_dim)
        
        self.fusion = MultiModalFusion(modal_num=self.inner_view_num, with_weight=1)

    def point_in_batches(self, tot_object_points, batch_size=64):
        num_objects = tot_object_points.shape[0]
        all_embs = None

        for i in range(0, num_objects, batch_size):
            torch.cuda.empty_cache()
            gc.collect()
            batch = tot_object_points[i:i+batch_size]
            batch_emb = self.object_encoder(batch)

            if all_embs is None:
                all_embs = torch.empty((num_objects, batch_emb.shape[1]), dtype=tot_object_points.dtype, device='cpu')

            all_embs[i:i+len(batch)] = batch_emb.detach().cpu()

            del batch, batch_emb
        return all_embs
        
    def forward(self, data_dict):
        tot_object_points = data_dict['tot_obj_pts'].permute(0, 2, 1)
        tot_bow_vec_object_attr_feats = data_dict['tot_bow_vec_object_attr_feats'].float() 
        tot_bow_vec_object_edge_feats = data_dict['tot_bow_vec_object_edge_feats'].float()
        tot_rel_pose = data_dict['tot_rel_pose'].float()
        
        start_object_idx = 0
        start_edge_idx = 0
        batch_size = data_dict['batch_size']
        
        embs = {}

        for module in self.modules:
            if module == 'gat':
                start_object_idx = 0
                start_edge_idx = 0
                structure_embed = None
                for idx in range(batch_size):
                    src_object_count = data_dict['graph_per_obj_count'][idx][0]
                    ref_object_count = data_dict['graph_per_obj_count'][idx][1]

                    src_edges_count = data_dict['graph_per_edge_count'][idx][0]
                    ref_edges_count = data_dict['graph_per_edge_count'][idx][1]
                    
                    src_objects_rel_pose = tot_rel_pose[start_object_idx : start_object_idx + src_object_count]
                    start_object_idx += src_object_count

                    ref_objects_rel_pose = tot_rel_pose[start_object_idx : start_object_idx + ref_object_count]
                    start_object_idx += ref_object_count

                    
                    src_edges = torch.transpose(data_dict['edges'][start_edge_idx : start_edge_idx + src_edges_count], 0, 1).to(torch.int32)
                    start_edge_idx += src_edges_count

                    ref_edges = torch.transpose(data_dict['edges'][start_edge_idx : start_edge_idx + ref_edges_count], 0, 1).to(torch.int32)
                    start_edge_idx += ref_edges_count

                    src_node_embeddings = self.node_embedding(src_objects_rel_pose)
                    ref_node_embeddings = self.node_embedding(ref_objects_rel_pose)

                    src_structure_embedding = self.structure_encoder(src_node_embeddings, src_edges)
                    ref_structure_embedding = self.structure_encoder(ref_node_embeddings, ref_edges)
                    
                    structure_embed = torch.cat([src_structure_embedding, ref_structure_embedding]) if structure_embed is None else \
                                   torch.cat([structure_embed, src_structure_embedding, ref_structure_embedding]) 

                emb = self.structure_embedding(structure_embed)
            
            elif module in ['point', 'pct']:
                assert(module=='point')
                device = tot_object_points.device
                point_batch_size = 256
                if tot_object_points.shape[0] <= 768:
                    point_batch_size = 768
                emb = self.point_in_batches(tot_object_points, batch_size=point_batch_size).to(device)
                emb = self.object_embedding(emb)

            elif module == 'rel':
                emb = self.meta_embedding_rel(tot_bow_vec_object_edge_feats)
            
            elif module == 'attr':
                emb = self.meta_embedding_attr(tot_bow_vec_object_attr_feats)

            elif module == 'hgn':
                start_object_idx = 0
                start_edge_idx = 0
                start_edge_attr_idx = 0
                structure_embed = None
                for idx in range(batch_size):
                    src_object_count = data_dict['graph_per_obj_count'][idx][0]
                    ref_object_count = data_dict['graph_per_obj_count'][idx][1]

                    src_hyperedges_count = data_dict['graph_per_hyperedge_count'][idx][0]
                    ref_hyperedges_count = data_dict['graph_per_hyperedge_count'][idx][1]

                    src_hyperedges_attr_count = data_dict['graph_per_hyperedge_attr_count'][idx][0]
                    ref_hyperedges_attr_count = data_dict['graph_per_hyperedge_attr_count'][idx][1]
                    
                    use_rel_trans = False
                    if use_rel_trans:
                        src_objects_rel_pose = tot_rel_pose[start_object_idx : start_object_idx + src_object_count]
                        start_object_idx += src_object_count
                        ref_objects_rel_pose = tot_rel_pose[start_object_idx : start_object_idx + ref_object_count]
                        start_object_idx += ref_object_count
                    else:
                        src_objects_rel_pose = data_dict['tot_rel_pose'][start_object_idx : start_object_idx + src_object_count].float()
                        start_object_idx += src_object_count
                        ref_objects_rel_pose = data_dict['tot_rel_pose'][start_object_idx : start_object_idx + ref_object_count].float()
                        start_object_idx += ref_object_count
                    
                    src_hyperedges = data_dict['hyperedges'][start_edge_idx : start_edge_idx + src_hyperedges_count].to(torch.int64).T
                    start_edge_idx += src_hyperedges_count
                    ref_hyperedges = data_dict['hyperedges'][start_edge_idx : start_edge_idx + ref_hyperedges_count].to(torch.int64).T
                    start_edge_idx += ref_hyperedges_count

                    src_hyperedge_attrs = data_dict['hyperedge_attrs'][start_edge_attr_idx : start_edge_attr_idx + src_hyperedges_attr_count].to(torch.float)
                    start_edge_attr_idx += src_hyperedges_attr_count
                    ref_hyperedge_attrs = data_dict['hyperedge_attrs'][start_edge_attr_idx : start_edge_attr_idx + ref_hyperedges_attr_count].to(torch.float)
                    start_edge_attr_idx += ref_hyperedges_attr_count

                    src_node_embeddings = self.node_embedding(src_objects_rel_pose)
                    ref_node_embeddings = self.node_embedding(ref_objects_rel_pose)

                    src_structure_embedding = self.hypergraph_encoder(src_node_embeddings, src_hyperedges, src_hyperedge_attrs)
                    ref_structure_embedding = self.hypergraph_encoder(ref_node_embeddings, ref_hyperedges, ref_hyperedge_attrs)
                    
                    structure_embed = torch.cat([src_structure_embedding, ref_structure_embedding]) if structure_embed is None else \
                                   torch.cat([structure_embed, src_structure_embedding, ref_structure_embedding]) 

                emb = self.hypergraph_embedding(structure_embed)
            
            else:
                raise NotImplementedError
            
            embs[module] = emb
        
        if len(self.modules) > 1:
            all_embs = []
            for module in self.modules:
                all_embs.append(embs[module])
            
            joint_emb = self.fusion(all_embs)
            embs['joint'] = joint_emb
        
        return embs