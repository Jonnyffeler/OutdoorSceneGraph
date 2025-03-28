import os
import os.path as osp
import glob
import numpy as np
import joblib, pickle
import gc
import psutil

import torch
import torch.utils.data as data

import sys
sys.path.append('..')
from OutdoorGraph.ObjectNode_it import ObjectNode

class LamarDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.split = split
        # load data (objects)
        self.dir = cfg.data.root_dir
        self.scene_name = cfg.data.scene_name
        self.only_roots = cfg.data.only_roots
        self.path_only_roots = ""
        print(f"initializing dataset for scene {self.scene_name} and split {self.split}...")
        self.roots = joblib.load(osp.join(self.dir, f"{self.scene_name}_roots_final_described_all_objs.joblib"))
        if self.only_roots:
            print("roots only")
            self.all_objects = self.roots
            self.path_only_roots = "_roots_only"
        else:
            self.all_objects = self._get_all_objects_from_roots(self.roots)
        self.batch_paths = sorted(glob.glob(osp.join(self.dir, f"{self.scene_name}_subscenes_metadata{self.path_only_roots}_{self.split}_batch_*")))
        self.nr_batches = len(self.batch_paths)
        self.batch_sizes = [100]*(self.nr_batches-1) + [len(self._load_data_from_file(self.batch_paths[-1]))]
        
        self.current_batch_idx = 0
        self.current_batch = None
        self.current_batch_size = 0
        self.global_indices = []
        self.total_size = 100*(self.nr_batches-1) + self._get_batch_size(self.nr_batches-1)
        self._load_batch(0)

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss/(1024*1024)
        print(f"Current memory usage: {memory_mb:.2f} MB")
    
    def _get_all_objects_from_roots(self, roots):
        all_objs = list()
        all_objs += roots
        for root in roots:
            all_objs += root.get_all_children()
        
        return all_objs
    
    def _load_data_from_file(self, path):
        with open(path, "rb") as f:
            subscenes_metadata = pickle.load(f)
        return subscenes_metadata


    def _get_batch_size(self, batch_idx):
        return self.batch_sizes[batch_idx]

    def _load_batch(self, batch_idx):
        print(f"loading batch {batch_idx}... ")
        if batch_idx >= len(self.batch_paths):
            raise IndexError(f"Batch index {batch_idx} out of range")
        
        if self.current_batch is not None:
            del self.current_batch
            gc.collect()

        self.current_batch = self._load_data_from_file(self.batch_paths[batch_idx])
        self.current_batch_idx = batch_idx
        self.current_batch_size = len(self.current_batch)

        start_idx = sum(self._get_batch_size(i) for i in range(batch_idx))
        self.global_indices = range(start_idx, start_idx + self.current_batch_size)
        print("done")

    def _global_to_local_idx(self, global_idx):
        if global_idx < 0 or global_idx >= self.total_size:
            raise IndexError(f"Index {global_idx} out of range")
        
        current_idx = 0
        for batch_idx, path in enumerate(self.batch_paths):
            batch_size = self._get_batch_size(batch_idx)
            if current_idx <= global_idx < current_idx + batch_size:
                local_idx = global_idx - current_idx
                return batch_idx, local_idx
            current_idx += batch_size
        
        raise IndexError("Index mapping error")

    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        batch_idx, local_idx = self._global_to_local_idx(idx)
        if batch_idx != self.current_batch_idx:
            self._load_batch(batch_idx)

        return self._getitem_local(local_idx)
    
    def reset(self):
        self._load_batch(0)
        self.current_batch_idx = 0
    
    def _get_obj_from_id(self, objects, key):
        for object in objects:
            if object.node_id == int(key):
                return object
        return None
    
    def get_objs_from_list(self, objects, scene):
        objs_in_scene = {}
        keys = list(scene.keys())
        for key in keys:
            obj = self._get_obj_from_id(objects, key)
            assert(obj is not None)# key error
            objs_in_scene[obj] = scene[obj.node_id]
        return objs_in_scene

    def get_pcds_from_obj_list(self, objects):
        pcds = []
        for object in objects.keys():
            ids = objects[object]
            if isinstance(ids, str):
                pcds.append(object.object['pcd_downsampled'])
            else:
                pcds.append(ids)
        return np.array(pcds)
    
    def ids_to_idx(self, edges, idxs):
        ids = [] 
        for edge in edges:
            ids.append([np.where(edge[0]==idxs)[0], np.where(edge[1]==idxs)[0]])
        return np.squeeze(np.array(ids))
    
    def ids_to_idx_hyper(self, hyperedges, idxs):
        cleaned = np.copy(hyperedges)
        for i, val in enumerate(cleaned[0]):
            assert(val in idxs)
            cleaned[0, i] = np.where(val==idxs)[0]
        return cleaned
    
    def get_overlapping_objs(self, objects_0, objects_1):
        intersection = set(objects_0) & set(objects_1)
        intersection = np.array(list(intersection))

        obj_ids_0 = np.array(objects_0)
        obj_ids_1 = np.array(objects_1)

        e1i_idxs = np.where(np.in1d(obj_ids_0, intersection))[0]
        e2i_idxs = np.where(np.in1d(obj_ids_1, intersection))[0]

        # take the "inverse"
        e1j_idxs = np.arange(len(obj_ids_0))
        e1j_idxs = np.delete(e1j_idxs, e1i_idxs)

        e2j_idxs = np.arange(len(obj_ids_1))
        e2j_idxs = np.delete(e2j_idxs, e2i_idxs)

        return e1i_idxs, e1j_idxs, e2i_idxs, e2j_idxs
    
    def _transform_hyperdges(self, hyperedges):
        transformed = np.copy(hyperedges)
        unique_values, counts = np.unique(hyperedges[1], return_counts=True)
        transformed[1] = np.array([i for i, count in zip(range(len(unique_values)), counts) for _ in range(count)])
        return transformed
        
    def _getitem_local(self, idx):
        subscene_metadata = self.current_batch[idx] 
        scene_0 = subscene_metadata['scene_0']
        scene_1 = subscene_metadata['scene_1']
        overlap = subscene_metadata['overlap']

        object_ids_0 = list(scene_0['objects'].keys())
        object_ids_1 = list(scene_1['objects'].keys())

        objects_0 = self.get_objs_from_list(self.all_objects, scene_0['objects'])
        objects_1 = self.get_objs_from_list(self.all_objects, scene_1['objects'])

        pcds_0 = self.get_pcds_from_obj_list(objects_0)
        pcds_1 = self.get_pcds_from_obj_list(objects_1)

        if self.split == 'train':
            if np.random.rand(1)[0] > 0.5:
                pcl_center = np.mean(np.row_stack(pcds_0), axis=0)
            else:
                pcl_center = np.mean(np.row_stack(pcds_1), axis=0)
        else:
            pcl_center = np.mean(np.row_stack(pcds_0), axis=0)

        src_object_points = pcds_0 - pcl_center
        ref_object_points = pcds_1 - pcl_center

        tot_object_points = torch.cat([torch.from_numpy(src_object_points), torch.from_numpy(ref_object_points)]).type(torch.FloatTensor)

        src_edges = self.ids_to_idx(scene_0['edges'], np.array(object_ids_0))
        ref_edges = self.ids_to_idx(scene_1['edges'], np.array(object_ids_1))

        edges = torch.cat([torch.from_numpy(src_edges), torch.from_numpy(ref_edges)])

        e1i_idxs, e1j_idxs, e2i_idxs, e2j_idxs = self.get_overlapping_objs(object_ids_0, object_ids_1)

        e2i_idxs += src_object_points.shape[0]
        e2j_idxs += src_object_points.shape[0]

        obj_attrs_0 = np.array([obj.object['node_embedding'] for obj in objects_0])
        obj_attrs_1 = np.array([obj.object['node_embedding'] for obj in objects_1])

        object_attrs = torch.cat([torch.from_numpy(obj_attrs_0), torch.from_numpy(obj_attrs_1)])

        node_embs_0 = np.array([obj.object['fv'] for obj in objects_0])
        node_embs_1 = np.array([obj.object['fv'] for obj in objects_1])
        #print(node_embs_0.shape)#768

        node_embs = torch.cat([torch.from_numpy(node_embs_0), torch.from_numpy(node_embs_1)])


        ######## hyperedges
        src_hyperedges =  self.ids_to_idx_hyper(scene_0['hyperedges']['full'], np.array(object_ids_0))
        src_hyperedges = self._transform_hyperdges(src_hyperedges)
        ref_hyperedges =  self.ids_to_idx_hyper(scene_1['hyperedges']['full'], np.array(object_ids_1))
        ref_hyperedges = self._transform_hyperdges(ref_hyperedges)
        hyperedges = torch.cat([torch.from_numpy(src_hyperedges).T, torch.from_numpy(ref_hyperedges).T])

        src_hyperedge_attrs =  scene_0['hyperedge_attrs']['full']
        ref_hyperedge_attrs =  scene_1['hyperedge_attrs']['full']
        hyperedge_attrs = torch.cat([torch.from_numpy(src_hyperedge_attrs), torch.from_numpy(ref_hyperedge_attrs)])

        tot_bow_vec_obj_edge_feats = torch.cat([torch.from_numpy(scene_0['bow_vec_object_edge_feats']), torch.from_numpy(scene_1['bow_vec_object_edge_feats'])])
        tot_bow_vec_obj_attr_feats = torch.cat([torch.from_numpy(scene_0['bow_vec_object_attr_feats']), torch.from_numpy(scene_1['bow_vec_object_attr_feats'])])


        data_dict = {} 
        data_dict['obj_ids'] = np.concatenate([np.array(object_ids_0), np.array(object_ids_1)])
        data_dict['tot_obj_pts'] = tot_object_points #pcd 
        data_dict['graph_per_obj_count'] = np.array([src_object_points.shape[0], ref_object_points.shape[0]])
        data_dict['graph_per_edge_count'] = np.array([src_edges.shape[0], ref_edges.shape[0]])
        
        data_dict['e1i'] = e1i_idxs
        data_dict['e1i_count'] = e1i_idxs.shape[0]
        data_dict['e2i'] = e2i_idxs
        data_dict['e2i_count'] = e2i_idxs.shape[0]
        data_dict['e1j'] = e1j_idxs
        data_dict['e1j_count'] = e1j_idxs.shape[0]
        data_dict['e2j'] = e2j_idxs
        data_dict['e2j_count'] = e2j_idxs.shape[0]
        
        data_dict['tot_obj_count'] = tot_object_points.shape[0]
        data_dict['tot_bow_vec_object_attr_feats'] = tot_bow_vec_obj_attr_feats
        data_dict['tot_bow_vec_object_edge_feats'] = tot_bow_vec_obj_edge_feats
        data_dict['tot_rel_pose'] = object_attrs # node_embs
        data_dict['edges'] = edges 

        data_dict['object_attrs'] = object_attrs
        data_dict['hyperedges'] = hyperedges
        data_dict['hyperedge_attrs'] = hyperedge_attrs
        data_dict['graph_per_hyperedge_count'] = np.array([src_hyperedges.shape[1], ref_hyperedges.shape[1]])
        data_dict['graph_per_hyperedge_attr_count'] = np.array([src_hyperedge_attrs.shape[0], ref_hyperedge_attrs.shape[0]])#np.array([len(np.unique(src_hyperedges[1])), len(np.unique(src_hyperedges[1]))])

        #data_dict['global_obj_ids'] = global_object_ids
        #data_dict['scene_ids'] = [src_scan_id, ref_scan_id]        
        data_dict['pcl_center'] = pcl_center
        data_dict['overlap'] = overlap
        
        return data_dict

    def _collate_entity_idxs(self, batch):
        e1i = np.concatenate([data['e1i'] for data in batch])
        e2i = np.concatenate([data['e2i'] for data in batch])
        e1j = np.concatenate([data['e1j'] for data in batch])
        e2j = np.concatenate([data['e2j'] for data in batch])

        e1i_start_idx = 0 
        e2i_start_idx = 0 
        e1j_start_idx = 0 
        e2j_start_idx = 0 
        prev_obj_cnt = 0
        
        for idx in range(len(batch)):
            e1i_end_idx = e1i_start_idx + batch[idx]['e1i_count']
            e2i_end_idx = e2i_start_idx + batch[idx]['e2i_count']
            e1j_end_idx = e1j_start_idx + batch[idx]['e1j_count']
            e2j_end_idx = e2j_start_idx + batch[idx]['e2j_count']

            e1i[e1i_start_idx : e1i_end_idx] += prev_obj_cnt
            e2i[e2i_start_idx : e2i_end_idx] += prev_obj_cnt
            e1j[e1j_start_idx : e1j_end_idx] += prev_obj_cnt
            e2j[e2j_start_idx : e2j_end_idx] += prev_obj_cnt
            
            e1i_start_idx, e2i_start_idx, e1j_start_idx, e2j_start_idx = e1i_end_idx, e2i_end_idx, e1j_end_idx, e2j_end_idx
            prev_obj_cnt += batch[idx]['tot_obj_count']
        
        e1i = e1i.astype(np.int32)
        e2i = e2i.astype(np.int32)
        e1j = e1j.astype(np.int32)
        e2j = e2j.astype(np.int32)

        return e1i, e2i, e1j, e2j

    def _collate_feats(self, batch, key):
        feats = torch.cat([data[key] for data in batch])
        return feats
    
    def collate_fn(self, batch):
        tot_object_points = self._collate_feats(batch, 'tot_obj_pts')
        tot_bow_vec_object_attr_feats = self._collate_feats(batch, 'tot_bow_vec_object_attr_feats')
        tot_bow_vec_object_edge_feats = self._collate_feats(batch, 'tot_bow_vec_object_edge_feats')    
        tot_rel_pose = self._collate_feats(batch, 'tot_rel_pose')
        object_attrs = self._collate_feats(batch, 'object_attrs')
        
        data_dict = {}
        data_dict['tot_obj_pts'] = tot_object_points
        data_dict['e1i'], data_dict['e2i'], data_dict['e1j'], data_dict['e2j'] = self._collate_entity_idxs(batch)

        data_dict['e1i_count'] = np.stack([data['e1i_count'] for data in batch])
        data_dict['e2i_count'] = np.stack([data['e2i_count'] for data in batch])
        data_dict['e1j_count'] = np.stack([data['e1j_count'] for data in batch])
        data_dict['e2j_count'] = np.stack([data['e2j_count'] for data in batch])
        data_dict['tot_obj_count'] = np.stack([data['tot_obj_count'] for data in batch])
        #data_dict['global_obj_ids'] = np.concatenate([data['global_obj_ids'] for data in batch])
        
        data_dict['tot_bow_vec_object_attr_feats'] = tot_bow_vec_object_attr_feats.double()
        data_dict['tot_bow_vec_object_edge_feats'] = tot_bow_vec_object_edge_feats.double()
        data_dict['tot_rel_pose'] = tot_rel_pose.double()
        data_dict['graph_per_obj_count'] = np.stack([data['graph_per_obj_count'] for data in batch])
        data_dict['graph_per_edge_count'] = np.stack([data['graph_per_edge_count'] for data in batch])
        data_dict['edges'] = self._collate_feats(batch, 'edges')
        #data_dict['scene_ids'] = np.stack([data['scene_ids'] for data in batch])
        data_dict['obj_ids'] = np.concatenate([data['obj_ids'] for data in batch])
        data_dict['pcl_center'] = np.stack([data['pcl_center'] for data in batch])

        data_dict['object_attrs'] = object_attrs
        data_dict['hyperedges'] = self._collate_feats(batch, 'hyperedges')
        data_dict['hyperedge_attrs'] = self._collate_feats(batch, 'hyperedge_attrs')
        data_dict['graph_per_hyperedge_count'] = np.stack([data['graph_per_hyperedge_count'] for data in batch])
        data_dict['graph_per_hyperedge_attr_count'] = np.stack([data['graph_per_hyperedge_attr_count'] for data in batch]) 
        
        data_dict['overlap'] = np.stack([data['overlap'] for data in batch])
        data_dict['batch_size'] = data_dict['overlap'].shape[0]

        return data_dict
        
if __name__ == '__main__':
    from configs import config_scan3r_gt
    cfg = config_scan3r_gt.make_cfg()
    ds = LamarDataset(cfg, split='val')
    print(len(ds))
    ds[0]    