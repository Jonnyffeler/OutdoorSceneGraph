import numpy as np
import json
import joblib
import pickle
from tqdm import tqdm
import os.path as osp
import argparse
import gc

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils import utils_subscenes
from utils.ObjectNode import ObjectNode

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the root folder')
    parser.add_argument('--split', type=str, help='data split', required=True)
    parser.add_argument('--scene', type=str, help='Scene name', required=True)
    parser.add_argument('--roots_only', action='store_true', help='If set, use only roots in SGAligner')
    args = parser.parse_args()
    
    return args

def main():
    args = parse_arguments()
    scene_name = args.scene
    out_path = osp.join(args.path, "out", scene_name)

    only_roots = args.roots_only
    path_only_roots = ""
    if only_roots:
        path_only_roots = "_roots_only"

    bow_file_name = "bow_attrs_LIN_LIN.json"
    split = args.split
    if split == "train":
        nr_scenes = 1000
    elif split == "val":
        nr_scenes = 200
    elif split == "test":
        nr_scenes = 250
    else:
        raise ValueError("Invalid split")
    
    npoint_downsample = 1024
    min_nr_points = 512

    ### Loading everything
    print("\n############################################")
    print(f"Creating subscenes for {scene_name} and {split} split...")
    print("############################################\n")

    print("loading objects and edges... ", end="", flush=True)
    roots = joblib.load(osp.join(out_path, f"{scene_name}_roots_described.joblib"))

    with open(osp.join(out_path, f"{scene_name}_edges.pkl"), "rb") as f:
        scenegraph_edges = pickle.load(f) 

    with open(osp.join(out_path, bow_file_name), "r") as f:
        bow_attrs = json.load(f)
    print("done")

    all_objs = list()
    all_objs += roots
    if not only_roots:
        for root in roots:
            all_objs += root.get_all_children()
    
    ### Downsampling pointcloud to be used for PointNet
    print("making sure pcd is correctly sampled... ", end="", flush=True)
    changed = False
    for obj in all_objs:
        if 'pcd_downsampled' in obj.object.keys():
            if len(obj.object['pcd_downsampled']) == npoint_downsample:
                continue
        obj.object['pcd_downsampled'] = utils_subscenes.pcl_farthest_sample_leo(obj.object['pcd'], npoint_downsample)
        changed = True
    if changed:
        joblib.dump(roots, osp.join(out_path, f"{scene_name}_roots_described.joblib"))
    print("done")
    
    edges = scenegraph_edges['edges']
    if only_roots:
        edges = edges[:scenegraph_edges['id_root_edges']]

    hyperedges = scenegraph_edges['hyperedges']
    hyperedge_attrs = scenegraph_edges['hyperedge_attrs']

    obj_pcds = []
    obj_bboxes = []
    obj_ids = []
    total_nr_points = 0
    for obj in all_objs:
        obj_pcds.append(obj.object['pcd'])
        obj_bboxes.append(np.column_stack(obj.object['bounding_box']))
        total_nr_points += len(obj.object['pcd'])
        obj_ids.append(obj.node_id)

    ### changing from idx (node.id) to ids in list
    print("formatting edges and hyperdges...", end="", flush=True)
    edges = [[obj_ids[i], obj_ids[j]] for i, j in edges]
    for key in hyperedges.keys():
        hyperedges[key] = utils_subscenes.hyperedges_idx_to_id(hyperedges[key], obj_ids)

    scenes_data = []
    overlaps = []
    print("done\n")
    print("creating scenes...")

    ### Creating subscenes
    for i in tqdm(range(nr_scenes)):
        scene_0_dict = {}
        scene_1_dict = {}
        
        ### sampling random plane that creates the two subscenes
        mean_point = np.mean(np.row_stack(obj_pcds), 0)
        normal, d, origin = utils_subscenes.sample_plane(mean_point)
        p_normal, p_d, p_origin = utils_subscenes.sample_plane(mean_point)
        
        ### split everything up into the two subscenes
        scene_0_dict['objects'] = utils_subscenes.get_objects_for_scene(obj_bboxes, all_objs, normal, d, min_nr_points, npoint_downsample)
        scene_1_dict['objects'] = utils_subscenes.get_objects_for_scene(obj_bboxes, all_objs, p_normal, p_d, min_nr_points, npoint_downsample)

        scene_0_dict['edges'], edges_big_0, edges_small_0 = utils_subscenes.get_edges_for_scene(edges, scene_0_dict['objects'], scenegraph_edges['id_root_edges'])
        scene_1_dict['edges'], edges_big_1, edges_small_1 = utils_subscenes.get_edges_for_scene(edges, scene_1_dict['objects'], scenegraph_edges['id_root_edges'])

        if only_roots:
            assert(len(edges_small_0)==0)
            assert(len(edges_small_1)==0)
        
        scene_0_dict['hyperedges'], scene_0_dict['hyperedge_attrs'] = utils_subscenes.get_hyperedges_for_scene(hyperedges, hyperedge_attrs, scene_0_dict['objects'])
        scene_1_dict['hyperedges'], scene_1_dict['hyperedge_attrs'] = utils_subscenes.get_hyperedges_for_scene(hyperedges, hyperedge_attrs, scene_1_dict['objects'])

        scene_0_dict['bow_vec_object_edge_feats'] = utils_subscenes.produce_bow_edge(list(scene_0_dict['objects'].keys()), edges_big_0, edges_small_0)
        scene_1_dict['bow_vec_object_edge_feats'] = utils_subscenes.produce_bow_edge(list(scene_1_dict['objects'].keys()), edges_big_1, edges_small_1)
        
        scene_0_dict['bow_vec_object_attr_feats'] = utils_subscenes.produce_bow_attr(all_objs, scene_0_dict['objects'], np.array(bow_attrs))
        scene_1_dict['bow_vec_object_attr_feats'] = utils_subscenes.produce_bow_attr(all_objs, scene_1_dict['objects'], np.array(bow_attrs))
        
        ### Computing the overlap of the two subscenes
        overlap = utils_subscenes.compute_overlap(all_objs, scene_0_dict, scene_1_dict)/total_nr_points
        overlaps.append(overlap)

        full_dictionnaries = {'scene_0': scene_0_dict, 'scene_1': scene_1_dict, 'overlap': overlap}
        scenes_data.append(full_dictionnaries)

        ### Saving the subscene metadata in batches (due to limited memory)
        if (i+1) % 100 == 0 or i == nr_scenes-1:  # Save every 50 scenes or at the end
            batch_num = i // 100
            print(f"Saving batch {batch_num} (scenes {batch_num*50} to {i})")
            with open(osp.join(out_path, f"{scene_name}_subscenes_metadata{path_only_roots}_{split}_batch_{batch_num}.pkl"), "wb") as f:
                pickle.dump(scenes_data, f)
            scenes_data = []  # Clear the list
            gc.collect()  # Force garbage collection

if __name__ == "__main__":
    main()