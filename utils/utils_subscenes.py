import numpy as np
import utils.utils_objects as utils_objects
from utils.ObjectNode import ObjectNode
import itertools
import unicodedata
import re
import fpsample

def sample_point_in_cylinder(R, h_min=0, h_max=0.2):
    r = np.sqrt(np.random.uniform(0, 1)) * R
    theta = np.random.uniform(0, 2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(h_min, h_max)
    
    return np.array((x, y, z))

def sample_plane(mean_point):
    sampled_origin = sample_point_in_cylinder(5) + mean_point
    sampled_direction_point = sample_point_in_cylinder(1, 0, 0.2) + sampled_origin
    normal = sampled_direction_point - sampled_origin
    normal /= np.linalg.norm(normal)
    plane_d = -np.dot(normal, sampled_origin)

    return normal, plane_d, sampled_origin

def check_bbox_plane(bbox, normal, d):
    vertices_bbox = utils_objects.bbox_to_vertices(bbox)[:, :, 0].T
    on_side = np.dot(vertices_bbox, normal) + d
    on_side = on_side > 0
    if np.all(on_side):
        return True
    if not np.any(on_side):
        return False
    else:
        return "between"

def get_objects_for_scene(root_bboxes, objs, normal, d, min_nr_points, npoint_downsample):
    scene_dict = {}
    for i, bbox in enumerate(root_bboxes):
        on_side = check_bbox_plane(bbox, normal, d)
        if on_side is True:
            scene_dict[objs[i].node_id] = "full"
        if on_side == "between":
            pcd = objs[i].object['pcd']
            distances = np.dot(pcd, normal) + d
            ids = np.where(distances > 0)[0]
            if len(ids) > min_nr_points:
                scene_dict[objs[i].node_id] = pcl_farthest_sample_leo(pcd[ids], npoint_downsample).astype(np.float32)
            
    return scene_dict

def get_edges_for_scene(edges, scene_dict, id_root_edges):
    edges_in_scene = []
    object_ids = list(scene_dict.keys())
    split_id = len(edges)
    find_split = True
    for i, edge in enumerate(edges):
        append_edge = True
        for edge_id in edge:
            if edge_id not in object_ids:
                append_edge = False
                break
        if append_edge:
            if find_split and i >= id_root_edges:# first time we have 'part of' edges 
                split_id = len(edges_in_scene)
                find_split = False
            edges_in_scene.append(edge)

    return edges_in_scene, edges_in_scene[:split_id], edges_in_scene[split_id:]

def _get_hyperedges_scene(hyperedges, hyperedge_attrs, scene_dict):
    obj_ids = list(scene_dict.keys())
    hyperedges_in_scene = []
    hyperedge_attrs_in_scene_ids = set()
    edge_ids = np.unique(hyperedges[1])
    for edge_nr, edge_id in enumerate(edge_ids):
        assert(edge_nr==edge_id)
        objs_in_hyperedge = hyperedges[0][np.where(hyperedges[1]==edge_id)[0]]
        for obj in objs_in_hyperedge:
            if obj in obj_ids:
                hyperedges_in_scene.append(np.array([obj, edge_id]))
                hyperedge_attrs_in_scene_ids.add(edge_id)

    hyperedge_attrs_in_scene = [hyperedge_attrs[id] for id in hyperedge_attrs_in_scene_ids]

    return np.column_stack(hyperedges_in_scene), np.row_stack(hyperedge_attrs_in_scene)

def get_hyperedges_for_scene(hyperedges, hyperedge_attrs, scene_dict):
    hyperedges_in_scene = {}
    hyperedge_attrs_in_scene = {}

    keys = list(hyperedges.keys())
    for key in keys:
        hyperedges_in, hyperedge_attrs_in = _get_hyperedges_scene(hyperedges[key], hyperedge_attrs[key], scene_dict)
        hyperedges_in_scene[key] = hyperedges_in
        hyperedge_attrs_in_scene[key] = hyperedge_attrs_in

    return hyperedges_in_scene, hyperedge_attrs_in_scene

def hyperedges_idx_to_id(hyperedges, obj_ids):
    hyperedges_idx = np.copy(hyperedges)
    for i, obj_nr in enumerate(hyperedges[0]):
        hyperedges_idx[0, i] = obj_ids[obj_nr]
    return hyperedges_idx

def compute_overlap(roots, scene_0_dict, scene_1_dict):
    overlap = 0

    data_dict_1 = scene_0_dict['objects']
    data_dict_2 = scene_1_dict['objects']

    keys_1 = list(data_dict_1.keys())
    keys_2 = list(data_dict_2.keys())
    for root in roots:
        if root.node_id in keys_1 and root.node_id in keys_2:
            ids_1 = data_dict_1[root.node_id]
            ids_2 = data_dict_2[root.node_id]
            if isinstance(ids_1, str) and isinstance(ids_2, str):
                overlap += len(root.object['pcd'])
            elif isinstance(ids_1, str):
                overlap += len(ids_2)
            elif isinstance(ids_2, str):
                overlap += len(ids_1)
            else:
                tol = 1e-4
                pcd1_rounded = np.round(ids_1 / tol) * tol
                pcd2_rounded = np.round(ids_2 / tol) * tol
                pcd1_set = set(map(tuple, pcd1_rounded))  # Convert each point to a tuple and then to a set
                pcd2_set = set(map(tuple, pcd2_rounded))

                intersection = pcd1_set & pcd2_set
                overlap += len(intersection)
                
    return overlap

def _get_obj_from_id(objects, key):
    for object in objects:
        if object.node_id == int(key):
            return object
    return None

def normalize_attr(attr):
    attr = unicodedata.normalize("NFKC", attr)  # Normalize Unicode characters
    attr = attr.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
    attr = re.sub(r'\s+', ' ', attr)  # Replace multiple spaces with a single space
    attr = attr.replace(" ", "")  # Remove spaces completely to match "shopfront" and "shop front"
    return attr

def produce_bow_attr(objs, obj_ids, bow):
    bow_attrs = np.zeros((len(obj_ids), len(bow)))
    keys = list(obj_ids.keys())
    for i, key in enumerate(keys):
        attrs = _get_obj_from_id(objs, key).object['attributes']
        for attr in attrs:
            norm_attr = normalize_attr(attr)
            assert(norm_attr in bow)
            j = np.where(norm_attr==bow)[0] 
            bow_attrs[i, j] = 1
    return bow_attrs

def pcl_farthest_sample_leo(point, npoint, return_idxs=False):
    N, D = point.shape
    if N < npoint:
        indices = np.random.choice(point.shape[0], npoint)
        points = point[indices]
        return points
    
    idx = fpsample.bucket_fps_kdtree_sampling(np.array(point), npoint)
    points = point[idx]
    if return_idxs: return points, idx
    return points

def produce_bow_edge(object_ids, edges_big, edges_small):
    bow_feats = np.zeros((len(object_ids), 2))
    edges_big_flat = set(itertools.chain.from_iterable(edges_big))
    edges_small_flat = set(itertools.chain.from_iterable(edges_small))
    for i, object_id in enumerate(object_ids):
        if object_id in edges_big_flat:
            bow_feats[i, 0] = 1
        if object_id in edges_small_flat:
            bow_feats[i, 1] = 1
    return bow_feats