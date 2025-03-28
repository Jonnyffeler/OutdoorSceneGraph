import numpy as np
import torch
from PIL import Image
import open3d as o3d 
from utils.ObjectNode import ObjectNode
import colorsys
import numba as nb

def initialize_run_time_dict(times_per_image):
    times_per_image['raycasting_segment'] = []
    times_per_image['downsample_dbscan'] = []
    times_per_image['number_intersecting'] = []
    times_per_image['first_merge'] = []
    times_per_image['loop_merge'] = []
    times_per_image['build_candidate_trees'] = []
    times_per_image['trees_number_int'] = []
    times_per_image['merging_trees'] = []
    return times_per_image

def prune_tree(node, size_threshold_min=0.05, size_threshold_max=0.95, threshold_max=0.95):
    if not node.children:
        return  # No children to prune

    total_size = len(node.object['pcd'])  # Size of the current node
    child_sizes = {child: len(child.object['pcd']) for child in node.children}  # Dict for tracking sizes

    # Set to track unique children (to prevent duplicates)
    unique_children = set()

    for child in node.children:
        if len(child.children) == 1:
            print("removing lonely child")
            lonely_child = child.children[0]
            if lonely_child not in unique_children:
                unique_children.add(lonely_child)
                lonely_child.change_parent(node)
            continue

        child_size_ratio = child_sizes[child] / total_size
        if len(child.children) == 0:
            if child not in unique_children and child_size_ratio <= threshold_max:
                unique_children.add(child)
            if child_size_ratio > threshold_max:print("removing too big kid")
            continue
        if size_threshold_min <= child_size_ratio <= size_threshold_max:
            if child not in unique_children:
                unique_children.add(child)
        else:
            print("removing too big/small layer")
            for grand_child in child.children:
                if grand_child not in unique_children:
                    unique_children.add(grand_child)
                    grand_child.change_parent(node)

    # Apply the pruned list of children after processing
    node.children = list(unique_children)

    # Recursively prune the remaining children
    for child in node.children:
        prune_tree(child, size_threshold_min, size_threshold_max)

def find_objects_in_view(objects, intrinsic_mat, im_width, im_height, pose, max_dist):
    if len(objects)==0:
        return {}
    
    f_x = intrinsic_mat[0, 0]
    f_y = intrinsic_mat[1, 1]

    # vectors for camera
    cam_pos = pose[:3, 3] # position of the camera (point)
    forw = pose[:3, 2]
    right = pose[:3, 1]
    up = pose[:3, 0]
    # make sure they are normalized
    assert(np.allclose(np.linalg.norm(np.array([forw, right, up]), axis=0), np.ones(3)))

    # normals of frustum planes: using a/b = b/c
    near_norm = forw
    far_norm = -forw
    right_norm = forw - f_x/im_width * right
    left_norm = forw + f_x/im_width * right
    up_norm = forw - f_y/im_height * up
    down_norm = forw + f_x/im_width * up
    normals = np.array([near_norm, far_norm, right_norm, left_norm, up_norm, down_norm])

    # D values for plane equations: from points on the frustum planes
    near_D = -np.dot(near_norm, cam_pos)
    far_D = -np.dot(far_norm, cam_pos + max_dist*forw)
    right_D = -np.dot(right_norm, cam_pos)
    left_D = -np.dot(left_norm, cam_pos)
    up_D = -np.dot(up_norm, cam_pos)
    down_D = -np.dot(down_norm, cam_pos)
    Ds = np.array([near_D, far_D, right_D, left_D, up_D, down_D])

    bboxes = np.array(list(objects.values()))
    vertices = bbox_to_vertices(bboxes)

    plane_eqs = np.tensordot(normals, vertices, axes=1) + Ds[:, np.newaxis, np.newaxis]
    plane_eqs = plane_eqs > 0 # if they are inside/right side of the plane

    bbox_outside = np.any(plane_eqs, axis=1) # check if the full bbox is outside one plane
    ids_objs_in_view = np.where(np.all(bbox_outside, axis=0))[0]

    objects_in_view = {}
    keys = list(objects.keys())
    for id in ids_objs_in_view:
        key = keys[id] # key is the object, objects[key] is its bounding box
        objects_in_view[key] = objects[key]

    return objects_in_view

def extr_intrinsics_from_saved(camera_intrinsics, poses_trajectory):
    intr_mat = np.diag(camera_intrinsics[2:5])
    intr_mat[2,2] = 1
    intr_mat[0,2] = camera_intrinsics[4]
    intr_mat[1,2] = camera_intrinsics[5]
    intr_mat = (intr_mat).astype(float)

    quaternions = np.array(poses_trajectory[[" qw", " qx", " qy", " qz"]])
    poses = []
    for quat in range(quaternions.shape[0]):
        rotation_mat = o3d.geometry.get_rotation_matrix_from_quaternion(quaternions[quat])
        translation = np.array(poses_trajectory[[" tx", " ty", " tz"]])[quat]
        poses.append(np.row_stack([np.column_stack([rotation_mat, translation]), np.identity(4)[3]]))
    
    return intr_mat, poses


def create_SAM_masking(image, mask_generator, minimum_area):
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        masks = mask_generator.generate(image)

    sorted_masks = sorted(masks, key=(lambda x: x['area'])) # sort masks according to size (area)
    sorted_masks = [mask for mask in sorted_masks if mask['area']>minimum_area]
    
    return sorted_masks

def find_feature_vector(image, mask, model, preprocess, device, paddings=[0, 20, -1]):
    features = []
    for mask_seg in mask:
        mask_segment = False
        image_to_use = image

        # find the bounding box of the mask segment
        bbox = mask_seg["bbox"]
        x_from, y_from, _, _ = bbox
        x_to, y_to = x_from + bbox[2], y_from + bbox[3]
        (height, width, _) = image.shape


        feature_per_padding = []
        for padding in paddings:
            
            if padding == -1:
                padding = 0
                mask_segment = True

            # add padding and make sure to stay inside image
            x_from = int(max(0, x_from-padding))
            y_from = int(max(0, y_from-padding))
            x_to = int(min(width, x_to+padding))
            y_to = int(min(height, y_to+padding))

            if mask_segment:
                img = np.copy(image)
                img[np.invert(mask_seg["segmentation"])] = np.array([254, 254, 254])
                image_to_use = img

            # crop image to bounding box
            cropped_image = Image.fromarray(image_to_use[y_from:y_to, x_from:x_to, :])
            
            image_pre = preprocess(images=cropped_image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model(**image_pre).last_hidden_state[0][0].detach().cpu().numpy()
            
            feature_per_padding.append(image_features)
        feature_per_padding = np.array(feature_per_padding)
        features.append(np.mean(feature_per_padding, axis=0))

    return np.array(features)

def compute_bbox(pcd):
    mins = np.min(pcd, axis=0)
    maxs = np.max(pcd, axis=0)

    return np.concatenate((mins, maxs))

def bbox_to_vertices(bboxes):
    if bboxes.ndim == 1:
        bboxes = bboxes[np.newaxis, :]
        
    x_min, y_min, z_min, x_max, y_max, z_max = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3], bboxes[:, 4], bboxes[:, 5]
    edges = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]]).transpose(1, 0, 2)

    return edges

def find_intersecting_bboxes(bbox, bboxes):
    # bbox, bboxes: np.arrays
    bboxes_min = bboxes[:, :3]
    bboxes_max = bboxes[:, 3:]
    bbox_min = bbox[:3]
    bbox_max = bbox[3:]
    
    # Intersecting
    intersecting = np.all(bboxes_max >= bbox_min, axis=1) & np.all(bboxes_min <= bbox_max, axis=1)

    return np.where(intersecting)[0]


#################### Fusing ###################
def remove_object(fusing_object, objects, objects_in_view, objects_intersecting):
    if fusing_object.parent is not None:
        fusing_object.parent.remove_child(fusing_object)
    objects_intersecting.pop(fusing_object)
    objects_in_view.pop(fusing_object)
    objects.pop(fusing_object)

def update_bbox(objects, objects_in_view, object_to_fuse_to):
    updated_bbox = compute_bbox(object_to_fuse_to.object['pcd'])
    objects[object_to_fuse_to] = updated_bbox
    objects_in_view[object_to_fuse_to] = updated_bbox
    
    return objects, objects_in_view 

def remove_trees_from_intersecting_objects(object_to_fuse_to, objects_intersecting):
    if object_to_fuse_to.parent is not None:
        objs_to_pop = []
        for obj in objects_intersecting.keys():
            if obj.parent is not None and obj.parent is not object_to_fuse_to.parent:
                objs_to_pop.append(obj)

        for obj in objs_to_pop:
            objects_intersecting.pop(obj)

    return objects_intersecting

def append_new_object(new_object, new_bbox, objects, objects_in_view):
    objects[new_object] = new_bbox
    objects_in_view[new_object] = new_bbox

    return objects, objects_in_view

def build_new_object(image_nr, seg_nr, pcd_seg, seg_area, bbox_perc, bbox_mask, feature_vector, max_tree_id):
    imseg_id = [[image_nr, seg_nr]]
    imseg_nr = [[len(pcd_seg)]]
    new_object = {'pcd': pcd_seg, 'fv': feature_vector, 'imseg_id': imseg_id, 'imseg_nr': imseg_nr, 'imseg_area': [[seg_area]], 'bbox_perc': [[bbox_perc]], 'bbox_mask': [bbox_mask]}
    new_object = ObjectNode(new_object, max_tree_id)

    return new_object

@nb.njit(fastmath=True)
def voxelize_and_hash(points, voxel_size):
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    
    # Encode single integer hash
    x, y, z = voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    voxel_hashes = x + (y << 20) + (z << 40)  #single int
    return voxel_hashes


@nb.njit(parallel=True, fastmath=True)
def count_intersecting_pcds(pcd, pcd_object, voxel_size):
    hashes_pcd = voxelize_and_hash(pcd, voxel_size)
    unique_hashes_a = set(hashes_pcd)
    hashes_obj = voxelize_and_hash(pcd_object, voxel_size)
    
    intersect_count = 0
    for i in nb.prange(len(hashes_obj)):  # Parallelized loop over Cloud B
        if hashes_obj[i] in unique_hashes_a:  # If the voxel hash is in Cloud A's set
            intersect_count += 1  # This is an intersecting point
    
    return intersect_count

def compute_nr_intersecting_points(pcd, objects, voxel_size):
    if len(objects) == 0:
        return np.array([])
    
    # speed up by looking at bounding boxes first
    bbox = compute_bbox(pcd)

    # ids where bboxes intersect
    ids_to_check = find_intersecting_bboxes(bbox, np.array(list(objects.values())))
    nr_intersecting_points = np.zeros(len(objects))
    keys = list(objects.keys()) # keys are objects
    for id_check in ids_to_check:
        object = keys[id_check]
        nr_intersecting_points[id_check] = count_intersecting_pcds(pcd, object.object['pcd'], voxel_size)
        
    return np.nan_to_num(np.array(nr_intersecting_points)).astype(int)

#@nb.njit(parallel=True, fastmath=True)
def fuse_pcds(pcd1, pcd2, voxel_size):
    fused_points = list(pcd1)
    hashes1 = set(voxelize_and_hash(pcd1, voxel_size))
    hashes2 = voxelize_and_hash(pcd2, voxel_size)

    for i, point in enumerate(pcd2):
        if hashes2[i] not in hashes1:  # If point is not in Cloud A
            fused_points.append(point)  # Add it to the fused cloud
            hashes1.add(hashes2[i])   # Add its voxel index to the set of hashes for fast lookup
    
    return np.array(fused_points)


def fuse_new_to_existing_object(object_to_fuse_to, new_object, voxel_size):
    objs_to_update = [object_to_fuse_to] + object_to_fuse_to.get_all_parents()
    for obj_to_update in objs_to_update:
        pcd = fuse_pcds(obj_to_update.object['pcd'], new_object.object['pcd'], voxel_size)
        # weighted mean
        fv = (len(obj_to_update.object['pcd'])*obj_to_update.object['fv'] + len(new_object.object['pcd'])*new_object.object['fv'])/(len(obj_to_update.object['pcd'])+len(new_object.object['pcd']))
        imseg_id = obj_to_update.object['imseg_id'] + new_object.object['imseg_id']
        imseg_nr = obj_to_update.object['imseg_nr'] + new_object.object['imseg_nr']
        bbox_mask = obj_to_update.object['bbox_mask'] + new_object.object['bbox_mask']
        seg_area = obj_to_update.object['imseg_area'] + new_object.object['imseg_area']
        bbox_perc = obj_to_update.object['bbox_perc'] + new_object.object['bbox_perc']

        obj_to_update.object['pcd'] = pcd
        obj_to_update.object['fv'] = fv
        obj_to_update.object['imseg_id'] = imseg_id
        obj_to_update.object['imseg_nr'] = imseg_nr
        obj_to_update.object['bbox_mask'] = bbox_mask
        obj_to_update.object['imseg_area'] = seg_area
        obj_to_update.object['bbox_perc'] = bbox_perc


def compute_fused_object(objects_to_fuse, voxel_size): # Fuse all objects together in the list of objects to fuse.
    fvs = np.zeros(len(objects_to_fuse[0]['fv']))
    fvs_normalizer = 0
    imseg_id = []
    imseg_nr = []
    bbox_mask = []
    imseg_area = []
    bbox_perc = []

    assert(len(objects_to_fuse) == 2)
    pcd = fuse_pcds(objects_to_fuse[0]['pcd'], objects_to_fuse[1]['pcd'], voxel_size)

    for object in objects_to_fuse:
        weight_for_average_fv = len(object['imseg_id'])
        fvs += weight_for_average_fv*object['fv']
        fvs_normalizer += weight_for_average_fv

        imseg_id += object['imseg_id']
        imseg_nr += object['imseg_nr']
        bbox_mask += object['bbox_mask']
        imseg_area += object['imseg_area']
        bbox_perc += object['bbox_perc']


    fv = fvs/fvs_normalizer

    return {'pcd': pcd, 'fv': fv, 'imseg_id': imseg_id, 'imseg_nr': imseg_nr, 'bbox_mask': bbox_mask, 'imseg_area': imseg_area, 'bbox_perc': bbox_perc}

def compute_similarity_scores(new_object, objects_to_compare, nr_intersecting_points, min_sem_sim, min_geom_sim, alpha_sem, alpha_geom, mode='normal'):
    # compute semantic similarities (cosine distance) 
    objects_to_compare_fvs = []
    objects_to_compare_lengths = [] # used later
    for object in objects_to_compare:
        objects_to_compare_fvs.append(object.object['fv'])
        objects_to_compare_lengths.append(len(object.object['pcd']))
    objects_to_compare_fvs = np.array(objects_to_compare_fvs)
    sem_sims = np.dot(objects_to_compare_fvs, new_object.object['fv'])/(2*np.linalg.norm(new_object.object['fv'])*np.linalg.norm(objects_to_compare_fvs, axis=1)) + 0.5
    
    sem_sims[np.where(sem_sims < min_sem_sim)] = 0

    new_object_length = len(new_object.object['pcd'])
    prc_intersecting_new = nr_intersecting_points/new_object_length
    prc_intersecting_comp = nr_intersecting_points/objects_to_compare_lengths
    mean_prc_intersecting = np.mean(np.array([prc_intersecting_new, prc_intersecting_comp]), axis=0)
    max_prc_intersecting = np.max(np.array([prc_intersecting_new, prc_intersecting_comp]), axis=0)
    prc_used = prc_intersecting_new
    if mode == 'mean':
        prc_used = mean_prc_intersecting
    if mode == 'max':
        prc_used = max_prc_intersecting
    
    geom_sims = prc_used
    geom_sims[np.where(geom_sims < min_geom_sim)] = 0

    return alpha_sem*sem_sims + alpha_geom*geom_sims



###################### Trees #########################
def compute_nr_intersecting_points_tree(main_tree, comparing_trees, voxel_size):
    if len(comparing_trees) == 0:
        return np.array([])

    nr_intersecting_points = []
    for comp_tree in comparing_trees:
        nr_intersecting_points.append(count_intersecting_pcds(main_tree.object['pcd'], comp_tree.object['pcd'], voxel_size))
        
    return np.array(nr_intersecting_points).astype(int), None


def compute_similarity_scores_tree(main_tree, comparing_trees, c_trees, min_sem_sim, min_geom_sim, alpha_sem, alpha_geom, mode='normal'):
    objects_to_compare = comparing_trees
    nr_intersecting_points = np.array(c_trees)
    # compute semantic similarities (cosine distance) 
    objects_to_compare_fvs = []
    objects_to_compare_lengths = [] # used later
    for object in objects_to_compare:
        objects_to_compare_fvs.append(object.object['fv'])
        objects_to_compare_lengths.append(len(object.object['pcd']))
    objects_to_compare_fvs = np.array(objects_to_compare_fvs)
    sem_sims = np.dot(objects_to_compare_fvs, main_tree.object['fv'])/(2*np.linalg.norm(main_tree.object['fv'])*np.linalg.norm(objects_to_compare_fvs, axis=1)) + 0.5
    
    sem_sims[np.where(sem_sims < min_sem_sim)] = 0

    new_object_length = len(main_tree.object['pcd'])
    prc_intersecting_new = nr_intersecting_points/new_object_length
    prc_intersecting_comp = nr_intersecting_points/objects_to_compare_lengths
    mean_prc_intersecting = np.mean(np.array([prc_intersecting_new, prc_intersecting_comp]), axis=0)
    max_prc_intersecting = np.max(np.array([prc_intersecting_new, prc_intersecting_comp]), axis=0)
    prc_used = prc_intersecting_new
    if mode == 'mean':
        prc_used = mean_prc_intersecting
    if mode == 'max':
        prc_used = max_prc_intersecting
    ############

    geom_sims = prc_used
    geom_sims[np.where(geom_sims < min_geom_sim)] = 0

    return alpha_sem*sem_sims + alpha_geom*geom_sims


def fuse_TreeNodes(node1, node2, max_tree_id, voxel_size):
    fused_objects = compute_fused_object([node1.object, node2.object], voxel_size)
    parent_node = ObjectNode(fused_objects, node1.id)
    parent_node.change_node_id(max_tree_id)
    parent_node.add_children_nodes_only([node1, node2])
    node1.change_parent(parent_node)
    node2.change_parent(parent_node)
    node2.change_tree_id(node1.id)


#################### Visualisations ##########################
def display_mesh_and_trees(mesh, roots, plot_bbox=True, plot_roots=True, plot_object_centers=True, plot_tree_ids=False, min_size_large_obj=0, center_radius=0.3, return_segments=False):
    segments = []
    if mesh is not None:
        segments.append(mesh)

    hues = np.linspace(0, 1, len(roots))
    
    for root in roots:
        if len(root.object['pcd']) > min_size_large_obj:
            root_color = np.array([0, 0.5, 0.75])
            random_idx = np.random.randint(len(hues))
            root_color[0] = hues[random_idx] # random hue for the tree
            hues = np.delete(hues, random_idx)

            root_pcd = root.object['pcd']
            root_node = np.mean(root_pcd, axis=0)
            root_node[2] = 15

            if plot_roots:
                parent_node = o3d.geometry.TriangleMesh.create_sphere(radius=0.7)
                parent_node.compute_vertex_normals()
                parent_node.paint_uniform_color(colorsys.hls_to_rgb(*root_color))
                parent_node.translate(root_node)
                segments.append(parent_node)

            if plot_bbox:
                bbox = o3d.geometry.AxisAlignedBoundingBox(np.min(root_pcd, 0), np.max(root_pcd, 0))
                bbox.color = colorsys.hls_to_rgb(*root_color)
                segments.append(bbox)

            if plot_tree_ids:
                text3d = o3d.t.geometry.TriangleMesh.create_text(str(root.id), depth=2).to_legacy()
                text3d.paint_uniform_color(colorsys.hls_to_rgb(*root_color))
                R = np.array([
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]
                ])
                text3d.rotate(R)
                text3d.scale(0.1, center=text3d.get_center())
                text3d.translate(root_node-text3d.get_center() + [0, 0, 7])
                segments.append(text3d)

            if len(root.children) == 0: # if root is simply an object and not a tree
                seg = o3d.geometry.PointCloud()
                seg.points = o3d.utility.Vector3dVector(root.object['pcd'])
                seg.paint_uniform_color(colorsys.hls_to_rgb(*root_color))
                segments.append(seg)
                continue

            total_size = len(root_pcd)

            for i, child in enumerate(root.get_all_children()):
                color_hls = np.copy(root_color)
                if len(child.object['pcd'])/total_size > 0.8:
                    color_hls[1] = 0.7 + 0.1*np.random.random(1)
                else:
                    color_hls[1] = 0.3 + 0.4*np.random.random(1)
                
                color = colorsys.hls_to_rgb(*color_hls)

                seg = o3d.geometry.PointCloud()
                seg.points = o3d.utility.Vector3dVector(child.object['pcd']+0.001*i*(+np.array([0, 0, 1])))
                seg.paint_uniform_color(color)
                segments.append(seg)

                if plot_object_centers:
                    obj_center = o3d.geometry.TriangleMesh.create_sphere(radius=center_radius)
                    obj_center.compute_vertex_normals()
                    obj_center.paint_uniform_color(color)
                    obj_center.translate(np.mean(child.object['pcd'], axis=0))
                    segments.append(obj_center)

    if return_segments:
        return segments
        
    o3d.visualization.draw_geometries(segments)

def display_mesh_and_subtrees(mesh, roots, plot_bbox=True, plot_roots=True, min_size_large_obj=0):
    segments = []
    if mesh is not None:
        segments.append(mesh)

    for root in roots:
        if len(root.children) > 0 and len(root.object['pcd']) > min_size_large_obj:
            subtrees = []
            for child in root.children:
                if len(child.children) > 0:
                    subtrees.append(child)

            if len(subtrees)>0:
                root_color = np.array([1,0,0])

                root_pcd = root.object['pcd']
                root_node = np.mean(root_pcd, axis=0)
                root_node[2] = 15

                if plot_roots:
                    parent_node = o3d.geometry.TriangleMesh.create_sphere(radius=0.7)
                    parent_node.compute_vertex_normals()
                    parent_node.paint_uniform_color(root_color)
                    parent_node.translate(root_node)
                    segments.append(parent_node)

                if plot_bbox:
                    bbox = o3d.geometry.AxisAlignedBoundingBox(np.min(root_pcd, 0), np.max(root_pcd, 0))
                    bbox.color = root_color
                    segments.append(bbox)

                segs = display_mesh_and_trees(mesh, subtrees, plot_bbox=False, plot_roots=False, plot_object_centers=True, min_size_large_obj=0, center_radius=0.3, return_segments=True)
                segments += segs

    o3d.visualization.draw_geometries(segments)


def print_tree(node, level=0):
    print("     " * level + str(node.node_id), ": ", len(node.object['pcd']))
    for child in node.children:
        print_tree(child, level + 1)