import open3d as o3d
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import pickle
from utils import utils_objects as utils_objects
#import utils.utils_objects as utils_objects
from utils.ObjectNode import ObjectNode
import time
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
from transformers import AutoModel, AutoImageProcessor
from collections import Counter
import joblib
from tqdm import tqdm
import argparse


def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the root folder')
    parser.add_argument('--scene', type=str, help='scene to be run')
    parser.add_argument('--show_runtimes', action='store_true', help='If set, runtimes are plotted')

    args = parser.parse_args()
    return parser, args

def main():
    parser, args = parse_args()
    scene_name = args.scene
    root_path = args.path
    out_path = osp.join(root_path, "out", scene_name)
    data_path = osp.join(root_path, "Lamar", scene_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        os.makedirs(osp.join(out_path, "sam"))
        os.makedirs(osp.join(out_path, "dino"))

    print("\n#########################################")
    print(f"Building Object Instances for scene: {scene_name}")
    print("#########################################\n")

    ### Reading everything
    print("reading in data...")
    start = time.time()
    imagedb = pd.read_csv(osp.join(data_path, "images.txt"), header = 0)
    poses_trajectory = pd.read_csv(osp.join(data_path, "trajectories.txt"), header = 0) 
    camera_intrinsics = np.array(pd.read_csv(osp.join(data_path, "sensors.txt"), header = 1).iloc[0, 4:10])
    im_width, im_height = camera_intrinsics[0], camera_intrinsics[1]
    intrinsic_mat, poses = utils_objects.extr_intrinsics_from_saved(camera_intrinsics, poses_trajectory)
    mesh = o3d.io.read_triangle_mesh(osp.join(data_path, "proc", "meshes", "mesh_simplified.ply"))
    mesh.compute_triangle_normals()
    print("done")

    ### loading SAM and DINOv2
    print("loading SAM and DINOv2...")
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_h"](checkpoint="../SAM_checkpoint/sam_vit_h_4b8939.pth")
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=16,
                pred_iou_thresh=0.88,
                min_mask_region_area=1000,
                box_nms_thresh=0.7,
                points_per_batch=144
            )

    preprocess_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    print("done")

    ### Parameters
    image_nrs = range(len(imagedb[" image_path"]))
    voxel_size = 0.15
    voxel_size_downsampling = 0.15
    minimum_area_of_segment = 500
    minimum_nr_points_of_segment = 25
    max_dist = 40 # how far objects can be to still look at within pose
    min_length_segment = 1.2

    # params views -> object
    obj_thresh = 1.1 # similarity to other objects (higher -> more new objects)
    obj_min_sem_sim = 0.75 # minimum semantic similarity (cosine similarity) needed to be fused to an object
    obj_min_geom_sim = 0.3 # minimum geometric similarity needed to be fused to an object
    # weights for semantic or geometric similarity
    alpha_sem_obj = 1.1
    alpha_geom_obj = 0.9
    mode = 'max' # when comparing two objects, how should percentage of intersecting points be normalized. 'normal' -> new object, 'max', or 'mean'
    args_sim_score = (obj_min_sem_sim, obj_min_geom_sim, alpha_sem_obj, alpha_geom_obj, mode)

    # params object -> larger object
    large_obj_thresh = 0.9 # was 0.9
    large_min_sem_sim = 0.65 #was 0.7
    large_min_geom_sim = 0.35 # was 0.35
    alpha_sem_obj = 0.85
    alpha_geom_obj = 1.15
    args_sim_score_large = (large_min_sem_sim, large_min_geom_sim, alpha_sem_obj, alpha_geom_obj, 'mean')

    # running lists (database)
    max_tree_id = 0
    objects = {} # the list of objects is a dictionary where the keys are the objects and the values are their 3D bbox

    # run time analysis
    times_per_view = []
    total_times = []

    ### used for raycasting
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    ### iterate through all images (views)
    for image_nr in tqdm(image_nrs):
        if True:
            times_per_image = {} # used for run time analysis
            
            ### loading image
            starting_time = time.time()
            im_path = osp.join(data_path, "raw_data", imagedb[" image_path"][image_nr][1:])
            image = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (680, 960), interpolation=cv2.INTER_AREA)

            ### creating masks by using SAM
            mask_path = osp.join(out_path, "sam", f"mask_{image_nr}.pkl")
            if os.path.isfile(mask_path):
                with open(mask_path, "rb") as read_file:
                    mask = pickle.load(read_file)
            else:
                mask = utils_objects.create_SAM_masking(image, mask_generator, minimum_area_of_segment)
                with open(mask_path, 'wb') as file_out:
                    pickle.dump(mask, file_out)
            mask = sorted(mask, key=(lambda x: x['area']), reverse=True)
            times_per_image['sam'] = time.time()-starting_time
            
            ### using DINOv2 to create features for each class in the masking
            start = time.time()
            feature_path = osp.join(out_path, "dino", f"dino_{image_nr}.npy")
            if os.path.isfile(feature_path):
                features_per_class = np.load(feature_path, allow_pickle=True)
            else:
                features_per_class = utils_objects.find_feature_vector(image, mask, dino, preprocess_dino, device)
                np.save(feature_path, features_per_class)
            torch.cuda.empty_cache()
            times_per_image['dino'] = time.time()-start

            ### Using raycasting to map to 3D
            start = time.time()
            pose = poses[image_nr]
            pose_inv = np.linalg.inv(pose)
            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                    intrinsic_matrix=intrinsic_mat,
                    extrinsic_matrix=pose_inv,
                    width_px=int(image.shape[1]),
                    height_px=int(image.shape[0])
                    )
            ans = scene.cast_rays(rays)
            hit = ans['t_hit'].isfinite().numpy()
            rays_numpy = rays.numpy()
            # 3D points (intersections of rays and mesh)
            rays_points = rays_numpy[:,:,:3] + rays_numpy[:,:,3:]*ans['t_hit'].numpy()[:,:,None] 
            times_per_image['raycasting_view'] = time.time()-start

            ### finding objects in camera/view frustum
            objects_in_view = utils_objects.find_objects_in_view(objects, intrinsic_mat, im_width, im_height, pose, max_dist)
            
            # for run time analysis
            times_per_image = utils_objects.initialize_run_time_dict(times_per_image)

            ### iterating through all proposed object instances
            for seg_nr, segment in enumerate(mask):
                ### finding 3D points of proposed object instance
                start = time.time()
                hit_seg = np.logical_and(hit, segment['segmentation'])
                pcd_seg = rays_points[hit_seg]
                times_per_image['raycasting_segment'].append(time.time()-start)

                ### Downsampling and denoising pointcloud
                start = time.time()
                if len(pcd_seg) < minimum_nr_points_of_segment: continue
                pcd_seg = o3d.t.geometry.PointCloud(pcd_seg).voxel_down_sample(voxel_size_downsampling).to_legacy()
                labels_seg = np.array(pcd_seg.cluster_dbscan(eps= 1, min_points=5))
                biggest_clusters = Counter(labels_seg).most_common(2)
                
                if biggest_clusters[0][0] != -1: # choose biggest cluster. If it is -1 we take the second biggest
                    pcd_seg =  np.array(pcd_seg.points)[labels_seg == biggest_clusters[0][0]]
                elif len(biggest_clusters)>1: # choose biggest cluster. If it is -1 we take the second biggest
                    pcd_seg =  np.array(pcd_seg.points)[labels_seg == biggest_clusters[1][0]]
                else:
                    pcd_seg = []
                times_per_image['downsample_dbscan'].append(time.time()-start)

                # if the segment doesn't have enough points or isnt' large enough we skip it
                if len(pcd_seg) < minimum_nr_points_of_segment: continue
                max_len_of_seg_bbox = np.max(np.abs(np.max(pcd_seg, axis=0)-np.min(pcd_seg, axis=0)))
                if max_len_of_seg_bbox < min_length_segment:continue

                ### building the new object
                feature_vector = features_per_class[seg_nr]
                bbox_mask = segment['bbox']
                seg_area = segment['area']
                bbox_perc = segment['area']/(segment['bbox'][2]*segment['bbox'][3])
                new_object = utils_objects.build_new_object(image_nr, seg_nr, pcd_seg, seg_area, bbox_perc, bbox_mask, feature_vector, max_tree_id)
                new_bbox = utils_objects.compute_bbox(pcd_seg)
                max_tree_id += 1

                ### finding the number of intersecting points with new object for all objects in view frustum
                start = time.time()
                nr_intersecting_points = utils_objects.compute_nr_intersecting_points(pcd_seg, objects_in_view, voxel_size)
                times_per_image['number_intersecting'].append(time.time()-start)

                # if there aren't any objects with partial overlap we directly initialize a new object
                if (nr_intersecting_points==0).all():
                    objects, objects_in_view = utils_objects.append_new_object(new_object, new_bbox, objects, objects_in_view)
                    continue

                ### computing similiarity scores for all objects with intersecting point clouds
                ids_intersecting_to_view = np.where(nr_intersecting_points>0)[0]
                nr_intersecting_points = nr_intersecting_points[ids_intersecting_to_view]
                keys = list(objects_in_view.keys()) # keys are the objects
                # dictionary of objects with intersecting pointcloud with proposed object instance
                objects_intersecting = {keys[id]: objects_in_view[keys[id]] for id in ids_intersecting_to_view}

                similarity_scores_obj = utils_objects.compute_similarity_scores(new_object, list(objects_intersecting.keys()), nr_intersecting_points, *args_sim_score)

                ### creating new object if no existing object is similar enough otherwise repeatedly merging them
                if np.all(similarity_scores_obj <= obj_thresh):
                    # create new object
                    objects, objects_in_view = utils_objects.append_new_object(new_object, new_bbox, objects, objects_in_view)
                else:
                    ### merging the new view to an existing object
                    start = time.time()
                    object_to_fuse_to = list(objects_intersecting.keys())[np.argmax(similarity_scores_obj)] # maximum similarity
                    utils_objects.fuse_new_to_existing_object(object_to_fuse_to, new_object, voxel_size_downsampling)

                    # update bbox
                    objects, objects_in_view = utils_objects.update_bbox(objects, objects_in_view, object_to_fuse_to)

                    # remove the chosen object from comparing lists
                    objects_intersecting.pop(object_to_fuse_to)
                    times_per_image['first_merge'].append(time.time()-start)

                    ### repeatedly checking if the newly fused object merges to other objects (only if not both are part of a tree yet)
                    start = time.time()
                    while True:
                        # if the newly fused object is part of a tree we have to remove objects from comparing lists that are in a tree
                        objects_intersecting = utils_objects.remove_trees_from_intersecting_objects(object_to_fuse_to, objects_intersecting)

                        # if the list of comparing objects is empty we stop looking to merge
                        if len(objects_intersecting)==0: break

                        # compute number of intersecting points and sim scores for all remaining intersecting objects
                        nr_intersecting_points = utils_objects.compute_nr_intersecting_points(object_to_fuse_to.object['pcd'], objects_intersecting, voxel_size)
                        similarity_scores_obj = utils_objects.compute_similarity_scores(object_to_fuse_to, list(objects_intersecting.keys()), nr_intersecting_points, *args_sim_score)

                        # if all remaining objects have too low similarity score we stop the merging process
                        if np.max(similarity_scores_obj) < obj_thresh: break

                        # most similar object, will be fused to object_to_fuse_to
                        fusing_object = list(objects_intersecting.keys())[np.argmax(similarity_scores_obj)] 
                        # fuse other object (not in a tree) to exisitng one (that might be a tree)
                        utils_objects.fuse_new_to_existing_object(object_to_fuse_to, fusing_object, voxel_size_downsampling) 
                        
                        utils_objects.remove_object(fusing_object, objects, objects_in_view, objects_intersecting)
                        objects, objects_in_view = utils_objects.update_bbox(objects, objects_in_view, object_to_fuse_to)
                    
                    new_object = object_to_fuse_to # the new object we will consider in tree building/merging process
                    times_per_image['loop_merge'].append(time.time()-start)



                ### Building trees

                ### finding all important trees
                start = time.time()
                main_tree = new_object.get_root()
                comparing_trees = set()
                # looking through all objects with intersecting pointclouds to initial newly proposed instance (that haven't already been merged)
                for obj_int in objects_intersecting.keys():
                    comp_tree = obj_int.get_all_parents()
                    if len(comp_tree) == 0: #object isn't part of a tree yet
                        comp_tree.append(obj_int)
                    # if any node of the comparing is part of the main tree get rid of whole tree
                    intersection = set(comp_tree).intersection([main_tree])
                    if len(intersection) == 0:
                        comparing_trees.update(comp_tree)
                comparing_trees = list(comparing_trees)
                times_per_image['build_candidate_trees'].append(time.time()-start)

                # if we don't have any trees to compare to we continue
                if len(comparing_trees)==0: continue 

                ### iteratively merging trees according to merging rules
                while True:
                    start = time.time()
                    nr_intersecting_tree, _ = utils_objects.compute_nr_intersecting_points_tree(main_tree, comparing_trees, voxel_size)
                    sim_score_tree = utils_objects.compute_similarity_scores_tree(main_tree, comparing_trees, nr_intersecting_tree, *args_sim_score_large)
                    times_per_image['trees_number_int'].append(time.time()-start)

                    if np.max(sim_score_tree) < large_obj_thresh: break

                    arg_max_tree = np.argmax(sim_score_tree)
                    best_tree = comparing_trees[arg_max_tree]

                    ### checking different merging cases
                    start = time.time()
                    # case 1 in paper (main tree is always a root) -> create new tree
                    if len(best_tree.children)==0 and best_tree.parent is None and len(main_tree.children)==0:
                        utils_objects.fuse_TreeNodes(main_tree, best_tree, max_tree_id, voxel_size_downsampling)
                        max_tree_id += 1
                    # case 2 in paper -> best tree is appended to main root
                    elif best_tree.parent is None and len(best_tree.children)==0:
                        main_tree.add_child(best_tree, voxel_size_downsampling)
                    # case 2 in paper -> main root (object) is appended to best tree
                    elif len(main_tree.children)==0: 
                        best_tree.add_child(main_tree, voxel_size_downsampling)
                    else:
                        # case 3 in paper -> fuse the smaller to the larger object
                        if best_tree.parent is None:
                            if len(best_tree.object['pcd']) < len(main_tree.object['pcd']):
                                main_tree.add_child(best_tree, voxel_size_downsampling)
                            else:
                                best_tree.add_child(main_tree, voxel_size_downsampling)
                        # case 4 in paper 
                        else:
                            best_tree.add_child(main_tree, voxel_size_downsampling)

                    ### removing all trees from comparing trees that are part of the new main tree
                    main_tree = main_tree.get_root()
                    trees_to_remove = []
                    for i, comp_tree in enumerate(comparing_trees):
                        if comp_tree.id == best_tree.id or comp_tree.id == main_tree.id:
                            trees_to_remove.append(comp_tree)
                    for remove_tree in trees_to_remove:
                        comparing_trees.remove(remove_tree)
                    times_per_image['merging_trees'].append(time.time()-start)

                    # stop tree building process if no trees to compare to are left
                    if len(comparing_trees)==0: break

            total_times.append(time.time()-starting_time)
            times_per_view.append(times_per_image)

        """
        except Exception as e:
            print(f"Error {e} occured at image {image_nr}")"
            """

    print(f"Done building objects. In total {len(objects)} were constructed.")

    ### Post processing
    print("Postprocessing...")
    objects_og = objects
    objects = []
    obj_out = []
    for obj in objects_og.keys():
        if len(obj.object['pcd'])>10 and len(obj.object['imseg_id']) > 1:
            objects.append(obj)
        else:
            obj_out.append(obj)

    roots = set()
    for o in objects:
        roots.add(o.get_root())
    roots = list(roots)

    all_objs = list()
    all_objs += roots
    for root in roots:
        all_objs += root.get_all_children()
    for obj in all_objs:
        object_pcd = obj.object['pcd']
        obj.object['bs_center'] = np.mean(object_pcd, axis=0)
        obj.object['bounding_box'] = [[np.min(object_pcd, 0)], [np.max(object_pcd, 0)]]

    print("objects kept: ", len(objects))
    print("objects removed: ", len(obj_out))
    print("nr of roots: ", len(roots))

    ### pruning tree 
    # run several times to get the final pruned tree
    for pruning in range(10):
        for root in roots:
            utils_objects.prune_tree(root, 0.1, 0.7)

    ### saving final roots:
    joblib.dump(roots, osp.join(out_path, f"{scene_name}_roots.joblib"))

    ### displaying run times:
    if args.show_runtimes:
        full_times = []
        for image_time in times_per_view:
            times = []
            for key in image_time.keys():
                times.append(sum(image_time[key]) if isinstance(image_time[key], list) else image_time[key]) # sum because per image and not per segment
            full_times.append(times)

        full_times = np.array(full_times)
        labels = ['Generating/Loading SAM Masks', 'Generating/Loading DINOv2 Features', "Raycasting per Image", 
                  "Raycasting Sum of Segments", "Downsampling and DBSCAN Denoising Object Pointcloud", 
                  "Computing Number of Intersecting Points", "First Object Merging", "Additional Object Mergings",
                  "Finding Candidate Trees", "Computing Number of Intersecting Points for Trees", "Merging Trees"]

        x = np.arange(len(times_per_view))
        bottom = np.zeros(len(times_per_view))

        plt.figure(figsize=(14, 6))
        for i in range(len(labels)):
            plt.bar(x, full_times[:, i], bottom=bottom, label=labels[i])
            bottom += full_times[:, i]  # Update the bottom to stack the next segment

        plt.xlabel("Iterations: Images")
        plt.ylabel("Time")
        plt.title("Run Times for Each Iteration")
        plt.legend()
        plt.savefig(osp.join(out_path, "times_over_time.png"))
        plt.show()

        # Calculate the mean of each substep
        mean_values = np.mean(full_times, axis=0)
        vars = np.var(full_times, axis=0)

        # Create the stacked bar plot for mean values
        x = [0]  # Single bar at position 0
        bottom = 0  # Initialize the bottom of the bar
        x_offset = np.linspace(-0.1, 0.1, len(mean_values))

        plt.figure(figsize=(10, 8))

        for i in range(len(mean_values)):
            plt.bar(x, mean_values[i], bottom=bottom, label=labels[i])
            y_center = bottom + mean_values[i]
            plt.errorbar([x[0] + x_offset[i]], y_center, yerr=vars[i], fmt='none', ecolor='black', capsize=5)
            bottom += mean_values[i]  # Update the bottom for stacking

        # Customize plot
        plt.xlabel("Mean")
        plt.ylabel("Time")
        plt.title("Mean Time of Each Substep")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(osp.join(out_path, "times_means.png"))
        plt.show()
    

if __name__ == "__main__":
    main()