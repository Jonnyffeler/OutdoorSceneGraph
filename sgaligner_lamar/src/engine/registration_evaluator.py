import sys
import argparse
import numpy as np
import pygcransac
import abc
import gc

import torch
import torch.nn as nn

from utils.point_cloud import apply_transform, get_nearest_neighbor
from utils import torch_util, registration
from GeoTransformer.config import make_cfg
from GeoTransformer.model import create_model
from GeoTransformer.geotransformer.utils.data import registration_collate_fn_stack_mode
import open3d as o3d

class RegistrationEvaluator(abc.ABC):
    def __init__(self, device,  cfg, snapshot, logger, visualise_registration=True):
        self.snapshot = snapshot
        self.device = device
        self.logger = logger
        self.visualise_registration = visualise_registration
        
        # Load Registration Model
        self.cfg = make_cfg()
        self.model = create_model(self.cfg).to(self.device)
        self.model.eval()
        self.load_snapshot(self.snapshot)

        # Params
        self.num_p2p_corrs = cfg.reg_model.num_p2p_corrs
        self.neighbor_limits = cfg.reg_model.neighbor_limits
        self.ransac_threshold = cfg.reg_model.ransac_threshold
        self.ransac_min_iters = cfg.reg_model.ransac_min_iters
        self.ransac_max_iters = cfg.reg_model.ransac_max_iters
        self.ransac_use_sprt = cfg.reg_model.ransac_use_sprt
        self.inlier_ratio_thresh = cfg.reg_model.inlier_ratio_thresh
        self.rmse_thresh = cfg.reg_model.rmse_thresh
        self.min_object_points = 50

    def load_snapshot(self, snapshot):
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=self.device)
        assert 'model' in state_dict, 'No model can be loaded.'
        self.model.load_state_dict(state_dict['model'], strict=True)
        self.logger.info('Registration Model has been loaded.')
    
    def evaluate_registration(self, src_points, ref_points, raw_points, est_transform, gt_transform, 
                              src_corr_points, ref_corr_points, gt_src_corr_points, gt_ref_corr_points):
        print("CD gt: ", registration.compute_modified_chamfer_distance(src_points, ref_points, raw_points, gt_transform, gt_transform))
        print("CD est: ", registration.compute_modified_chamfer_distance(src_points, ref_points, raw_points, est_transform, gt_transform))
        chamfer_distance = registration.compute_modified_chamfer_distance(src_points, ref_points, raw_points, est_transform, gt_transform)
        inlier_ratio = registration.compute_inlier_ratio(ref_corr_points, src_corr_points, gt_transform)
        rre, rte = registration.compute_registration_error(gt_transform, est_transform) 
        registration_rmse = registration.compute_registration_rmse(gt_ref_corr_points, gt_src_corr_points, est_transform)
        fmr = float(inlier_ratio >= self.inlier_ratio_thresh)
        accepted = float(registration_rmse < self.rmse_thresh)
        print("recall: ", registration_rmse, self.rmse_thresh)

        return chamfer_distance, inlier_ratio, rre, rte, accepted, fmr
    
    def perform_registration(self, src_points, ref_points, gt_transform, device='cuda'):
        npoint = 10000  # Total number of points to process
        batch_size = 500  # Number of points to process in each batch

        # Downsample if necessary
        if src_points.shape[0] > npoint:
            indices = np.random.choice(src_points.shape[0], npoint, replace=False)
            src_points = src_points[indices]
        
        if ref_points.shape[0] > npoint:
            indices = np.random.choice(ref_points.shape[0], npoint, replace=False)
            ref_points = ref_points[indices]
        
        # Initialize lists to store results from each batch
        all_output_dicts = []
        batch_nr = 0
        # Split the point clouds into batches
        for i in range(0, npoint, batch_size):
            if i >= src_points.shape[0] or i >= ref_points.shape[0]:break
            batch_nr+=1
            src_batch = src_points[i:i + batch_size]
            ref_batch = ref_points[i:i + batch_size]

            src_feats = np.ones_like(src_batch[:, :1])
            ref_feats = np.ones_like(ref_batch[:, :1])

            data_dict = {
                "ref_points": ref_batch.astype(np.float32),
                "src_points": src_batch.astype(np.float32),
                "ref_feats": ref_feats.astype(np.float32),
                "src_feats": src_feats.astype(np.float32),
                "transform": gt_transform.astype(np.float32)
            }

            with torch.no_grad():
                data_dict = registration_collate_fn_stack_mode([data_dict], 
                                self.cfg.backbone.num_stages, self.cfg.backbone.init_voxel_size, 
                                self.cfg.backbone.init_radius, self.neighbor_limits)
                # Move data to GPU
                data_dict = torch_util.to_cuda(data_dict)
                try: 
                    output_dict = self.model(data_dict)
                except:
                    return None
                output_dict = torch_util.release_cuda(output_dict)
                all_output_dicts.append(output_dict)

            del data_dict, output_dict
            torch.cuda.empty_cache()
            gc.collect()

        # Combine the results from all batches
        combined_output_dict = {}
        combined_output_dict['estimated_transform'] = np.mean(np.stack([output_dict['estimated_transform'] for output_dict in all_output_dicts]), axis=0)
        for key in ['ref_corr_points', 'src_corr_points', 'corr_scores']: #all_output_dicts[0].keys():
            combined_output_dict[key] = np.concatenate([output_dict[key] for output_dict in all_output_dicts], axis=0)
        return combined_output_dict

    def perform_registration_old(self, src_points, ref_points, gt_transform):
        npoint = 500 #500
        if src_points.shape[0] > npoint: 
            indices = np.random.choice(src_points.shape[0], npoint, replace=False)
            src_points = src_points[indices]
        
        if ref_points.shape[0] > npoint: 
            indices = np.random.choice(ref_points.shape[0], npoint, replace=False)
            ref_points = ref_points[indices]
            
        src_feats = np.ones_like(src_points[:, :1])
        ref_feats = np.ones_like(ref_points[:, :1])

        data_dict = {
            "ref_points": ref_points.astype(np.float32),
            "src_points": src_points.astype(np.float32),
            "ref_feats": ref_feats.astype(np.float32),
            "src_feats": src_feats.astype(np.float32),
            "transform" : gt_transform.astype(np.float32)
        }

        with torch.no_grad():
            data_dict = registration_collate_fn_stack_mode([data_dict], 
                            self.cfg.backbone.num_stages, self.cfg.backbone.init_voxel_size, 
                            self.cfg.backbone.init_radius, self.neighbor_limits)
            # output dict
            data_dict = torch_util.to_cuda(data_dict)
            output_dict = self.model(data_dict)
        output_dict = torch_util.release_cuda(output_dict)
        return output_dict

    def run_normal_registration(self, reg_data_dict, evaluate_registration=True):
        src_points = reg_data_dict['src_points']
        ref_points = reg_data_dict['ref_points']
        raw_points = reg_data_dict['raw_points'] if 'raw_points' in reg_data_dict else None
        gt_transform = reg_data_dict['gt_transform'] if 'gt_transform' in reg_data_dict else None
        
        gt_src_corr_points = reg_data_dict['gt_src_corr_points'] if 'gt_src_corr_points' in reg_data_dict else None
        gt_ref_corr_points = reg_data_dict['gt_ref_corr_points'] if 'gt_ref_corr_points' in reg_data_dict else None
        output_dict = self.perform_registration(src_points, ref_points, gt_transform)
        if output_dict is None: return None

        est_transform = output_dict["estimated_transform"]
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']

        corr_scores =  output_dict['corr_scores']
        mean_corr_score = np.mean(corr_scores)

        if evaluate_registration:
            chamfer_distance, inlier_ratio, rre, rte, recall, fmr = self.evaluate_registration(src_points, ref_points, raw_points, 
                                                                                               est_transform, gt_transform, 
                                                                                               src_corr_points, ref_corr_points, 
                                                                                               gt_src_corr_points, gt_ref_corr_points)
            # if recall == 0.0: return None
            return {
                'CD' : chamfer_distance,
                'IR' : inlier_ratio,
                'RRE' : rre,
                'RTE' : rte,
                'recall' : recall,
                'FMR' : fmr
            }
        
        else:
            return None#est_transform, mean_corr_score
    
    def run_aligner_registration(self, reg_data_dict, evaluate_registration=True):
        node_corrs = reg_data_dict['node_corrs']
        src_points = reg_data_dict['src_points']
        ref_points = reg_data_dict['ref_points']
        raw_points = reg_data_dict['raw_points'] if 'raw_points' in reg_data_dict else None

        src_plydata = reg_data_dict['src_plydata'] 
        ref_plydata = reg_data_dict['ref_plydata']

        gt_transform = reg_data_dict['gt_transform']
        gt_src_corr_points = reg_data_dict['gt_src_corr_points'] if 'gt_src_corr_points' in reg_data_dict else None
        gt_ref_corr_points = reg_data_dict['gt_ref_corr_points'] if 'gt_ref_corr_points' in reg_data_dict else None

        point_corrs = {'src' : [], 'ref' : [], 'scores' : []}

        for node_corr in node_corrs:
            node_points_src = src_points[src_plydata[node_corr[0]]]
            node_points_ref = ref_points[ref_plydata[node_corr[1]]]

            if node_points_src.shape[0] < self.min_object_points or node_points_ref.shape[0] < self.min_object_points: continue
            output_dict = self.perform_registration(node_points_src, node_points_ref, gt_transform)
            if output_dict is None: continue
            
            output_dict = torch_util.release_cuda(output_dict)
            ref_corr_points = output_dict['ref_corr_points']
            src_corr_points = output_dict['src_corr_points']
            corr_scores = output_dict['corr_scores']
            
            if corr_scores.shape[0] > self.num_p2p_corrs // len(node_corrs):
                sel_indices = np.argsort(-corr_scores)[: self.num_p2p_corrs // len(node_corrs)]
                ref_corr_points = ref_corr_points[sel_indices]
                src_corr_points = src_corr_points[sel_indices]
                corr_scores = corr_scores[sel_indices]
            
            point_corrs['src'].append(src_corr_points)
            point_corrs['ref'].append(ref_corr_points)
            point_corrs['scores'].append(corr_scores)
        
        if len(point_corrs['src']) == 0 or len(point_corrs['ref']) == 0: return None
        
        point_corrs['src'] = np.concatenate(point_corrs['src'])
        point_corrs['ref'] = np.concatenate(point_corrs['ref'])
        point_corrs['scores'] = np.concatenate(point_corrs['scores'])

        corrs_ransac = np.concatenate([point_corrs['src'], point_corrs['ref']], axis=1)
        # corrs_ransac = corrs_ransac[np.where(point_corrs['scores'] > 0.5)]
        
        min_coordinates = np.min(corrs_ransac, axis=0)
        transformed_corrs_ransac = corrs_ransac - min_coordinates
        
        est_transform, _ = pygcransac.findRigidTransform(np.ascontiguousarray(transformed_corrs_ransac), probabilities = [], 
                                                        threshold = self.ransac_threshold, neighborhood_size = 4, sampler = 1, 
                                                        min_iters = self.ransac_min_iters, max_iters = self.ransac_max_iters, 
                                                        spatial_coherence_weight = 0.0, 
                                                        use_space_partitioning = not self.ransac_use_sprt, neighborhood = 0, conf = 0.999, 
                                                        use_sprt = self.ransac_use_sprt)
        
        T1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [-min_coordinates[0], -min_coordinates[1], -min_coordinates[2], 1]])
        T2inv = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [min_coordinates[3], min_coordinates[4], min_coordinates[5], 1]])
        
        if est_transform is None: return None
        
        est_transform = T1 @ est_transform @ T2inv
        est_transform = est_transform.T

        return est_transform
        
        if not evaluate_registration: return est_transform

        chamfer_distance, inlier_ratio, rre, rte, recall, fmr = self.evaluate_registration(src_points, ref_points, raw_points, 
                                                                                            est_transform, gt_transform, 
                                                                                            corrs_ransac[:, :3], corrs_ransac[:, 3:], 
                                                                                            gt_src_corr_points, gt_ref_corr_points)

        return {
            'CD' : chamfer_distance,
            'IR' : inlier_ratio,
            'RRE' : rre,
            'RTE' : rte,
            'recall' : recall,
            'FMR' : fmr
        }
    
    def preprocess_point_cloud(self, pcd, voxel_size, max_nr_points):
        if len(pcd.points) > max_nr_points:
            pcd_down = pcd.voxel_down_sample(voxel_size)
        else:
            pcd_down = pcd
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh
    
    def execute_fast_global_registration(self, source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        option = o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold)
        try:
            result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, option)
        except:
            return None
        return result

    
    def perform_registration_o3d(self, src_points, ref_points, approx_nr_points=50000):
        voxel_size = 0.6

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(src_points)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(ref_points)

        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size, approx_nr_points)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size, approx_nr_points)

        result_fast = self.execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        if result_fast is None: return None
        return result_fast.transformation
    

    def run_baseline_registration(self, reg_data_dict):
        src_points = reg_data_dict['src_points']
        ref_points = reg_data_dict['ref_points']
        gt_transform = reg_data_dict['gt_transform'] if 'gt_transform' in reg_data_dict else None
        
        normal_reg_transform = self.perform_registration_o3d(src_points, ref_points, approx_nr_points=50000)

        return normal_reg_transform
    
    def create_point_correspondences_from_transform(self, src_points, estimated_transform, num_correspondences=10000):
        # Option 1: Use a subset of source points and transform them
        if len(src_points) > num_correspondences:
            # Randomly sample from source points
            idx = np.random.choice(len(src_points), num_correspondences, replace=False)
            sampled_src = src_points[idx]
        else:
            sampled_src = src_points
        
        # Make homogeneous coordinates
        src_homogeneous = np.hstack([sampled_src, np.ones((len(sampled_src), 1))])
        
        # Transform source points to reference frame
        transformed_src = (estimated_transform @ src_homogeneous.T).T

        transformed_src = transformed_src[:, :3] / transformed_src[:, 3][:, np.newaxis]
        
        # For simplicity, we'll use the direct transformation (Option 1)
        # and assign high confidence scores
        output_dict = {
            'ref_corr_points': transformed_src,  # These are the corresponding points in ref frame
            'src_corr_points': sampled_src,      # Original source points 
            'corr_scores': np.ones(len(sampled_src)),  # High confidence since they match our transform
            'estimated_transform': estimated_transform
        }
    
        return output_dict


    def run_aligner_registration_o3d(self, reg_data_dict):
        node_corrs = reg_data_dict['node_corrs']
        src_points = reg_data_dict['src_points']
        ref_points = reg_data_dict['ref_points']
        raw_points = reg_data_dict['raw_points'] if 'raw_points' in reg_data_dict else None

        src_plydata = reg_data_dict['src_plydata'] 
        ref_plydata = reg_data_dict['ref_plydata']

        gt_transform = reg_data_dict['gt_transform']

        point_corrs = {'src' : [], 'ref' : [], 'scores' : []}

        for node_corr in node_corrs:
            node_points_src = src_points[src_plydata[node_corr[0]]]
            node_points_ref = ref_points[ref_plydata[node_corr[1]]]

            if node_points_src.shape[0] < self.min_object_points or node_points_ref.shape[0] < self.min_object_points: continue
            est_trans_o3d = self.perform_registration_o3d(node_points_src, node_points_ref)
            if est_trans_o3d is None: continue
            output_dict = self.create_point_correspondences_from_transform(node_points_src, est_trans_o3d)
            
            ref_corr_points = output_dict['ref_corr_points']
            src_corr_points = output_dict['src_corr_points']
            corr_scores = output_dict['corr_scores']
            
            if corr_scores.shape[0] > self.num_p2p_corrs // len(node_corrs):
                sel_indices = np.argsort(-corr_scores)[: self.num_p2p_corrs // len(node_corrs)]
                ref_corr_points = ref_corr_points[sel_indices]
                src_corr_points = src_corr_points[sel_indices]
                corr_scores = corr_scores[sel_indices]
            
            point_corrs['src'].append(src_corr_points)
            point_corrs['ref'].append(ref_corr_points)
            point_corrs['scores'].append(corr_scores)
        
        if len(point_corrs['src']) == 0 or len(point_corrs['ref']) == 0: return None
        
        point_corrs['src'] = np.concatenate(point_corrs['src'])
        point_corrs['ref'] = np.concatenate(point_corrs['ref'])
        point_corrs['scores'] = np.concatenate(point_corrs['scores'])

        corrs_ransac = np.concatenate([point_corrs['src'], point_corrs['ref']], axis=1)
        # corrs_ransac = corrs_ransac[np.where(point_corrs['scores'] > 0.5)]
        
        min_coordinates = np.min(corrs_ransac, axis=0)
        transformed_corrs_ransac = corrs_ransac - min_coordinates
        
        est_transform, _ = pygcransac.findRigidTransform(np.ascontiguousarray(transformed_corrs_ransac), probabilities = [], 
                                                        threshold = self.ransac_threshold, neighborhood_size = 4, sampler = 1, 
                                                        min_iters = self.ransac_min_iters, max_iters = self.ransac_max_iters, 
                                                        spatial_coherence_weight = 0.0, 
                                                        use_space_partitioning = not self.ransac_use_sprt, neighborhood = 0, conf = 0.999, 
                                                        use_sprt = self.ransac_use_sprt)
        
        T1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [-min_coordinates[0], -min_coordinates[1], -min_coordinates[2], 1]])
        T2inv = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [min_coordinates[3], min_coordinates[4], min_coordinates[5], 1]])
        
        if est_transform is None: return None
        
        est_transform = T1 @ est_transform @ T2inv
        est_transform = est_transform.T
        
        return est_transform

        

    def run_registration(self, reg_data_dict):
        import time

        gt_transform = reg_data_dict['gt_transform'] if 'gt_transform' in reg_data_dict else None

        normal_reg_results = {}
        start = time.time()
        normal_reg_transform = self.run_baseline_registration(reg_data_dict)
        normal_reg_results['est_trans'] = normal_reg_transform
        normal_reg_results['time'] = time.time()-start
        if normal_reg_transform is None:
            normal_reg_results['RRE'], normal_reg_results['RTE'] = 999999, 999999
        else:
            normal_reg_results['RRE'], normal_reg_results['RTE'] = registration.compute_registration_error(gt_transform, normal_reg_transform) 
        
        aligner_reg_results = {}
        start = time.time()
        aligner_reg_transform = self.run_aligner_registration(reg_data_dict)
        aligner_reg_results['est_trans'] = aligner_reg_transform
        aligner_reg_results['time'] = time.time()-start
        if aligner_reg_transform is None:
            aligner_reg_results['RRE'], aligner_reg_results['RTE'] = 999999, 999999
        else:
            aligner_reg_results['RRE'], aligner_reg_results['RTE'] = registration.compute_registration_error(gt_transform, aligner_reg_transform) 

        alignero3d_reg_results = {}
        start = time.time()
        alignero3d_transform = self.run_aligner_registration_o3d(reg_data_dict)
        alignero3d_reg_results['est_trans'] = alignero3d_transform
        alignero3d_reg_results['time'] = time.time()-start
        if alignero3d_transform is None:
            alignero3d_reg_results['RRE'], alignero3d_reg_results['RTE'] = 999999, 999999
        else:
            alignero3d_reg_results['RRE'], alignero3d_reg_results['RTE'] = registration.compute_registration_error(gt_transform, alignero3d_transform) 
        
        return (normal_reg_results, aligner_reg_results, alignero3d_reg_results)
        print("----------normal------------------")
        normal_reg_results_dict = self.run_normal_registration(reg_data_dict)
        print(normal_reg_results_dict)
        if normal_reg_results_dict is None: return None, None
        print("---------aligner----------")
        aligner_reg_results_dict = self.run_aligner_registration(reg_data_dict)
        print(aligner_reg_results_dict)
        # print('\n', aligner_reg_results_dict)
        # print('\n', normal_reg_results_dict)
        # print('====')
        
        return normal_reg_results_dict, aligner_reg_results_dict

