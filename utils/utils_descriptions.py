import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utils.ObjectNode import ObjectNode
import time
import base64
import json
from tqdm import tqdm
from pydantic import BaseModel
import os.path as osp

def downsample_pointcloud_voxel(points, voxel_size=0.3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downsampled_pcd.points)

def zoom_to_rectangle(img, x_from, y_from, x_to, y_to, min_ratio=0.5, line_thickness=10, min_margin=5):
    h, w, _ = img.shape  # Original image size

    # Ensure bounding box stays within visible margins
    x_from = max(min_margin, x_from)
    y_from = max(min_margin, y_from)
    x_to = min(w - min_margin, x_to)
    y_to = min(h - min_margin, y_to)

    bbox_width = x_to - x_from
    bbox_height = y_to - y_from

    # Check if the rectangle already covers at least `min_ratio` in either dimension
    if bbox_width >= min_ratio * w or bbox_height >= min_ratio * h:
        cv2.rectangle(img, (x_from, y_from), (x_to, y_to), color=(255, 0, 0), thickness=line_thickness)
        return img

    # Compute scale factors to make the rectangle 50% of width or height
    scale_x = (min_ratio * w) / bbox_width 
    scale_y = (min_ratio * h) / bbox_height
    scale_factor = min(scale_x, scale_y)  # Use the smaller scaling to prevent over-zooming

    new_width = int(bbox_width * scale_factor)
    new_height = int(bbox_height * scale_factor)

    # Find center of bbox
    cx, cy = (x_from + x_to) // 2, (y_from + y_to) // 2

    # Compute new crop boundaries
    x1 = max(0, cx - new_width // 2)
    x2 = min(w, cx + new_width // 2)
    y1 = max(0, cy - new_height // 2)
    y2 = min(h, cy + new_height // 2)

    cropped_img = img[y1:y2, x1:x2]
    zoomed_img = cv2.resize(cropped_img, (w, h), interpolation=cv2.INTER_LINEAR)
    thickness_scale = h / (y2 - y1)  # Ratio of original height to cropped height
    new_thickness = max(1, int(line_thickness * thickness_scale))

    # Scale bounding box to new image size
    new_x_from = max(min_margin, int((x_from - x1) * (w / (x2 - x1))))
    new_y_from = max(min_margin, int((y_from - y1) * (h / (y2 - y1))))
    new_x_to = min(w - min_margin, int((x_to - x1) * (w / (x2 - x1))))
    new_y_to = min(h - min_margin, int((y_to - y1) * (h / (y2 - y1))))

    cv2.rectangle(zoomed_img, (new_x_from, new_y_from), (new_x_to, new_y_to), color=(255, 0, 0), thickness=new_thickness)

    return zoomed_img

def select_unique_pose_indices(poses, position_thresh=0.1, angle_thresh=10):
    selected_indices = [0]  # Always keep the first pose

    for i in range(1, len(poses)):
        origin_i = poses[i, :3, 3]
        direction_i = poses[i, :3, 2]  # Forward direction

        unique = True 

        for idx in selected_indices:
            origin_s = poses[idx, :3, 3]
            direction_s = poses[idx, :3, 2]  # Forward direction

            ### Computing Euclidean distance between origins
            pos_distance = np.linalg.norm(origin_i - origin_s)

            ### Computing angle difference using dot product
            cos_angle = np.clip(np.dot(direction_i, direction_s), -1, 1)
            angle_distance = np.arccos(cos_angle) * (180 / np.pi)

            ### If too similar, discard this pose
            if pos_distance < position_thresh and angle_distance < angle_thresh:
                unique = False
                break

        if unique:
            selected_indices.append(i)

    return selected_indices

def produce_images_with_rectangles(out_path, roots, poses, intrinsic_mat, im_width, im_height, imagedb, raw_data_path, scene_name, top_k = 3, plot_images = False):
    poses_np = np.linalg.inv(np.array(poses))
    poses_in_np = np.array(poses)
    for object in tqdm(roots):
        start = time.time()
        root_obj = object.object
        scaling_factor = (len(root_obj['pcd']) / 10000) ** (1/3) 
        
        if len(root_obj['pcd']) < 10000:
            obj_pcd = root_obj['pcd']
        else:
            obj_pcd = downsample_pointcloud_voxel(root_obj['pcd'], scaling_factor*0.2)
        imseg_ids = np.array(root_obj['imseg_id']) # first col: image_nr, second col: mask_nr
        imseg_area = np.array(root_obj['imseg_area']).flatten()
        imseg_bbox_percs = np.array(root_obj['bbox_perc']).flatten()
        imseg_bboxes = root_obj['bbox_mask']

        unique_im_nrs, imseg_indices = np.unique(imseg_ids[:,0], return_inverse=True)
        imseg_poses = poses_np[unique_im_nrs] # get all image poses

        # for each pose we compute how many points of the objects pcd are in frame (do we see only a part of an object?)
        obj_pcd_hom = np.column_stack([obj_pcd, np.ones(len(obj_pcd))])
        in_view_pts = np.dot(imseg_poses, obj_pcd_hom.T)

        in_view_pts = in_view_pts[:, 0:3, :] # get rid of 4th column (ones)
        in_view_pts = np.dot(intrinsic_mat, in_view_pts)

        in_view_bool = in_view_pts[2] > 0 # z value >0 -> in front of the camera
        in_view_pts = in_view_pts/in_view_pts[2, :, :] # divide by w
        in_view_pts = in_view_pts[0:2] # get rid of w

        nr_in_view_pts = (in_view_bool & (in_view_pts[0]<im_width) & (in_view_pts[1]<im_height) & (in_view_pts>0).all(axis=0)).sum(axis=1)
        nr_in_view_pts = nr_in_view_pts[imseg_indices]
        
        criterion = nr_in_view_pts*imseg_area*imseg_bbox_percs
        top_k_ids = np.argsort(-criterion)[:80]

        selected_ids = select_unique_pose_indices(poses_in_np[imseg_ids[top_k_ids, 0]], position_thresh=7, angle_thresh=45)

        top_k_ids = top_k_ids[selected_ids][:top_k]

        for i, k in enumerate(top_k_ids):
            path_img_file = imagedb[" image_path"][imseg_ids[k, 0]]
            path_img = os.path.join(raw_data_path,path_img_file[1:])
            img = np.copy(cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB))
            bbox = imseg_bboxes[k]
            x_from, y_from, _, _ = bbox
            x_to, y_to = x_from + bbox[2], y_from + bbox[3]
            x_from = max(x_from-7, 4)
            y_from = max(y_from-7, 4)
            x_to = min(x_to+7, img.shape[1]-4)
            y_to = min(y_to+7, img.shape[0]-4)

            img_marked = zoom_to_rectangle(img, x_from, y_from, x_to, y_to, min_ratio=0.3, line_thickness=10, min_margin=10)
            img_marked = cv2.cvtColor(img_marked, cv2.COLOR_BGR2RGB)

            cv2.imwrite(osp.join(out_path, f"{object.node_id}_{i}.jpg"), img_marked) 

            if plot_images:
                plt.imshow(img_marked)
                plt.show()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
class Object_descriptions(BaseModel):
    label: str
    description: str
    attributes: list[str]
    certainty: float

def create_OpenAI_descriptions(out_path, client, base64_images, file_names_2, image_nr, scene_name):
    save_path = osp.join(out_path, f"{scene_name}_GPT_descriptions_image{image_nr}.json")
    object_descriptions = []
    for i, base64_img in tqdm(enumerate(base64_images), total=len(base64_images), desc="Processing"):
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {
                "role": "system",
                "content": "You are an expert image captioning model that analyzes the main and most present object of an image that is within a red bounding box. You will output your analysis in JSON format."
                },
                {
                "role": "user",
                "content": [
                {
                "type": "text",
                "text": "I give you an image with a red bounding box in it. You will \
                    focus on the red bounding box and analyze what the main object is that is most present in the bounding box (i.e. takes up most of the bounding boxes' space),\
                    you will describe it and give the output as a JSON object. \n \
                    First, look at the red bounding box, and find a single word that best describes the main, largest object\
                    within the red bounding box. Only consider the largest object that is most present and takes up most space within the red bounding box. \
                    Use the parts of the image surrounding the red bounding box only as a clue of what the marked object could be but do not describe objects outside the bounding box.\
                    Output the single word in the JSON object with the key 'label'. \n \
                    Then find a short, precise sentence that further\
                    describes key attributes of the marked object in the red bounding box, output the sentence in 'description'. Don't state anything about the bounding box. Find attributes or descriptions that seem fitting and important to the marked object.\
                    Next, take the key attributes that you found before and output them in 'attributes' as single words in a list instead of a sentence.\
                    Finally, output 'certainty' with a value between 0 and 1 depending on how sure you are that you were able to recognize the correct object in the red bounding box."
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_img}"
                },
                }
            ]
                }
            ],
            response_format=Object_descriptions
        )

        json_file = json.loads(response.choices[0].message.content)
        json_file['obj_id'] = file_names_2[i]
        object_descriptions.append(json_file)

        if i%50==0:
            with open(save_path, "w") as f:
                json.dump(object_descriptions, f, indent=4) 
                
    with open(save_path, "w") as f:
        json.dump(object_descriptions, f, indent=4) 

    return object_descriptions

def get_descr_from_id(descrs, id):
    for desc in descrs:
        if int(desc['obj_id'])==int(id):
            return desc
    return None

def distill_descriptions(client, object_descriptions):
    object_summaries = []
    object_descriptions_0 = object_descriptions[0]
    object_descriptions_1 = object_descriptions[1]

    for file_1 in tqdm(object_descriptions_1):
        id_obj = file_1['obj_id']
        file_0 = get_descr_from_id(object_descriptions_0, id_obj)
        assert(file_1 is not None)
        assert(file_0 is not None)
        
        files_list = f"File 1:\n \
                    {file_0}\n \n \
                    File 2: \n \
                    {file_1}"
        
        for i, obj_desc in tqdm(enumerate(object_descriptions[2:])):
            file = get_descr_from_id(obj_desc, id_obj)
            if file is not None:
                files_list += f"File {i+3}: \n \
                        {file}"
        
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                    "role": "system",
                    "content": "You are an expert in combinding information. You will get two descriptions of objects and you will combinde them into one. You will output your analysis in JSON format."
                    },
                    {
                    "role": "user",
                    "content": [
                    {
                    "type": "text",
                    "text": f"After this prompt I will give you several object descriptions in JSON format. The descriptions will be of the same objects but seen from different perspectives, so they might be different.\
                        You will distill the information of all descriptions into one single object description and output it in a JSON file.\n\
                        The JSON files of the descriptions contain the following keys: 'label', which is the objects category. 'description', which is a short sentence \
                        further describing the object. 'attributes', which are single words in a list of attributes of the object. 'certainty', which signifies the confidence of the description.\
                        and 'obj_id' which you will ignore.\n\
                        First, look at all keys from all descriptions. Take the certainties of the descriptions into account and using only the given descriptions estimate what the object most likely looks like in reality. For example if you have 3 descriptions and 2 of them are very similar and the other is different, treat the other as an outlier and only focus on the two similar ones.\
                        IMPORTANT: Do not invent a new description! Also don't just concatenate the answers!\n\
                        Summarize and distill the knowledge of the descriptions into one. Then output your distillation of the object into a JSON file, using the same keys.\
                        Estimate how certain you are about the summary and output that in the key 'certainty'. Here are the JSON files: \n \
                        {files_list}"
                    }
                    ]
                    }
                ],
                response_format=Object_descriptions,
            )

        json_file = json.loads(completion.choices[0].message.content)
        json_file['obj_id'] = file_1['obj_id']
        object_summaries.append(json_file)

    return object_summaries