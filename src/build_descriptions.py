import joblib
import numpy as np
import pandas as pd
import time
import json
from openai import OpenAI
import joblib
import argparse
import os.path as osp
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils import utils_descriptions as utils_descriptions
from utils import utils_objects as utils_objects
from utils.ObjectNode import ObjectNode

def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the root folder')
    parser.add_argument('--scene', type=str, help='scene to be run')

    args = parser.parse_args()
    return parser, args

def main():
    print("\n############################")
    print("Building Object Descriptions")
    print("############################\n")
    parser, args = parse_args()
    scene_name = args.scene
    root_path = args.path
    out_path = osp.join(root_path, "out", scene_name)
    data_path = osp.join(root_path, "Lamar", scene_name)


    ### Reading everything
    print("reading in data...")
    start = time.time()
    imagedb = pd.read_csv(osp.join(data_path, "images.txt"), header = 0)
    poses_trajectory = pd.read_csv(osp.join(data_path, "trajectories.txt"), header = 0) 
    camera_intrinsics = np.array(pd.read_csv(osp.join(data_path, "sensors.txt"), header = 1).iloc[0, 4:10])
    im_width, im_height = camera_intrinsics[0], camera_intrinsics[1]
    intrinsic_mat, poses = utils_objects.extr_intrinsics_from_saved(camera_intrinsics, poses_trajectory)
    print("done loading data in: ", time.time()-start)

    roots = joblib.load(osp.join(out_path, f"{scene_name}_roots.joblib"))
    all_objects = list()
    all_objects += roots
    for root in roots:
        all_objects += root.get_all_children()

    ### producing the images with drawn in rectangles and saving them in the out path
    print("producing images with rectangles")
    top_k = 3
    if not os.path.exists(osp.join(out_path, "images_marked")):
        os.makedirs(osp.join(out_path, "images_marked"))
    utils_descriptions.produce_images_with_rectangles(osp.join(out_path, "images_marked"), all_objects, poses, intrinsic_mat, im_width, im_height, imagedb, osp.join(data_path, "raw_data"), scene_name, top_k, plot_images = False)
    
    ### generating the descriptions for all top_k present views 
    object_descriptions = []
    for image_nr in range(top_k):
        # Create image encodings
        file_names = [f"{obj.node_id}_{image_nr}" for obj in all_objects]
        file_names_2 = []

        print(f"encoding images for top_k: {image_nr}...")
        base64_images = []
        for file_name in file_names:
            im_path = osp.join(out_path, "images_marked", f"{file_name}.jpg")
            if osp.exists(im_path):
                file_names_2.append(int(file_name[:-2]))
                base64_images.append(utils_descriptions.encode_image(im_path))
        print("top_k: ", image_nr, "encoded images. Generating descriptions...")
        
        client = OpenAI()
        object_descriptions.append(utils_descriptions.create_OpenAI_descriptions(out_path, client, base64_images, file_names_2, image_nr, scene_name))
        print("top_k: ", image_nr, "generated descriptions")
    
    # main description (first one, present for all objects)
    base_descriptions = object_descriptions[0]

    with open(osp.join(out_path, f"{scene_name}_base_descr_GPT.json"), "w") as f:
        json.dump(base_descriptions, f, indent=4) 

    ### Distilling descriptions into one
    print("Distilling descriptions...")
    object_summaries = utils_descriptions.distill_descriptions(client, object_descriptions)    
    ids_summarized = [obj['obj_id'] for obj in object_summaries]
    for i, obj in enumerate(base_descriptions):
        if obj['obj_id'] in ids_summarized:
            base_descriptions[i] = utils_descriptions.get_descr_from_id(object_summaries, obj['obj_id'])
    
    print("Done. Saving the described objects.")

    # saving final descriptions
    with open(osp.join(out_path, f"{scene_name}_All_GPT_descriptions_combined.json"), "w") as f:
        json.dump(base_descriptions, f, indent=4) 

    ### saving descriptions, labels and attributes in the object nodes
    for object in all_objects:
        descr = utils_descriptions.get_descr_from_id(base_descriptions, object.node_id)
        assert(descr is not None)
        object.object['label'] = descr['label']
        object.object['attributes'] = descr['attributes'] 
        object.object['description'] = descr['description'] 
        object.object['certainty_desc'] = descr['certainty']

    # save final roots:
    joblib.dump(roots, osp.join(out_path, f"{scene_name}_roots_described.joblib"))

if __name__ == "__main__":
    main()