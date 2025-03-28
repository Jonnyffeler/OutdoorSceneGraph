import json
import joblib
import os.path as osp
import argparse

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils import utils_subscenes

def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the root folder')
    parser.add_argument('--scenes', nargs="+", default=["LIN", "CAB", "HGE"], help='scene to be run')
    
    args = parser.parse_args()
    return parser, args

def main():
    parser, args = parse_args()
    scene_names = args.scenes
    root_path = args.path
    out_path = osp.join(root_path, "out")

    print("\n############################################################")
    print(f"Creating Bag of Words Attributes for {scene_names}")
    print("############################################################\n")
    
    ### Loading roots
    print("Loading objects... ", end="", flush=True)
    roots = [joblib.load(osp.join(out_path, scene_name, f"{scene_name}_roots_described.joblib")) for scene_name in scene_names]
    print("Done. Creating bag of words")

    ### Creating bag of words
    bow = set()
    for root in roots:
        for sub_root in root:
            for attr in sub_root.object['attributes']:
                bow.add(utils_subscenes.normalize_attr(attr))
            all_kids = sub_root.get_all_children()
            for kid in all_kids:
                for attr in kid.object['attributes']:
                    bow.add(utils_subscenes.normalize_attr(attr))
    bow_attrs = list(bow)

    print("Done. Saving the bag of words.")

    ### Saving
    name = "_".join(scene_names)
    for scene_name in scene_names:
        with open(osp.join(out_path, scene_name, f"bow_attrs_{name}.json"), "w") as f:
            json.dump(bow_attrs, f, indent=4)

if __name__ == "__main__":
    main()