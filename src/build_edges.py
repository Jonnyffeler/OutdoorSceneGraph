import numpy as np
import joblib
import os.path as osp
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import argparse

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from utils import utils_edges as utils

def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to the root folder')
    parser.add_argument('--scene', type=str, help='scene to be run')
   
    args = parser.parse_args()
    return parser, args

def main():
    print("\n##############")
    print("Building Edges")
    print("##############\n")

    parser, args = parse_args()
    scene_name = args.scene
    root_path = args.path
    out_path = osp.join(root_path, "out", scene_name)

    ### loading roots
    roots = joblib.load(osp.join(out_path, f"{scene_name}_roots_described.joblib"))

    boxes = []
    for root in roots:
        boxes.append(np.column_stack(root.object['bounding_box']))
    boxes = np.array(boxes).squeeze(1)

    all_objects = list()
    for root in roots:
        all_objects += root.get_all_children()

    print("Building edges...")
    ### creating edges
    # edges between roots (using MST)
    scene_graph = utils.build_scene_graph(boxes, iou_threshold=0.0)
    edges = list(scene_graph.edges)
    id_root_edges = len(edges)

    # edges between objects and their parent
    edges += utils.get_object_edges(roots, all_objects)

    print("Done. Building hyperedges...")

    ### creating hyperedges between roots
    sentence_model =  SentenceTransformer("all-MiniLM-L6-v2")
    node_embeddings, hyperedges, hyperedge_attrs, _ = utils.produce_hyperedges(roots, sentence_model)

    print("Done. Saving scene graph.")

    # saving node embeddings in the objects (used for SGAligner)
    for i, root in enumerate(roots):
        root.object['node_embedding'] = node_embeddings[i]
    joblib.dump(roots, osp.join(out_path, f"{scene_name}_roots_described.joblib"))

    ### saving scene graph edges
    scenegraph_edges = {"edges": edges, "hyperedges": hyperedges, "hyperedge_attrs": hyperedge_attrs, "id_root_edges": id_root_edges}

    with open(osp.join(out_path, f"{scene_name}_edges.pkl"), "wb") as f:
        pickle.dump(scenegraph_edges, f)

if __name__ == "__main__":
    main()
