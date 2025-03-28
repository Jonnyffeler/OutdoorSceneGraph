import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering, DBSCAN


######### Hyperedges ########
def cluster_sim_matrix(similarity_matrix, threshold):
    distance_matrix = 1 - similarity_matrix  # Convert similarity to distance
    linkage_matrix = linkage(distance_matrix, method='average')
    labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
    return labels 

def labels_to_hyperedges(labels):
    curr_max_hyperedge_idx = 0
    min = np.min(labels)
    if min > 0:
        labels = labels - min # make sure index stars at 0 

    hyperedges = [] 
    for label in np.unique(labels):
        if label > -1:
            ids = np.where(labels==label)[0]
            if len(ids)>1:
                label_vec = np.ones_like(ids)*curr_max_hyperedge_idx
                hyperedges.append(np.array([ids, label_vec]))
                curr_max_hyperedge_idx += 1
    
    if len(hyperedges)==0:
        return [] 

    return np.concatenate(hyperedges, axis=1)

def create_hyperedge_attrs(embeddings, labels):
    hyperedge_attrs = [] 
    for label in np.unique(labels):
        if label > -1:
            ids = np.where(labels==label)[0] 
            if len(ids) > 1:
                hyperedge_attrs.append(np.mean(embeddings[ids], axis=0))
    return np.array(hyperedge_attrs)

def cluster_embeddings_dbscan(embeddings, eps, min_samp):
    dbscan = DBSCAN(eps=eps, min_samples=min_samp, metric='cosine')
    dbscan.fit(embeddings)

    labels = dbscan.labels_
    return labels

def find_object_clusters(obj_centers, eps, min_samp):
    dbscan = DBSCAN(eps=eps, min_samples=min_samp)
    dbscan.fit(obj_centers)

    clusters = dbscan.labels_
    return clusters

def find_neighbors(main_object_id, obj_centers, n_neighbors, radius):
    main_obj = obj_centers[main_object_id]
    distances = np.linalg.norm(obj_centers-main_obj, axis=1)
    ids_within = np.where(distances<radius)[0]
    ids = ids_within[np.argsort(distances[ids_within])[:n_neighbors]]  

    return ids

def produce_hyperedges(objects, model, hyperedge_types_to_use=["full", "neighbors", "clustering"]):
    obj_ids = []
    obj_centers = [] 
    obj_labels = []
    obj_attrs = []
    obj_descriptions = []

    for object in objects:
        obj_ids.append(object.node_id)
        obj_centers.append(object.object['bs_center'])
        obj_labels.append(object.object['label'])
        obj_attrs.append(object.object['attributes'])
        obj_descriptions.append(object.object['description'])
    obj_centers = np.array(obj_centers)
    
    ### producing object encodings
    label_sentences = ["This object is a {}".format(label) for label in obj_labels] 
    object_attributes = model.encode(label_sentences)
    label_enhanced_sentences = ["This object is a {} with the following description: {}".format(label, description) for label, description in zip(obj_labels, obj_descriptions)] 
    object_attributes_enhanced = model.encode(label_enhanced_sentences)


    ### producing different types of hyperedges
    hyperedges = {}
    hyperedge_attrs = {}

    ### full sentence encoding
    if 'full' in hyperedge_types_to_use:
        embeddings = model.encode(obj_descriptions)
        labels = cluster_embeddings_dbscan(embeddings, eps=0.2, min_samp=2)
        assert(len(labels)==len(obj_labels))

        hyperedges['full'] = labels_to_hyperedges(labels)
        hyperedge_attrs['full'] = create_hyperedge_attrs(embeddings, labels)

    ### neighbors
    if 'neighbors' in hyperedge_types_to_use:
        hyperedges_neigh = []
        hyperedge_attrs_neigh = []
        main_object_ids = list(range(len(obj_labels)))
        curr_max_hyperedge_idx = 0
        for main_object_id in main_object_ids:
            neighbors_in_sphere = find_neighbors(main_object_id, obj_centers, n_neighbors=3, radius=8)
            if len(neighbors_in_sphere) > 1:
                new_hyperedge = np.ones_like(neighbors_in_sphere)*curr_max_hyperedge_idx
                new_hyperedge = np.row_stack([neighbors_in_sphere, new_hyperedge])
                new_hyperedge_attr = np.mean(object_attributes_enhanced[neighbors_in_sphere], axis=0).reshape(1, -1)
                hyperedges_neigh.append(new_hyperedge)
                hyperedge_attrs_neigh.append(new_hyperedge_attr)
                curr_max_hyperedge_idx += 1

        if len(hyperedges_neigh)==0:
            hyperedges['neighbors'] = []
            hyperedge_attrs['neighbors'] = []
        else:
            hyperedges['neighbors'] = np.column_stack(hyperedges_neigh)
            hyperedge_attrs['neighbors'] = np.row_stack(hyperedge_attrs_neigh)

    ### clustering
    if 'clustering' in hyperedge_types_to_use:
        hyperedges_clust = []
        hyperedge_attrs_clust = []
        object_clusters = find_object_clusters(obj_centers, eps=8, min_samp=3)
        curr_max_hyperedge_idx = 0
        for cluster in np.unique(object_clusters):
            if cluster > -1:
                ids = np.where(object_clusters==cluster)[0]
                new_hyperedge = np.ones_like(ids)*curr_max_hyperedge_idx
                new_hyperedge = np.row_stack([ids, new_hyperedge])
                new_hyperedge_attr = np.mean(object_attributes[ids], axis=0).reshape(1, -1)
                hyperedges_clust.append(new_hyperedge)
                hyperedge_attrs_clust.append(new_hyperedge_attr)
                curr_max_hyperedge_idx += 1
        
        if len(hyperedges_clust)==0:
            hyperedges['clustering'] = []
            hyperedge_attrs['clustering'] = []
        else:
            hyperedges['clustering'] = np.column_stack(hyperedges_clust)
            hyperedge_attrs['clustering'] = np.row_stack(hyperedge_attrs_clust)
    
    return object_attributes_enhanced, hyperedges, hyperedge_attrs, embeddings

######### Edges ########
def compute_intersection_volume(boxes):
    N = len(boxes)
    intersection = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            min_corner = np.maximum(boxes[i, :3], boxes[j, :3])
            max_corner = np.minimum(boxes[i, 3:], boxes[j, 3:])
            inter_dim = np.maximum(max_corner - min_corner, 0)
            intersection[i, j] = intersection[j, i] = np.prod(inter_dim)
    
    return intersection

def compute_iou_matrix(boxes):
    N = len(boxes)
    
    volumes = np.prod(boxes[:, 3:] - boxes[:, :3], axis=1)
    intersection = compute_intersection_volume(boxes)
    union = volumes[:, None] + volumes[None, :] - intersection
    iou_matrix = np.where(union > 0, intersection / union, 0)
    np.fill_diagonal(iou_matrix, 0)  # Remove self-connections

    return iou_matrix

def build_iou_graph(iou_matrix, threshold=0.1):
    G = nx.Graph()
    N = len(iou_matrix)
    
    # Add edges where IoU > threshold
    for i in range(N):
        for j in range(i + 1, N):
            if iou_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=iou_matrix[i, j])
    
    return G

def build_scene_graph(boxes, iou_threshold=0.1):
    iou_matrix = compute_iou_matrix(boxes)
    G = build_iou_graph(iou_matrix, threshold=iou_threshold)
    
    # Extract MST per connected component
    mst_graph = nx.Graph()
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        mst = nx.minimum_spanning_tree(subgraph, weight='weight')
        mst_graph.add_edges_from(mst.edges(data=True))
    
    return mst_graph

def create_object_edge(obj, ids):
    assert(obj.parent is not None)
    fro = np.where(obj.node_id == ids)[0][0]
    to = np.where(obj.parent.node_id == ids)[0][0]

    return [(fro, to)]

def get_object_edges(roots, all_objects):
    edges = []
    ids = [obj.node_id for obj in roots]
    ids += [obj.node_id for obj in all_objects]
    for obj in all_objects:
        edges += create_object_edge(obj, np.array(ids))
    return edges