import os,json
import glob
import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map 
import numpy as np
from shutil import copyfile
from plyfile import PlyData

# try:
#     from sets import Set
# except ImportError:
#     Set = set

LABEL_FILE_NAME_RAW = 'labels.instances.annotated.v2.ply'
LABEL_FILE_NAME = 'labels.instances.align.annotated.v2.ply'
    
helpmsg = 'Generate labels.instances.align.annotated.v2.ply from labels.instances.annotated.v2.ply'
parser = argparse.ArgumentParser(description=helpmsg,formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c','--config',default='./configs/config_default.yaml',required=False)
parser.add_argument('--thread', type=int, default=0, help='The number of threads to be used.')
parser.add_argument('--overwrite', action='store_true', help='overwrite or not.')
args = parser.parse_args()

path_3rscan_data = '/media/T7/3Rscan/scenes/'
cfg_path_split = '/media/T7/3Rscan/files/'

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def get_all_scan_id_splits(path:str):
    files = glob.glob(os.path.join(path,'*_scans.txt'))
    splits = {}
    for file in files:
        name = os.path.basename(file).split('.')[0]
        splits[name] = read_txt_to_list(file)
    return splits

def read_all_scan_ids(path:str):
    # get all splits
    splits = get_all_scan_id_splits(path)
    scan_ids = []
    for v in splits.values():
        scan_ids += v
    # train_ids = read_txt_to_list(os.path.join(define.PATH_FILE,'train_scans.txt'))
    # val_ids = read_txt_to_list(os.path.join(define.PATH_FILE,'validation_scans.txt'))
    # test_ids = read_txt_to_list(os.path.join(define.PATH_FILE,'test_scans.txt'))
    return scan_ids

def resave_ply(filename_in, filename_out, matrix):
    """ Reads a PLY file from disk.
    Args:
    filename: string
    
    Returns: np.array, np.array, np.array
    """
    file = open(filename_in, 'rb')
    plydata = PlyData.read(file)
    points = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).transpose()
    points4f = np.insert(points, 3, values=1, axis=1)
    points = points4f * matrix
    
    plydata['vertex']['x'] = np.asarray(points[:,0]).flatten()
    plydata['vertex']['y'] = np.asarray(points[:,1]).flatten()
    plydata['vertex']['z'] = np.asarray(points[:,2]).flatten()
    
    plydata.write(filename_out)
    
def read_transform_matrix():
    rescan2ref = {}
    with open(os.path.join(path_3rscan_data,"3RScan.json"), "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            for scans in scene["scans"]:
                if "transform" in scans:
                    rescan2ref[scans["reference"]] = np.matrix(scans["transform"]).reshape(4,4)
    return rescan2ref

        
def process(scan_id):
    file_in = os.path.join(path_3rscan_data, scan_id, LABEL_FILE_NAME_RAW)
    file_out = os.path.join(path_3rscan_data, scan_id, LABEL_FILE_NAME)                
    if os.path.isfile(file_out):
        if not args.overwrite:
            return 
    if scan_id in rescan2ref:
        print("changing")
        resave_ply(file_in,file_out,rescan2ref[scan_id])
    else:
        print("none")
        copyfile(file_in, file_out)

if __name__ == "__main__": 
    '''read all scan ids'''
    scan_ids = sorted(read_all_scan_ids(cfg_path_split))
    rescan2ref = read_transform_matrix()

    print(rescan2ref)
    
    if args.thread > 0:
        process_map(process, scan_ids, max_workers=args.thread, chunksize=1 )
    else:
        pbar = tqdm(scan_ids)
        for scan_id in pbar:
            pbar.set_description(f"process scan {scan_id}")
            process(scan_id)
            