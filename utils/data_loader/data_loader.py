import copy
import struct
import numpy as np
from ..data_converter import *
from ..structures import *

def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[1],point[0],point[2]])
    return np.asarray(pc_list,dtype=np.float32)


def remove_dontcare(ann_info):
    """Remove annotations that do not need to be cared.

    Args:
        ann_info (dict): Dict of annotation infos. The ``'DontCare'``
            annotations will be removed according to ann_file['name'].

    Returns:
        dict: Annotations after filtering.
    """
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(ann_info['name']) if x != 'DontCare'
    ]
    for key in ann_info.keys():
        img_filtered_annotations[key] = (
            ann_info[key][relevant_annotation_indices])
    return img_filtered_annotations

def drop_arrays_by_name(gt_names, used_classes):
    """Drop irrelevant ground truths by name.

    Args:
        gt_names (list[str]): Names of ground truths.
        used_classes (list[str]): Classes of interest.

    Returns:
        np.ndarray: Indices of ground truths that will be dropped.
    """
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds
    
def read_txt_label(txtpath, calibpath):
    """Get annotation info according to the given index.

    Args:
        index (int): Index of the annotation data to get.

    Returns:
        dict: annotation information consists of the following keys:

            - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                3D ground truth bboxes.
            - gt_labels_3d (np.ndarray): Labels of ground truths.
            - gt_bboxes (np.ndarray): 2D ground truth bboxes.
            - gt_labels (np.ndarray): Labels of ground truths.
            - gt_names (list[str]): Class names of ground truths.
    """
    # Use index to get the annos, thus the evalhook could also use this api
    info = get_kitti_image_info(txtpath,calibpath)

    rect = info['calib']['R0_rect'].astype(np.float32)
    Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

    annos = info['annos']
    # we need other objects to avoid collision when sample
    annos = remove_dontcare(annos)
    loc = annos['location']
    dims = annos['dimensions']
    rots = annos['rotation_y']
    gt_names = annos['name']
    gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                    axis=1).astype(np.float32)

    # convert gt_bboxes_3d to velodyne coordinates
    gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
        Box3DMode.LIDAR, np.linalg.inv(rect @ Trv2c))
    gt_bboxes = annos['bbox']

    selected = drop_arrays_by_name(gt_names, ['DontCare'])
    gt_bboxes = gt_bboxes[selected].astype('float32')
    gt_names = gt_names[selected]

    CLASSES = ('car', 'pedestrian', 'cyclist')
    gt_labels = []
    for cat in gt_names:
        if cat in CLASSES:
            gt_labels.append(CLASSES.index(cat))
        else:
            gt_labels.append(-1)
    gt_labels = np.array(gt_labels).astype(np.int64)
    gt_labels_3d = copy.deepcopy(gt_labels)

    anns_results = dict(
        gt_bboxes_3d=gt_bboxes_3d,
        gt_labels_3d=gt_labels_3d,
        bboxes=gt_bboxes,
        labels=gt_labels,
        gt_names=gt_names)
    return anns_results
