from utils.visualizer import *
from utils.structures import *
from utils.data_loader import *

file_name = '007420'
data_source = 'velodyne'
bin_path = 'data/{}/{}.bin'.format(data_source, file_name)
txt_path = 'data/label_2/%s.txt' %file_name
calib_path = 'data/calib/%s.txt' %file_name
out_dir = './mesh/%s/' %data_source


def main():
    points=read_bin_velodyne(bin_path)
    points[..., 0] *= -1

    gt_bboxes = read_txt_label(txt_path, calib_path)['gt_bboxes_3d'].tensor
    gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
    gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2

    show_result(points, gt_bboxes, None, out_dir, file_name)
    print('Done with {} {}'.format(data_source, file_name))


if __name__=="__main__":
    main()









