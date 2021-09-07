from utils.visualizer import *
from utils.structures import *
from utils.data_loader import *
import glob

# file_name = '007420'
# data_source = 'velodyne'
# bin_path = '../fusion/data/{}/{}.bin'.format(data_source, file_name)
# txt_path = 'data/label_2/%s.txt' %file_name
# calib_path = 'data/calib/%s.txt' %file_name
# out_dir = '../fusion/data/mesh/%s/' %data_source


def main():
    root_path = input('Input kitti root path:')
    filelist = glob.glob(root_path+'/calib/*')
    out_dir = input('Output velodyne bin file path:')

    for rec in filelist:
        file_name = rec.split('/')[-1].split('.')[0]
        # points=read_bin_velodyne(rec)
        # points[..., 0] *= -1

        txt_path = root_path + '/label_2/' + file_name + '.txt'
        calib_path = rec

        gt_bboxes = read_txt_label(txt_path, calib_path)['gt_bboxes_3d'].tensor
        gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH)
        gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2

        show_result(None, gt_bboxes, None, out_dir, file_name)
        # show_result(points, gt_bboxes, None, out_dir, file_name)
        # print('Done with {} {}'.format(data_source, file_name))


if __name__=="__main__":
    main()









