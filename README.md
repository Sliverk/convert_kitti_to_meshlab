# Convert KITTI Velodyne Bin to Meshlab Obj

This code is based on MMLab [mmdetection3D](https://github.com/open-mmlab/mmdetection3d).

## Eviroment
Python3.6 (I do not test on other machine)

## Usage
Step 1: Install requirements
```shell
pip install -r requirements.txt
```

Step 2: Change the path in show_kitti_3d_v2.py to your data path.
```python
# Line5 - Line10
file_name = '007420'
data_source = 'velodyne'
bin_path = 'data/{}/{}.bin'.format(data_source, file_name)
txt_path = 'data/label_2/%s.txt' %file_name
calib_path = 'data/calib/%s.txt' %file_name
out_dir = './mesh/%s/' %data_source
```

Step 3: Execute the show_kitti_3d_v2.py.
```shell
python show_kitti_3d_v2.py
```
