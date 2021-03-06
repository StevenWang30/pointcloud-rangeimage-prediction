import open3d as o3d
import os
import numpy as np
import IPython
import math
from pointcloud_to_rangeimage import *

lidar_angular_xy_range_ = 360
max_lidar_angular_z_ = 2
min_lidar_angular_z_ = -24.5
range_x_ = 16
range_y_ = 1000
nearest_bound_ = 0.5
furthest_bound_ = 120
if_show_ground_ = True


def transform_point_cloud_to_range_image_txt_file(KITTI_raw_data_path, save_folder):
    files = os.listdir(KITTI_raw_data_path)
    # files.sort(key=lambda x: int(x.split('.')[0]))
    for seq in files:
        pc_path = os.path.join(KITTI_raw_data_path, seq, "velodyne_points/data")
        pcd_files = os.listdir(pc_path)

        for pc in pcd_files:
            if pc.split('.')[-1] != 'pcd':
                continue
            file_name = pc.split('.')[0]
            pointcloud_path = os.path.join(pc_path, pc)
            pc_cur = o3d.io.read_point_cloud(pointcloud_path)
            pointcloud = np.asarray(pc_cur.points)
            range_image_array = pointcloud_to_rangeimage(pointcloud, lidar_angular_xy_range_, max_lidar_angular_z_,
                                                         min_lidar_angular_z_, range_x_, range_y_)

            range_image_array = remove_balck_line_and_remote_points(range_image_array)

            folder = os.path.join(KITTI_raw_data_path, seq, save_folder)
            if not os.path.isdir(folder): os.makedirs(folder)
            save_name = os.path.join(KITTI_raw_data_path, seq, save_folder, file_name+'.txt')

            np.savetxt(save_name, range_image_array, fmt='%.6f')
            print('save range image txt to ', save_name)


def transform_point_cloud_txt_file_to_range_image_txt_file(KITTI_raw_data_path, save_folder):
    pc_path = os.path.join(KITTI_raw_data_path, "velodyne_points", "data")
    pcd_files = os.listdir(pc_path)

    for pc in pcd_files:
        if pc.split('.')[-1] != 'txt':
            continue
        pointcloud_path = os.path.join(pc_path, pc)
        # pc_cur = o3d.io.read_point_cloud(pointcloud_path)
        # pointcloud = np.asarray(pc_cur.points)
        pointcloud = np.loadtxt(pointcloud_path)
        pointcloud = pointcloud[:, :3]
        range_image_array = pointcloud_to_rangeimage(pointcloud, lidar_angular_xy_range_, max_lidar_angular_z_,
                                                     min_lidar_angular_z_, range_x_, range_y_)

        range_image_array = remove_balck_line_and_remote_points(range_image_array)

        folder = os.path.join(KITTI_raw_data_path, save_folder)
        if not os.path.isdir(folder): os.makedirs(folder)
        save_name = os.path.join(KITTI_raw_data_path, save_folder, pc)

        np.savetxt(save_name, range_image_array, fmt='%.6f')
        print('save range image txt to ', save_name)


def save_range_image_txt_and_source_to_npy(seq_names, save_path):

    data = np.zeros((0, range_x_, range_y_, 1))
    source = []

    for i in range(len(seq_names)):
        root = os.path.join(seq_names[i], 'range_image_data')
        root = seq_names[i]
        print('Sequence: ', seq_names[i])
        rangeimage_txts = os.listdir(root)
        rangeimage_txts.sort(key=lambda x: int(x.split('.')[0]))
        for f in rangeimage_txts:
            if f.split('.')[-1] != 'txt':
                continue
            print('compose file:', os.path.join(root, f))
            range_image_data = np.loadtxt(os.path.join(root, f))
            range_image_data = np.expand_dims(range_image_data, -1)
            data = np.append(data, [range_image_data], axis=0)
            source.append(seq_names[i])
    # IPython.embed()
    save_dict = {}
    save_dict['range_image'] = data
    save_dict['source'] = source
    np.save(save_path, save_dict)
    print('save npy file to ', save_path)


if __name__ == '__main__':
    # KITTI_raw_data_path = '/data/KITTI_rawdata'
    # save_folder = 'range_image_data'
    # transform_point_cloud_to_range_image_txt_file(KITTI_raw_data_path, save_folder)


    # generate training_list and val and test

    # save_folder = '/data/KITTI_rangeimage_predict'
    # training_seq_names = ['/data/KITTI_rawdata/2011_09_26_drive_0009_extract',
    #                       '/data/KITTI_rawdata/2011_09_26_drive_0011_extract',
    #                       '/data/KITTI_rawdata/2011_09_26_drive_0013_extract',
    #                       '/data/KITTI_rawdata/2011_09_26_drive_0014_extract',
    #                       '/data/KITTI_rawdata/2011_09_26_drive_0017_extract',
    #                       '/data/KITTI_rawdata/2011_09_26_drive_0048_extract']
    # save_path = os.path.join(save_folder, 'training_data.npy')
    # save_range_image_txt_and_source_to_npy(training_seq_names, save_path)



    # val_seq_names = ['/data/KITTI_rawdata/2011_09_26_drive_0113_extract']
    # save_path = os.path.join(save_folder, 'validation_data.npy')
    # save_range_image_txt_and_source_to_npy(val_seq_names, save_path)

    # training_seq_names = [
    #     '/data/KITTI_rangeimage_predict/draw_pic_data/raw_data_train/Campus/2011_09_28_drive_0016_extract']
    # save_folder = 'range_image_data'
    # transform_point_cloud_txt_file_to_range_image_txt_file(training_seq_names[0], save_folder)
    # save_path = '/data/KITTI_rangeimage_predict/draw_pic_data/raw_data_train/Campus/Campus.npy'
    # save_range_image_txt_and_source_to_npy(training_seq_names, save_path)

    # training_seq_names = ['/data/KITTI_rangeimage_predict/draw_pic_data/draw_pic/road']
    # save_folder = 'range_image_data'
    # # transform_point_cloud_txt_file_to_range_image_txt_file(training_seq_names[0], save_folder)
    # save_path = '/data/KITTI_rangeimage_predict/draw_pic_data/draw_pic/road_draw.npy'
    # save_range_image_txt_and_source_to_npy(training_seq_names, save_path)

    # training_seq_names = ['/data/KITTI_rangeimage_predict/draw_pic_data/draw_pic/data_new/road']
    # save_folder = 'range_image_data'
    # # transform_point_cloud_txt_file_to_range_image_txt_file(training_seq_names[0], save_folder)
    # save_path = '/data/KITTI_rangeimage_predict/draw_pic_data/draw_pic/data_new/road.npy'
    # save_range_image_txt_and_source_to_npy(training_seq_names, save_path)

    # # ACMMM data
    # seq_name = ['/data/KITTI_rangeimage_predict/ACMMM_data/txt_data/campus',
    #             '/data/KITTI_rangeimage_predict/ACMMM_data/txt_data/ciy',
    #             '/data/KITTI_rangeimage_predict/ACMMM_data/txt_data/person',
    #             '/data/KITTI_rangeimage_predict/ACMMM_data/txt_data/residential']

    # # TU velodyne data
    # seq_name = ['/data/rangeimage_prediction/rangeimage_txt_file/Parking_lot_30',
    #             '/data/rangeimage_prediction/rangeimage_txt_file/Residential_area_30',
    #             '/data/rangeimage_prediction/rangeimage_txt_file/Urban_30']

    # autoware 32E data
    # seq_name = ['/data/rangeimage_prediction_32E/rangeimage_txt_file/autoware_32e']

    # kaist VLP-16 data
    seq_name = ['/data/rangeimage_prediction_VLP16/rangeimage_txt_file/outdoor']

    for i in range(len(seq_name)):
        save_path = seq_name[i] + '.npy'
        save_range_image_txt_and_source_to_npy([seq_name[i]], save_path)





