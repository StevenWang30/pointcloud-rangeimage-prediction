import os
import numpy as np
import argparse

from six.moves import cPickle

from keras import backend

from keras.layers import Input, Dense, Flatten
from keras import backend as K
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
# from data_utils_save import SequenceGenerator
from data_utils import SequenceGenerator as SequenceGenerator_data
# from kitti_settings import *
import hickle as hkl
import data_process

from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import time

import scipy.misc
from my_callback import My_Callback

# change the model . fit_generator
# from keras.models import Model
from my_training import My_Model as Model

# import tensorflow as tf
#
import IPython



def main(args):
    compare_with_copy = True
    save_model = True  # if weights will be saved
    weights_file = os.path.join(args.model_dir, 'prednet_kitti_weights.hdf5')  # where weights will be saved
    json_file = os.path.join(args.model_dir, 'prednet_kitti_model.json')
    #
    # # Data files
    # train_file = os.path.join(args.data_dir, 'X_train.hkl')
    # train_sources = os.path.join(args.data_dir, 'sources_train.hkl')
    # val_file = os.path.join(args.data_dir, 'X_val.hkl')
    # val_sources = os.path.join(args.data_dir, 'sources_val.hkl')

    # data1
    # train_data_list = "/tensorflow/code/rangeImage_data/train_data_list.txt"
    # val_data_list = "/tensorflow/code/rangeImage_data/val_data_list.txt"


    # data2 from kitti
    # train_data_list = "none"
    # val_data_list = "none"
    # val_data, val_sources_data = data_process.load(args, val_data_list, args.kitti_val_data_bin_file)
    # train_data, train_sources_data = data_process.load(args, train_data_list, args.kitti_train_data_bin_file)


    # Model parameters
    n_channels, im_height, im_width = (args.channel, args.height, args.width)
    input_shape = (n_channels, im_height, im_width) if backend.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
    # stack_sizes = (n_channels, 48, 96, 192)
    stack_sizes = (n_channels, 24, 48, 96)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)
    layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
    # layer_loss_weights = np.array([1., 0.1, 0.1, 0.1])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
    layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
    nt = args.nt  # number of timesteps used for sequences in training
    time_loss_weights = 1./ (nt - 1) * np.ones((nt, 1))  # equally weight all timesteps except the first
    time_loss_weights[0] = 0
    # args.norm_value = np.max(train_data)

    # IPython.embed()
    prednet = PredNet(stack_sizes, R_stack_sizes,A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode='error', return_sequences=True)

    inputs = Input(shape=(nt,) + input_shape)
    # IPython.embed()
    errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)


    # errors = errors * 100
    # IPython.embed()

    errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
    errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
    final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time

    # final_errors = final_errors * 100
    # final_errors = K.dot(final_errors, 100)
    # final_errors = Lambda(lambda x: x * 100)(final_errors)

    model = Model(inputs=inputs, outputs=final_errors)
    model.compile(loss='mean_absolute_error', optimizer='adam')

    # IF LOAD CKPT fILE
    # load old checkpoint weight
    # old_checkpoint_path = '/code/rangeImage_prediction/good_results/model_data_keras2/good_frame5.ckpt'
    # old_checkpoint_path = '/code/rangeImage_prediction/good_results/cp_nt10.ckpt'
    model.load_weights(args.weight_path)

    # train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=args.batch_size, shuffle=True)
    # val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=args.batch_size, N_seq=args.N_seq_val)
    train_generator = SequenceGenerator_data(args, args.training_data, nt, batch_size=args.batch_size, shuffle=True)
    val_generator = SequenceGenerator_data(args, args.validation_data, nt, batch_size=args.batch_size, N_seq=args.N_seq_val)
    # IPython.embed()
    # a = val_generator.create_all()[0][0]
    # a=val_data[0]
    # a[0][0]
    # im = a[:, :, 0]
    # RESULTS_SAVE_DIR = "../mid_result_vis/epoch1"
    # # cv2.imwrite(RESULTS_SAVE_DIR + "/" + 'plot_abaa.png', im)
    # scipy.misc.imsave(RESULTS_SAVE_DIR + "/" + 'plot_abaa.png', im)
    #     IPython.embed()



    lr_schedule = lambda epoch: 0.001 if epoch < 20 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
    callbacks = [LearningRateScheduler(lr_schedule)]
    if save_model:
        if not os.path.exists(args.model_dir): os.mkdir(args.model_dir)
        callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

    callbacks.append(TensorBoard(log_dir=args.log_dir))
    my_callback_object = My_Callback()
    callbacks.append(my_callback_object)

    # save model every 1 epoch
    checkpoint_path = args.ckpt_save_dir
    cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
    callbacks.append(cp_callback)

    history = model.fit_generator(train_generator, args.samples_per_epoch / args.batch_size, args.nb_epoch, args=args,
              callbacks = callbacks, validation_data=val_generator, validation_steps=args.N_seq_val / args.batch_size)

    # if compare_with_copy:
    #     for ii in range(train_data.shape[0] // nt):
    #         i = ii * nt
    #         i_1 = i + 1
    #         E = train_data[i] - train_data[i_1]
    #     ii = train_data.shape[0] // nt

    if save_model:
        json_string = model.to_json()
        with open(json_file, "w") as f:
            f.write(json_string)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path related arguments
    parser.add_argument('--data_dir',
                        default='/data/KITTI_rangeimage_predict')
    # parser.add_argument('--training_data',
    #                     default='/data/KITTI_rangeimage_predict/training_data.npy')
    # parser.add_argument('--validation_data',
    #                     default='/data/KITTI_rangeimage_predict/validation_data.npy')
    # parser.add_argument('--result_dir',
    #                     default='/data/KITTI_rangeimage_predict/results')

    # parser.add_argument('--validation_data',
    #                     default='/data/KITTI_rangeimage_predict/draw_pic_data/raw_data_train/Road/Road.npy')
    # parser.add_argument('--result_dir',
    #                     default='/data/KITTI_rangeimage_predict/draw_pic_data/raw_data_train/Road/results')

    # draw picture sxb data!!!!!!!!!!!
    # parser.add_argument('--training_data',
    #                     default='/data/KITTI_rangeimage_predict/draw_pic_data/draw_pic/data_new/campus.npy')
    # parser.add_argument('--validation_data',
    #                     default='/data/KITTI_rangeimage_predict/draw_pic_data/draw_pic/data_new/campus.npy')
    # parser.add_argument('--result_dir',
    #                     default='/data/KITTI_rangeimage_predict/draw_pic_data/draw_pic/data_new/campus')

    # # video wsk data!!!!
    # parser.add_argument('--training_data',
    #                     default='/data/KITTI_rangeimage_predict/draw_pic_data/raw_data_train/Road/Road.npy')
    # parser.add_argument('--validation_data',
    #                     default='/data/KITTI_rangeimage_predict/draw_pic_data/raw_data_train/Road/Road.npy')
    # parser.add_argument('--result_dir',
    #                     default='/data/KITTI_rangeimage_predict/video_data/road')

    # # ACMMM_data
    # parser.add_argument('--training_data',
    #                     default='/data/KITTI_rangeimage_predict/ACMMM_data/txt_data/campus.npy')
    # parser.add_argument('--validation_data',
    #                     default='/data/KITTI_rangeimage_predict/ACMMM_data/txt_data/campus.npy')
    # parser.add_argument('--result_dir',
    #                     default='/data/KITTI_rangeimage_predict/ACMMM_data/results/campus')

    # TU_velodyne data
    parser.add_argument('--training_data',
                        default='/data/rangeimage_prediction/rangeimage_txt_file/Urban_30.npy')
    parser.add_argument('--validation_data',
                        default='/data/rangeimage_prediction/rangeimage_txt_file/Urban_30.npy')
    parser.add_argument('--result_dir',
                        default='/data/rangeimage_prediction/results/Urban_30')

    parser.add_argument('--model_dir', default='./model_data_keras2/')
    parser.add_argument('--weight_path', default='/code/rangeImage_prediction/good_results/cp_nt10.ckpt')


    # Model related arguments
    parser.add_argument('--nb_epoch', default=80,
                        help='nb_epoch') # default 150
    parser.add_argument('--batch_size', default=3, type=int,
                        help='input batch size')
    parser.add_argument('--samples_per_epoch', default=200, type=int,
                        help='samples_per_epoch') # default 500
    parser.add_argument('--N_seq_val', default=20, type=int,
                        help='number of sequences to use for validation')
    parser.add_argument('--nt', default=10, type=int,
                        help='number of timesteps used for sequences in training')
    parser.add_argument('--norm_value', default=80, type=float,
                        help='value for normalizing the input data into 0-1')
    parser.add_argument('--log_dir', default='./log/train', help='log file for tensorboard')

    # Data related arguments
    # origin myself dataset arguments
    parser.add_argument('--rangeimage_size', default=[0, 64, 2000, 1],
                        help='nb_epoch')
    parser.add_argument('--channel', default=1,
                        help='img channel')
    # parser.add_argument('--width', default=2000,
    #                     help='img width')
    # parser.add_argument('--height', default=64,
    #                     help='img height')

    # Misc arguments
    parser.add_argument('--seed', default=123, type=int,
                        help='manual seed')
    parser.add_argument('--ckpt_save_dir', default="ckpt/cp.ckpt",
                        help='folder to output checkpoints')
    # parser.add_argument('--disp_iter', type=int, default=20,
    #                     help='frequency to display')


    args = parser.parse_args()

    args.width = args.rangeimage_size[2]
    args.height = args.rangeimage_size[1]

    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # if not os.path.isdir(args.ckpt):
    #     os.makedirs(args.ckpt)
    #     print("Make dir to " + args.ckpt)

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
        print("Make dir to " + args.log_dir)
    else:
        filelist = os.listdir(args.log_dir)
        for f in filelist:
            filepath = os.path.join(args.log_dir, f)
            if os.path.isfile(filepath):
                os.remove(filepath)
                print(str(filepath) + " removed!")
        print("remove all old log files")

    if not os.path.exists(args.result_dir): os.mkdir(args.result_dir)
    mse_result_dir = os.path.join(args.result_dir, 'mse_result')
    if not os.path.exists(mse_result_dir): os.mkdir(mse_result_dir)
    args.mse_result_path = os.path.join(mse_result_dir,
                                        'mse_result-' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(
                                            ' ', '-') + '.txt')
    result_save_path = args.mse_result_path

    f = open(result_save_path, 'a+')
    f.write("start\n")
    f.close()
    fig_result_dir = os.path.join(args.result_dir, 'fig_result')
    if not os.path.exists(fig_result_dir): os.mkdir(fig_result_dir)
    args.fig_result_dir = fig_result_dir

    np.random.seed(args.seed)

    main(args)
