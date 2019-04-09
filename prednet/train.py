import os
import numpy as np
import argparse

from six.moves import cPickle

from keras import backend
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from prednet import PredNet
from data_utils_save import SequenceGenerator
from data_utils import SequenceGenerator as SequenceGenerator_data
# from kitti_settings import *
import hickle as hkl
import data_process

import IPython


def main(args):
    save_model = True  # if weights will be saved
    weights_file = os.path.join(args.weight_dir, 'prednet_kitti_weights.hdf5')  # where weights will be saved
    json_file = os.path.join(args.weight_dir, 'prednet_kitti_model.json')

    # Data files
    train_file = os.path.join(args.data_dir, 'X_train.hkl')
    train_sources = os.path.join(args.data_dir, 'sources_train.hkl')
    val_file = os.path.join(args.data_dir, 'X_val.hkl')
    val_sources = os.path.join(args.data_dir, 'sources_val.hkl')

    # train_data_list = "/tensorflow/code/rangeImage_data/train_data_list.txt"
    # val_data_list = "/tensorflow/code/rangeImage_data/val_data_list.txt"
    train_data_list = "/data/rangeimage_train_list.txt"
    val_data_list = "/data/rangeimage_val_list.txt"

    # IPython.embed()
    train_data, train_sources_data = data_process.load(args, train_data_list, args.train_data_bin_file)
    val_data, val_sources_data = data_process.load(args, val_data_list, args.val_data_bin_file)

    # # read hkl file test
    # import hickle as hkl
    # tr_f = hkl.load(train_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
    # tr_s = hkl.load(train_sources)
    # va_f = hkl.load(val_file)
    # va_s = hkl.load(val_sources)
    # import IPython
    # IPython.embed()

    # Training parameters


    # Model parameters
    n_channels, im_height, im_width = (args.channel, args.height, args.width)
    input_shape = (n_channels, im_height, im_width) if backend.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)
    stack_sizes = (n_channels, 48, 96, 192)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (3, 3, 3)
    Ahat_filt_sizes = (3, 3, 3, 3)
    R_filt_sizes = (3, 3, 3, 3)
    layer_loss_weights = np.array([1., 0., 0., 0.])  # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
    layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
    nt = 5  # number of timesteps used for sequences in training
    time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  # equally weight all timesteps except the first
    time_loss_weights[0] = 0


    prednet = PredNet(stack_sizes, R_stack_sizes,A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                      output_mode='error', return_sequences=True)

    inputs = Input(shape=(nt,) + input_shape)
    # IPython.embed()
    errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
    errors_by_time = TimeDistributed(Dense(1, trainable=False), weights=[layer_loss_weights, np.zeros(1)], trainable=False)(errors)  # calculate weighted error by layer
    errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
    final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
    model = Model(inputs=inputs, outputs=final_errors)
    model.compile(loss='mean_absolute_error', optimizer='adam')

    # train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=args.batch_size, shuffle=True)
    # val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=args.batch_size, N_seq=args.N_seq_val)
    train_generator = SequenceGenerator_data(train_data, train_sources_data, nt, batch_size=args.batch_size, shuffle=True)
    val_generator = SequenceGenerator_data(val_data, val_sources_data, nt, batch_size=args.batch_size, N_seq=args.N_seq_val)

    lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
    callbacks = [LearningRateScheduler(lr_schedule)]
    if save_model:
        if not os.path.exists(args.weight_dir): os.mkdir(args.weight_dir)
        callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

    history = model.fit_generator(train_generator, args.samples_per_epoch / args.batch_size, args.nb_epoch, callbacks=callbacks,
                    validation_data=val_generator, validation_steps=args.N_seq_val / args.batch_size)

    if save_model:
        json_string = model.to_json()
        with open(json_file, "w") as f:
            f.write(json_string)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path related arguments
    parser.add_argument('--data_dir',
                        default='../../kitti_data/')
    parser.add_argument('--data_docker_root_dir',
                        default='/data')
    parser.add_argument('--rangeimage_data_dir',
                        default='../../rangeImage_data/')
    parser.add_argument('--train_data_bin_file',
                        default='train_bin.npz')
    parser.add_argument('--val_data_bin_file',
                        default='val_bin.npz')
    parser.add_argument('--test_data_bin_file',
                        default='test_bin.npz')
    parser.add_argument('--result_dir',
                        default='./kitti_results/')
    parser.add_argument('--weight_dir',
                        default='./model_data_keras2/')

    # Model related arguments
    parser.add_argument('--nb_epoch', default=150,
                        help='nb_epoch')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='input batch size')
    parser.add_argument('--samples_per_epoch', default=500, type=int,
                        help='samples_per_epoch')
    parser.add_argument('--N_seq_val', default=100, type=int,
                        help='number of sequences to use for validation')
    # parser.add_argument('--epoch_iters', default=2000, type=int,
    #                     help='iterations of each epoch (irrelevant to batch size)')
    # parser.add_argument('--optim', default='SGD', help='optimizer')
    # parser.add_argument('--lr_encoder', default=2e-2, type=float, help='LR')
    # parser.add_argument('--lr_decoder', default=2e-2, type=float, help='LR')
    # parser.add_argument('--lr_pow', default=0.9, type=float,
    #                     help='power in poly to drop LR')
    # parser.add_argument('--beta1', default=0.9, type=float,
    #                     help='momentum for sgd, beta1 for adam')
    # parser.add_argument('--weight_decay', default=1e-4, type=float,
    #                     help='weights regularizer')
    # parser.add_argument('--deep_sup_scale', default=0.4, type=float,
    #                     help='the weight of deep supervision loss')
    # parser.add_argument('--fix_bn', action='store_true',
    #                     help='fix bn params')

    # Data related arguments
    parser.add_argument('--rangeimage_size', default=[0,64,2000,1],
                        help='nb_epoch')
    parser.add_argument('--channel', default=1,
                        help='img channel')
    parser.add_argument('--width', default=2000,
                        help='img width')
    parser.add_argument('--height', default=64,
                        help='img height')
    # parser.add_argument('--num_class', default=150, type=int,
    #                     help='number of classes')
    # parser.add_argument('--num_class', default=7, type=int,
    #                     help='number of classes')
    # parser.add_argument('--workers', default=16, type=int,
    #                     help='number of data loading workers')
    # parser.add_argument('--imgSize', default=[300, 375, 450, 525, 600],
    #                     nargs='+', type=int,
    #                     help='input image size of short edge (int or list)')
    # # parser.add_argument('--imgSize', default=[300],
    # #                     nargs='+', type=int,
    # #                     help='input image size of short edge (int or list)')
    # parser.add_argument('--imgMaxSize', default=1000, type=int,
    #                     help='maximum input image size of long edge')
    # parser.add_argument('--padding_constant', default=8, type=int,
    #                     help='maxmimum downsampling rate of the network')
    # parser.add_argument('--segm_downsampling_rate', default=8, type=int,
    #                     help='downsampling rate of the segmentation label')
    # parser.add_argument('--random_flip', default=True, type=bool,
    #                     help='if horizontally flip images when training')

    # Misc arguments
    parser.add_argument('--seed', default=123, type=int,
                        help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    # parser.add_argument('--disp_iter', type=int, default=20,
    #                     help='frequency to display')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))


    args.ckpt = os.path.join(args.ckpt)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    np.random.seed(args.seed)

    main(args)
