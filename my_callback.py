import keras
import IPython
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from prednet import PredNet
from keras.layers import Input, Dense, Flatten
from keras.models import Model
import numpy as np
import os
import cv2
# import time
# from tifffile import imsave
#
# import scipy.misc
from skimage.measure import compare_ssim


class My_Callback(keras.callbacks.Callback):
    def test_image(self, epoch, validation_data, args):

        nt = args.nt
        batch_size = 1

        # train_model = self.model
        train_model = self.callbacks[5].model
        checkpoint_path = args.ckpt
        # checkpoint_path = "ckpt/cp_back.ckpt"

        train_model.load_weights(checkpoint_path)

        # IPython.embed()

        # Create testing model (to output predictions)
        layer_config = train_model.layers[1].get_config()
        layer_config['output_mode'] = 'prediction'
        data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
        test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
        input_shape = list(train_model.layers[0].batch_input_shape[1:])
        input_shape[0] = nt
        inputs = Input(shape=tuple(input_shape))
        predictions = test_prednet(inputs)
        test_model = Model(inputs=inputs, outputs=predictions)
        X_test = validation_data.create_all()
        X_test_origin = validation_data.create_all_origin()
        # X_test = X_test[0:10]
        # X_test = X_test / args.norm_value
        # IPython.embed()

        # im = X_test[0][0] * args.norm_value
        # RESULTS_SAVE_DIR = "../mid_result_vis/epoch" + str(epoch)
        # cv2.imwrite(RESULTS_SAVE_DIR + "/" + 'plot_abaa.png', im)
        #         IPython.embed()
        X_hat = test_model.predict(X_test, batch_size)
        if data_format == 'channels_first':
            X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
            X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

        X_not_scaled = X_test * args.norm_value
        X_hat_not_scaled = X_hat * args.norm_value

        # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
        mse_model = np.mean((X_not_scaled[:, -1] - X_hat_not_scaled[:, -1]) ** 2)  # look at all timesteps except the first
        mse_prev = np.mean((X_not_scaled[:, -1] - X_not_scaled[:, -2]) ** 2)

        # Compare SSIM
        ssim_model_a = np.zeros(X_not_scaled.shape[0])
        ssim_prev_a = np.zeros(X_not_scaled.shape[0])
        for i in range(X_not_scaled.shape[0]):
            A = X_not_scaled[i, -1, ..., 0]
            B = X_hat_not_scaled[i, -1, ..., 0]
            (score, diff) = compare_ssim(A, B, full=True)
            ssim_model_a[i] = score
            A = X_not_scaled[i, -1, ..., 0]
            B = X_not_scaled[i, -2, ..., 0]
            (score, diff) = compare_ssim(A, B, full=True)
            ssim_prev_a[i] = score
        ssim_model = np.mean(ssim_model_a)
        ssim_prev = np.mean(ssim_prev_a)

        result_save_path = args.mse_result_path

        f = open(result_save_path, 'a+')
        f.write("epoch %d\n" % epoch)
        f.write("Model MSE: %f\n" % mse_model)
        f.write("Previous Frame MSE: %f\n\n\n" % mse_prev)
        f.write("Model SSIM: %f\n" % ssim_model)
        f.write("Previous Frame SSIM: %f\n\n\n" % ssim_prev)
        f.close()

        # Plot some predictions

        # rand_num = int(np.random.random(1) * X_not_scaled.shape[0])
        rand_num = 0



        gs = gridspec.GridSpec(2, nt)
        gs.update(wspace=0., hspace=0.)

        X_not_scaled = X_not_scaled[:, :, :, :, 0]
        X_hat_not_scaled = X_hat_not_scaled[:, :, :, :, 0]
        X_test_origin = X_test_origin[:, :, :, :, 0]

        misc_value = np.max(X_not_scaled) / 255
        im_X = (X_not_scaled[rand_num, -1] / misc_value).astype(np.uint8)
        im_X_hat = (X_hat_not_scaled[rand_num, -1] / misc_value).astype(np.uint8)
        im_dif_hat = np.abs(im_X_hat - im_X)
        im_X_previous = (X_not_scaled[rand_num, -2] / misc_value).astype(np.uint8)
        im_dif_previous = np.abs(im_X_previous - im_X)


        im = im_X
        im = np.concatenate((im, im_X_hat), axis=0).astype(np.uint8)
        im = np.concatenate((im, im_dif_hat), axis=0).astype(np.uint8)
        im = np.concatenate((im, im_X_previous), axis=0).astype(np.uint8)
        im = np.concatenate((im, im_dif_previous), axis=0).astype(np.uint8)

        #
        #
        # for t in range(1, X_not_scaled.shape, 1):
        #     im_test = X_not_scaled[rand_num, t] / misc_value
        #     im_hat = X_hat_not_scaled[rand_num, t] / misc_value
        #     im = np.concatenate((im, im_test), axis=0).astype(np.uint8)
        #     im = np.concatenate((im, im_hat), axis=0).astype(np.uint8)
        #
        #     # IPython.embed()
        #     # plt.subplot(gs[t])
        #     # plt.imshow(X_not_scaled[rand_num, t], interpolation='none')
        #     # plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
        #     #                 labelbottom='off', labelleft='off')
        #     # if t == 0: plt.ylabel('Actual', fontsize=10)
        #     #
        #     # plt.subplot(gs[t + nt])
        #     # plt.imshow(X_hat_not_scaled[rand_num, t], interpolation='none')
        #     # plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
        #     #                 labelbottom='off', labelleft='off')
        #     # if t == 0: plt.ylabel('Predicted', fontsize=10)
        #     # plt.subplot(gs[t])
        #     # plt.imshow(X_not_scaled[rand_num, t])
        #     # if t == 0: plt.ylabel('Actual', fontsize=10)
        #     #
        #     # plt.subplot(gs[t + nt])
        #     # plt.imshow(X_hat_not_scaled[rand_num, t])
        #     # if t == 0: plt.ylabel('Predicted', fontsize=10)
        #
        # # scipy.misc.imsave(RESULTS_SAVE_DIR + "/" + 'plot_' + str(i) + '.png', im)
        fig_save_path = os.path.join(args.fig_result_dir, "epoch_"+str(epoch)+".png")
        cv2.imwrite(fig_save_path, im)

        for i in range(nt):
            # IPython.embed()
            np.savetxt(os.path.join(args.fig_result_dir, "X_hat_frame" + str(rand_num) + "_time" + str(i) + ".txt"), X_hat_not_scaled[rand_num, i], fmt='%f')
            np.savetxt(os.path.join(args.fig_result_dir, "X_gt_frame" + str(rand_num) + "_time" + str(i) + ".txt"), X_not_scaled[rand_num, i], fmt='%f')
            misc_value = np.max(X_not_scaled) / 255
            im_X = (X_not_scaled[rand_num, i] / misc_value).astype(np.uint8)
            fig_save_path = os.path.join(args.fig_result_dir, "X_gt_frame" + str(rand_num) + "_time" + str(i) + ".png")
            cv2.imwrite(fig_save_path, im_X)
            im_X_hat = (X_hat_not_scaled[rand_num, i] / misc_value).astype(np.uint8)
            fig_save_path = os.path.join(args.fig_result_dir, "X_hat_frame" + str(rand_num) + "_time" + str(i) + ".png")
            cv2.imwrite(fig_save_path, im_X_hat)

        # for i in range(3):
        #     # save groundtruth image
        #     # imsave(fig_save_path.replace('.png', '_groundtruth.tif'), X_not_scaled[i, -1])
        #     fig_save_path = os.path.join(args.fig_result_dir, "my_rangeimage.png")
        #     np.savetxt(fig_save_path.replace('.png', '_frame' + str(i) + '_groundtruth.txt'), X_not_scaled[i, -1], fmt='%f')
        #     # save predict image
        #     # imsave(fig_save_path.replace('.png', '_predict.tif'), X_hat_not_scaled[i, -1])
        #     np.savetxt(fig_save_path.replace('.png', '_frame' + str(i) + '_predict.txt'), X_hat_not_scaled[i, -1], fmt='%f')
        #     # save origin image
        #     # imsave(fig_save_path.replace('.png', '_previous.tif'), X_not_scaled[i, -2])
        #     np.savetxt(fig_save_path.replace('.png', '_frame' + str(i) + '_previous.txt'), X_not_scaled[i, -2], fmt='%f')

        # IPython.embed()

        # im_dif = X_hat_not_scaled[rand_num, 9] / misc_value
        # im_dif_t = X_not_scaled[rand_num, 9] / misc_value
        # im_dif = np.concatenate((im_dif, im_dif_t), axis=0).astype(np.uint8)
        # im_dif_t = np.fabs(X_not_scaled[rand_num, 9] - X_not_scaled[rand_num, 8]) / misc_value
        # im_dif = np.concatenate((im_dif, im_dif_t), axis=0).astype(np.uint8)
        # for t in reversed(list(range(1, 10))):
        #     im_dif_t = np.fabs(X_hat_not_scaled[rand_num, 9] - X_not_scaled[rand_num, t]) / misc_value
        #     im_dif = np.concatenate((im_dif, im_dif_t), axis=0).astype(np.uint8)
        # cv2.imwrite(RESULTS_SAVE_DIR + "/" + 'plot_dif_' + str(rand_num) + '.png', im_dif)

        # plt.clf()

    # def on_epoch_begin(self, epoch, validation_data_my):
    #     IPython.embed()
    #     pass

    def my_on_epoch_end(self, epoch, validation_data, args):
        print("on epoch end: process val data and save: ...")
        # IPython.embed()
        # if(epoch % 10 == 0):
        My_Callback.test_image(self, epoch, validation_data, args)
        # self.validation_data.shape