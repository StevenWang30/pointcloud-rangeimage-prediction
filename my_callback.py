import keras
import IPython
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from prednet import PredNet
from keras.layers import Input, Dense, Flatten
from keras.models import Model
import numpy as np
import os
import cv2

import scipy.misc

class My_Callback(keras.callbacks.Callback):
    def test_image(self, epoch, validation_data, args):

        nt = 10
        batch_size = 1

        # train_model = self.model
        train_model = self.callbacks[5].model
        checkpoint_path = "ckpt/cp.ckpt"
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
        X_test = X_test[0:20]
        # X_test = X_test / args.norm_value

        # im = X_test[0][0] * args.norm_value
        # RESULTS_SAVE_DIR = "../mid_result_vis/epoch" + str(epoch)
        # cv2.imwrite(RESULTS_SAVE_DIR + "/" + 'plot_abaa.png', im)
        #         IPython.embed()
        X_hat = test_model.predict(X_test, batch_size)
        if data_format == 'channels_first':
            X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
            X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

        # Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
        mse_model = np.mean((X_test[:, 1:] - X_hat[:, 1:]) ** 2)  # look at all timesteps except the first
        mse_prev = np.mean((X_test[:, :-1] - X_test[:, 1:]) ** 2)
        RESULTS_SAVE_DIR = "../mid_result_vis/epoch" + str(epoch)
        if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)

        f = open(RESULTS_SAVE_DIR + "/" + 'prediction_scores.txt', 'w')
        f.write("Model MSE: %f\n" % mse_model)
        f.write("Previous Frame MSE: %f" % mse_prev)
        f.close()

        # Plot some predictions


        ###
        nt = 2
        ###
        nt = 2
        ###
        # aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
        # plt.figure(figsize=(nt, 2 * aspect_ratio))
        gs = gridspec.GridSpec(2, nt)
        gs.update(wspace=0., hspace=0.)

        X_test = X_test[:, :, :, :, 0]
        X_hat = X_hat[:, :, :, :, 0]

        # IPython.embed()
        # *255
        X_test = X_test * args.norm_value
        X_hat = X_hat * args.norm_value

        # IPython.embed()

        for i in range(5):
            #for t in range(nt):
            # IPython.embed()
            im_dif = X_hat[i, 9] - X_test[i, 9]
            misc_value = np.max(X_test) / 255
            im_dif = im_dif / misc_value
            im = im_dif
            for t in range(1, 10, 1):
                im_test = X_test[i, t] / misc_value
                im_hat = X_hat[i, t] / misc_value
                im = np.concatenate((im, im_test), axis=0).astype(np.uint8)
                im = np.concatenate((im, im_hat), axis=0).astype(np.uint8)

                # IPython.embed()
                # plt.subplot(gs[t])
                # plt.imshow(X_test[i, t], interpolation='none')
                # plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                #                 labelbottom='off', labelleft='off')
                # if t == 0: plt.ylabel('Actual', fontsize=10)
                #
                # plt.subplot(gs[t + nt])
                # plt.imshow(X_hat[i, t], interpolation='none')
                # plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off',
                #                 labelbottom='off', labelleft='off')
                # if t == 0: plt.ylabel('Predicted', fontsize=10)
                # plt.subplot(gs[t])
                # plt.imshow(X_test[i, t])
                # if t == 0: plt.ylabel('Actual', fontsize=10)
                #
                # plt.subplot(gs[t + nt])
                # plt.imshow(X_hat[i, t])
                # if t == 0: plt.ylabel('Predicted', fontsize=10)

            # scipy.misc.imsave(RESULTS_SAVE_DIR + "/" + 'plot_' + str(i) + '.png', im)
            cv2.imwrite(RESULTS_SAVE_DIR + "/" + 'plot_' + str(i) + '.png', im)

            im_dif = X_hat[i, 9] / misc_value
            im_dif_t = X_test[i, 9] / misc_value
            im_dif = np.concatenate((im_dif, im_dif_t), axis=0).astype(np.uint8)
            im_dif_t = np.fabs(X_test[i, 9] - X_test[i, 8]) / misc_value
            im_dif = np.concatenate((im_dif, im_dif_t), axis=0).astype(np.uint8)
            for t in reversed(list(range(1, 10))):
                im_dif_t = np.fabs(X_hat[i, 9] - X_test[i, t]) / misc_value
                im_dif = np.concatenate((im_dif, im_dif_t), axis=0).astype(np.uint8)
            cv2.imwrite(RESULTS_SAVE_DIR + "/" + 'plot_dif_' + str(i) + '.png', im_dif)

            plt.clf()

    # def on_epoch_begin(self, epoch, validation_data_my):
    #     IPython.embed()
    #     pass

    def my_on_epoch_end(self, epoch, validation_data, args):
        print("on epoch end: process val data and save: ...")
        # IPython.embed()
        # if(epoch % 10 == 0):
        My_Callback.test_image(self, epoch, validation_data, args)
        # self.validation_data.shape