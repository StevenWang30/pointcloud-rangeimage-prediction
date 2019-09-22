import hickle as hkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator
import IPython
import data_process

# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, args, data_path, nt,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 data_format=K.image_data_format()):
        # self.data = data  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        dict_load = np.load(data_path, allow_pickle=True).item()
        data = dict_load['range_image']
        source = dict_load['source']
        self.data = data
        self.sources = source
        self.norm_scaler = args.norm_value

        self.args = args
        #
        # IPython.embed()
        # self.sources = source # source for each image so when creating sequences can assure that consecutive frames are from same video

        # IPython.embed()
        self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode

        if self.data_format == 'channels_first':
            self.data = np.transpose(self.data, (0, 3, 1, 2))
        self.im_shape = self.data[0].shape

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.data.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.data.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = next(self.index_generator), self.batch_size
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        # IPython.embed()
        for i, l in enumerate(index_array[0]):
            # print(i)
            # print(idx)
            # print(idx+self.nt)
            # print(self.data.shape)
            # IPython.embed()
            idx = self.possible_starts[l]
            batch_x[i] = self.preprocess(self.data[idx:idx+self.nt])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        # return X.astype(np.float32) / 255
        # return X.astype(np.float32) / self.args.norm_value
        ret = X.astype(np.float32) / self.norm_scaler
        ret[ np.where(ret > 1) ] = 1
        return ret

    def preprocess_rangeimage_data(self, X):
        return X.astype(np.float32) / self.args.norm_value

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.data[idx:idx+self.nt])
        return X_all
