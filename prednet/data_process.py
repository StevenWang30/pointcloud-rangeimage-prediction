
import numpy as np
import os
import IPython


# train_data, train_sources_data = data_process.load(data_list)

def load(args, data_path, save_bin):
    save_bin_path = os.path.join(args.data_docker_root_dir, save_bin)

    if(os.path.exists(save_bin_path)):
        data = np.load(save_bin_path)
        print("load bin file from " + save_bin_path)
        print("finish the data process.")
        return data['arr_0'], data['arr_1']

    path = np.loadtxt(data_path, dtype=str)
    args.rangeimage_size[0] = path.shape[0]
    data = np.zeros(args.rangeimage_size)
    # sources = np.zeros(args.rangeimage_size[0], dtype=str)
    sources = []
    # IPython.embed()
    for i in range(path.shape[0]):
        data_name_path = path[i].replace('/home/skwang/data', args.data_docker_root_dir)
        # IPython.embed()
        data_name = data_name_path.split('/')[-1]
        data_path = data_name_path.replace('/' + data_name, '/')
        data_arr = np.loadtxt(data_name_path, dtype=float)
        print("load range image data from " + data_name_path + "...")
        temp = np.zeros(args.rangeimage_size[1:]) # change [64,2000] to [64, 2000, 1]
        temp[:, :, 0] = data_arr[:, :]
        # IPython.embed()
        data[i] = temp
        sources.append(data_path)

    np.savez(save_bin_path, data, sources)
    print("save bin file into " + save_bin_path)
    print("finish the data process.")
        # arr = np.loadtxt(data_path, dtype=float)

    # IPython.embed()
    # print(arr)
    return data, sources