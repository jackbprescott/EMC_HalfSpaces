import os
import json
import pickle as pkl
import numpy as np


def create_necessary_dirs(file_path):
    dirs = file_path.split('/')
    for i in range(1, len(dirs)):
        path = '/'.join(dirs[:i])
        if not os.path.isdir(path):
            os.mkdir(path)


def write_json(obj, file_path, indent=4):
    create_necessary_dirs(file_path)
    with open(file_path, 'w') as handle:
        json.dump(obj, handle, indent=indent)


def read_json(file_path):
    with open(file_path, 'r') as handle:
        return json.load(handle)


def write_pickle(obj, file_path):
    create_necessary_dirs(file_path)
    with open(file_path, 'wb') as handle:
        pkl.dump(obj, handle)


def read_pickle(file_path):
    with open(file_path, 'rb') as handle:
        return pkl.load(handle)


def write_np(np_array, file_path):
    create_necessary_dirs(file_path)
    with open(file_path, 'wb') as handle:
        np.save(handle, np_array)


def read_np(file_path):
    with open(file_path, 'rb') as handle:
        return np.load(handle)


def write_str_list(str_list, file_path, separator='\n'):
    with open(file_path) as handle:
        handle.write(separator.join(str_list))


def read_str_list(file_path, separator='\n'):
    with open(file_path) as handle:
        return handle.read().split(separator)