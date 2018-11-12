# coding=utf-8

import os
import pickle

CACHED_DIR = os.path.join(os.path.abspath('.'), 'cached_objects')


def dump_obj_pkl(obj, file_name, dir_path=CACHED_DIR):
    if not file_name:
        raise ValueError('pkl object file name invalid!')

    if not file_name.endswith('.pkl'):
        file_name = file_name + '.pkl'

    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj_pkl(file_name, dir_path=CACHED_DIR):
    if not file_name:
        raise ValueError('pkl object file name invalid!')

    if not file_name.endswith('.pkl'):
        file_name = file_name + '.pkl'

    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'rb') as f:
        return pickle.load(f)
