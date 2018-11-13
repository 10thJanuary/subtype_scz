# coding=utf-8

import os
import CONFIG
import pandas as pd
import numpy as np
from general_io import dump_obj_pkl, load_obj_pkl


def create_vbm_df():
    cached_path = os.path.join(os.path.abspath('.'), 'cached_objects')
    if not os.path.exists(cached_path):
        os.mkdir(cached_path)

    cached_objects = ['SZ_VBM.pkl', 'NC_VBM.pkl']
    for obj in cached_objects:
        if not os.path.exists(os.path.join(cached_path, obj)):
            _load_subject_vbm()
            break

    CONFIG.SZ_VBM = load_obj_pkl(file_name='SZ_VBM')
    CONFIG.NC_VBM = load_obj_pkl(file_name='NC_VBM')


def _load_subject_vbm():
    nc_subjs, sz_subjs = _get_complete_vbm_subjects()
    df_nc = pd.DataFrame(index=nc_subjs)
    df_sz = pd.DataFrame(index=sz_subjs)

    for vbm_feature in CONFIG.vbm_features:
        vbm_path = os.path.join(CONFIG.vbm_base_path, vbm_feature)

        nc_feature_array = np.array([])
        for subj in df_nc.index:
            vbm_file = subj + '_' + vbm_feature + '.pkl'
            vbm_array = load_obj_pkl(file_name=vbm_file, dir_path=vbm_path).reshape(1, -1)

            if nc_feature_array.size == 0:
                nc_feature_array = vbm_array
            else:
                nc_feature_array = np.vstack((nc_feature_array, vbm_array))

        sz_feature_array = np.array([])
        for subj in df_sz.index:
            vbm_file = subj + '_' + vbm_feature + '.pkl'
            vbm_array = load_obj_pkl(file_name=vbm_file, dir_path=vbm_path).reshape(1, -1)

            if sz_feature_array.size == 0:
                sz_feature_array = vbm_array
            else:
                sz_feature_array = np.vstack((sz_feature_array, vbm_array))

        assert nc_feature_array.shape[1] == sz_feature_array.shape[1]
        feature_names = [vbm_feature + '_' + str(i) for i in range(nc_feature_array.shape[1])]

        df = pd.DataFrame(data=nc_feature_array, index=df_nc.index, columns=feature_names)
        df_nc = pd.merge(df_nc, df, how='inner', left_index=True, right_index=True)

        df = pd.DataFrame(data=sz_feature_array, index=df_sz.index, columns=feature_names)
        df_sz = pd.merge(df_sz, df, how='inner', left_index=True, right_index=True)

    dump_obj_pkl(obj=df_nc, file_name='NC_VBM')
    dump_obj_pkl(obj=df_sz, file_name='SZ_VBM')


def _get_complete_vbm_subjects():
    res_feature_subjs = {}

    for vbm_feature in CONFIG.vbm_features:
        vbm_path = os.path.join(CONFIG.vbm_base_path, vbm_feature)
        feature_subjs = [subj[:(len(subj) - len('_' + vbm_feature + '.pkl'))] for subj in os.listdir(vbm_path)]
        res_feature_subjs.update({vbm_feature: feature_subjs})

    feature_subjs = set()
    for _, val in res_feature_subjs.items():
        if len(feature_subjs) == 0:
            feature_subjs = set(val)
        else:
            feature_subjs = feature_subjs.intersection(set(val))

    nc_subjs, sz_subjs = [], []
    for subj in list(feature_subjs):
        if subj.startswith('NC'):
            nc_subjs.append(subj)
        elif subj.startswith('SZ'):
            sz_subjs.append(subj)

    return sorted(nc_subjs), sorted(sz_subjs)
