# coding=utf-8

import os
import CONFIG
import pandas as pd
import numpy as np
from general_io import dump_obj_pkl, load_obj_pkl
from scz_center_973 import get_center_info


def create_info_df(cached_path=os.path.join(os.path.abspath('.'), 'cached_objects'), from_cached=True):
    if not os.path.exists(cached_path):
        os.mkdir(cached_path)

    create_new_flag = False
    if from_cached:
        cached_objects = ['SZ_INFO.pkl', 'NC_INFO.pkl']
        for obj in cached_objects:
            if not os.path.exists(os.path.join(cached_path, obj)):
                create_new_flag = True
                break
    else:
        create_new_flag = True

    if create_new_flag:
        _create_subj_info()

    CONFIG.SZ_INFO = load_obj_pkl(file_name='SZ_INFO', dir_path=cached_path)
    CONFIG.NC_INFO = load_obj_pkl(file_name='NC_INFO', dir_path=cached_path)


def _create_subj_info():
    # read SZ NC subj info csv
    encoding = 'gb2312'
    index_name = 'subject'
    df_nc = pd.read_csv(CONFIG.NC_info_path, encoding=encoding).set_index(index_name)
    df_sz = pd.read_csv(CONFIG.SZ_info_path, encoding=encoding).set_index(index_name)

    # filter SZ NC: subject missing basic info
    df_nc, df_sz = _filter_subject_info(df_nc=df_nc, df_sz=df_sz)

    # read center info: center_no, center_info
    subject_center_info = df_nc.index.map(get_center_info)
    df_nc = df_nc.assign(center_no=[c['center_no_string'] for c in subject_center_info])
    df_nc = df_nc.assign(center_name=[c['center_name'] for c in subject_center_info])

    subject_center_info = df_sz.index.map(get_center_info)
    df_sz = df_sz.assign(center_no=[c['center_no_string'] for c in subject_center_info])
    df_sz = df_sz.assign(center_name=[c['center_name'] for c in subject_center_info])

    dump_obj_pkl(obj=df_nc, file_name='NC_INFO')
    dump_obj_pkl(obj=df_sz, file_name='SZ_INFO')


def _filter_subject_info(df_nc, df_sz):
    # sex in ['1', '2']
    df_nc = df_nc.loc[(df_nc['sex'] == '1') | (df_nc['sex'] == '2')]
    df_sz = df_sz.loc[(df_sz['sex'] == '1') | (df_sz['sex'] == '2')]

    # 0 < age_research < 100
    mask = (df_nc['age_research'].replace(' ', '-1').astype(np.float32) > 0) & \
           (df_nc['age_research'].replace(' ', '-1').astype(np.float32) < 100)
    df_nc = df_nc.loc[mask]

    mask = (df_sz['age_research'].replace(' ', '-1').astype(np.float32) > 0) & \
           (df_sz['age_research'].replace(' ', '-1').astype(np.float32) < 100)
    df_sz = df_sz.loc[mask]

    # education != ' '
    df_nc = df_nc.loc[df_nc['education'] != ' ']
    df_sz = df_sz.loc[df_sz['education'] != ' ']

    # sz panss total != ' ', nc has no panss
    df_sz = df_sz.loc[df_sz['PANSS_Total'] != ' ']

    return df_nc, df_sz
