# coding=utf-8

import os
import joblib
import CONFIG
import pandas as pd
import numpy as np
from general_io import dump_obj_pkl, load_obj_pkl
from scz_center_973 import get_center_info


def create_info_raw():
    cached_path = os.path.join(os.path.abspath('.'), 'cached_objects')
    if not os.path.exists(cached_path):
        os.mkdir(cached_path)

    cached_objects = ['SZ_INFO_RAW.pkl', 'NC_INFO_RAW.pkl']
    for obj in cached_objects:
        if not os.path.exists(os.path.join(cached_path, obj)):
            _create_subj_info()
            break

    CONFIG.SZ_INFO_RAW = load_obj_pkl(file_name='SZ_INFO_RAW')
    CONFIG.NC_INFO_RAW = load_obj_pkl(file_name='NC_INFO_RAW')


def _create_subj_info():
    # read SZ NC subj info csv
    encoding = 'ISO-8859-15'
    index_name = 'subject'
    df_nc = pd.read_csv(CONFIG.NC_info_path, encoding=encoding).set_index(index_name)
    df_sz = pd.read_csv(CONFIG.SZ_info_path, encoding=encoding).set_index(index_name)

    # filter SZ NC: subject has info but no fMRI
    df_nc, df_sz = _filter_fmri_name(df_nc=df_nc, df_sz=df_sz)

    # filter SZ NC: subject missing basic info
    df_nc, df_sz = _filter_subject_info(df_nc=df_nc, df_sz=df_sz)

    # filter SZ NC: subject ts array invalid, some region's ts == 0 constantly
    df_nc, df_sz = _filter_invalid_ts(df_nc=df_nc, df_sz=df_sz)

    # read center info: center_no, center_info
    subject_center_info = df_nc.index.map(get_center_info)
    df_nc = df_nc.assign(center_no=[c['center_no_string'] for c in subject_center_info])
    df_nc = df_nc.assign(center_name=[c['center_name'] for c in subject_center_info])

    subject_center_info = df_sz.index.map(get_center_info)
    df_sz = df_sz.assign(center_no=[c['center_no_string'] for c in subject_center_info])
    df_sz = df_sz.assign(center_name=[c['center_name'] for c in subject_center_info])

    dump_obj_pkl(obj=df_nc, file_name='NC_INFO_RAW')
    dump_obj_pkl(obj=df_sz, file_name='SZ_INFO_RAW')


def _filter_fmri_name(df_nc, df_sz):
    nc_ts_names, sz_ts_names = [], []
    for subj_name in os.listdir(CONFIG.rsfMRI_path):
        if subj_name.startswith('NC'):
            nc_ts_names.append(subj_name)
        elif subj_name.startswith('SZ'):
            sz_ts_names.append(subj_name)

    df_nc = df_nc.loc[df_nc.index.isin(nc_ts_names)]
    df_sz = df_sz.loc[df_sz.index.isin(sz_ts_names)]

    return df_nc, df_sz


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


def _filter_invalid_ts(df_nc, df_sz):
    dir_path = CONFIG.rsfMRI_path

    subj_ts = {subj: joblib.load(os.path.join(dir_path, subj, CONFIG.ts_file_name)).T for subj in df_nc.index}
    df_nc = df_nc.loc[_remove_ts_subj(subj_ts)]

    subj_ts = {subj: joblib.load(os.path.join(dir_path, subj, CONFIG.ts_file_name)).T for subj in df_sz.index}
    df_sz = df_sz.loc[_remove_ts_subj(subj_ts)]

    return df_nc, df_sz


def _remove_ts_subj(subj_ts):
    remove_subj = []
    for subj, ts in subj_ts.items():
        for col in range(ts.shape[1]):
            if np.all(ts[:, col] == 0):
                remove_subj.append(subj)
                break
    [subj_ts.pop(subj) for subj in remove_subj]

    return sorted(list(subj_ts.keys()))
