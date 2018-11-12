# coding=utf-8

import os
import joblib
import CONFIG
import numpy as np
import pandas as pd
from general_io import dump_obj_pkl, load_obj_pkl


def create_fmri_df():
    cached_path = os.path.join(os.path.abspath('.'), 'cached_objects')
    if not os.path.exists(cached_path):
        os.mkdir(cached_path)

    cached_objects = ['SZ_fMRI.pkl', 'NC_fMRI.pkl']
    for obj in cached_objects:
        if not os.path.exists(os.path.join(cached_path, obj)):
            _load_subject_ts()
            break

    CONFIG.SZ_fMRI = load_obj_pkl(file_name='SZ_fMRI')
    CONFIG.NC_fMRI = load_obj_pkl(file_name='NC_fMRI')


def _load_subject_ts():
    sz_subj_names, nc_subj_names = [], []
    for subj_name in os.listdir(CONFIG.rsfMRI_path):
        if subj_name.startswith('NC'):
            nc_subj_names.append(subj_name)
        elif subj_name.startswith('SZ'):
            sz_subj_names.append(subj_name)

    df_nc = pd.DataFrame(index=sorted(nc_subj_names))
    df_sz = pd.DataFrame(index=sorted(sz_subj_names))

    ts_path = [os.path.join(CONFIG.rsfMRI_path, subj, CONFIG.ts_file_name) for subj in df_nc.index]
    df_nc = df_nc.assign(ts_path=ts_path)

    ts_path = [os.path.join(CONFIG.rsfMRI_path, subj, CONFIG.ts_file_name) for subj in df_sz.index]
    df_sz = df_sz.assign(ts_path=ts_path)

    ts_array = [joblib.load(ts_path).T for ts_path in df_nc['ts_path'].tolist()]
    df_nc = df_nc.assign(ts_array=ts_array)

    ts_array = [joblib.load(ts_path).T for ts_path in df_sz['ts_path'].tolist()]
    df_sz = df_sz.assign(ts_array=ts_array)

    # subject invalid ts: some roi ts == 0 constantly
    ts_filter = df_nc['ts_array'].apply(_complete_ts)
    df_nc = df_nc.loc[ts_filter]

    ts_filter = df_sz['ts_array'].apply(_complete_ts)
    df_sz = df_sz.loc[ts_filter]

    dump_obj_pkl(obj=df_nc, file_name='NC_fMRI')
    dump_obj_pkl(obj=df_sz, file_name='SZ_fMRI')


def _complete_ts(ts_array):
    for col in range(ts_array.shape[1]):
        if np.all(ts_array[:, col] == 0):
            return False

    return True
