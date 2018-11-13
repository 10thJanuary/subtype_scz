# coding=utf-8

import os
import joblib
import CONFIG
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
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

    # compute correlation matrix
    df_nc, df_sz = _compute_correlation_matrix(df_nc=df_nc, df_sz=df_sz)

    # compute correlation roi degree
    df_nc, df_sz = _compute_correlation_degree(df_nc=df_nc, df_sz=df_sz)

    dump_obj_pkl(obj=df_nc, file_name='NC_fMRI')
    dump_obj_pkl(obj=df_sz, file_name='SZ_fMRI')


def _complete_ts(ts_array):
    for col in range(ts_array.shape[1]):
        if np.all(ts_array[:, col] == 0):
            return False

    return True


def _compute_correlation_matrix(df_nc, df_sz):
    nc_measure = ConnectivityMeasure(kind='correlation')
    nc_correlation_array = nc_measure.fit_transform(df_nc['ts_array'].values)
    for i in range(nc_correlation_array.shape[0]):
        np.fill_diagonal(nc_correlation_array[i], 0)
    df_nc = df_nc.assign(correlation_matrix=nc_correlation_array.tolist())

    sz_measure = ConnectivityMeasure(kind='correlation')
    sz_correlation_array = sz_measure.fit_transform(df_sz['ts_array'].values)
    for i in range(sz_correlation_array.shape[0]):
        np.fill_diagonal(sz_correlation_array[i], 0)
    df_sz = df_sz.assign(correlation_matrix=sz_correlation_array.tolist())

    return df_nc, df_sz


def _compute_correlation_degree(df_nc, df_sz):
    nc_correlation_array = df_nc['correlation_matrix'].values
    nc_degree_array = np.array([])
    for i in range(nc_correlation_array.shape[0]):
        correlation_array = np.abs(nc_correlation_array[i])
        correlation_array = np.where(correlation_array >= 0.5, 1, 0)
        roi_degree = np.sum(correlation_array, axis=1).reshape(1, -1)

        if nc_degree_array.size == 0:
            nc_degree_array = roi_degree
        else:
            nc_degree_array = np.vstack((nc_degree_array, roi_degree))

    sz_correlation_array = df_sz['correlation_matrix'].values
    sz_degree_array = np.array([])
    for i in range(sz_correlation_array.shape[0]):
        correlation_array = np.abs(sz_correlation_array[i])
        correlation_array = np.where(correlation_array >= 0.5, 1, 0)
        roi_degree = np.sum(correlation_array, axis=1).reshape(1, -1)

        if sz_degree_array.size == 0:
            sz_degree_array = roi_degree
        else:
            sz_degree_array = np.vstack((sz_degree_array, roi_degree))

    assert nc_degree_array.shape[1] == sz_degree_array.shape[1]
    feature_names = ['corr_degree_' + str(i) for i in range(nc_degree_array.shape[1])]

    df = pd.DataFrame(data=nc_degree_array, index=df_nc.index, columns=feature_names)
    df_nc = pd.merge(df_nc, df, how='inner', left_index=True, right_index=True)

    df = pd.DataFrame(data=sz_degree_array, index=df_sz.index, columns=feature_names)
    df_sz = pd.merge(df_sz, df, how='inner', left_index=True, right_index=True)

    return df_nc, df_sz
