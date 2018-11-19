# coding=utf-8

import os
import joblib
import CONFIG
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from general_io import dump_obj_pkl, load_obj_pkl


def create_fmri_df(from_cached=True):
    cached_path = os.path.join(os.path.abspath('.'), 'cached_objects')
    if not os.path.exists(cached_path):
        os.mkdir(cached_path)

    create_new_flag = False
    if from_cached:
        cached_objects = ['SZ_fMRI.pkl', 'NC_fMRI.pkl']
        for obj in cached_objects:
            if not os.path.exists(os.path.join(cached_path, obj)):
                create_new_flag = True
                break
    else:
        create_new_flag = True

    if create_new_flag:
        _load_subject_ts()

    CONFIG.SZ_fMRI = load_obj_pkl(file_name='SZ_fMRI')
    CONFIG.NC_fMRI = load_obj_pkl(file_name='NC_fMRI')


def _complete_ts(ts_array):
    for col in range(ts_array.shape[1]):
        if np.all(ts_array[:, col] == 0):
            return False

    return True


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

    # invalid ts array: some roi ts == 0 constantly
    ts_filter = df_nc['ts_array'].apply(_complete_ts)
    df_nc = df_nc.loc[ts_filter]

    ts_filter = df_sz['ts_array'].apply(_complete_ts)
    df_sz = df_sz.loc[ts_filter]

    # correlation matrix
    df_nc, df_sz = _compute_correlation_matrix(df_nc=df_nc, df_sz=df_sz)

    # correlation matrix roi degree
    df_nc, df_sz = _compute_correlation_degree(df_nc=df_nc, df_sz=df_sz)

    # mutual information matrix
    df_nc, df_sz = _load_subj_mi(df_nc=df_nc, df_sz=df_sz)

    dump_obj_pkl(obj=df_nc, file_name='NC_fMRI')
    dump_obj_pkl(obj=df_sz, file_name='SZ_fMRI')


def _compute_correlation_matrix(df_nc, df_sz):
    corr_kind = 'correlation'

    # NC correlation
    nc_measure = ConnectivityMeasure(kind=corr_kind)
    nc_correlation_array = nc_measure.fit_transform(df_nc['ts_array'].values)
    for i in range(nc_correlation_array.shape[0]):
        np.fill_diagonal(nc_correlation_array[i], 0)
    df_nc = df_nc.assign(corr_matrix=[nc_correlation_array[i] for i in range(nc_correlation_array.shape[0])])

    # SZ correlation
    sz_measure = ConnectivityMeasure(kind=corr_kind)
    sz_correlation_array = sz_measure.fit_transform(df_sz['ts_array'].values)
    for i in range(sz_correlation_array.shape[0]):
        np.fill_diagonal(sz_correlation_array[i], 0)
    df_sz = df_sz.assign(corr_matrix=[sz_correlation_array[i] for i in range(sz_correlation_array.shape[0])])

    return df_nc, df_sz


def _compute_correlation_degree(df_nc, df_sz):
    threshold_values = [0.4, 0.45, 0.5, 0.55, 0.6]

    # NC ROI degree
    nc_correlation_array = df_nc['corr_matrix'].values
    for value in threshold_values:
        nc_degree_array = np.array([])

        for i in range(nc_correlation_array.shape[0]):
            corr_array = np.abs(nc_correlation_array[i])
            corr_array = np.where(corr_array >= value, 1, 0)
            roi_degree = np.sum(corr_array, axis=1).reshape(1, -1)

            if nc_degree_array.size == 0:
                nc_degree_array = roi_degree
            else:
                nc_degree_array = np.vstack((nc_degree_array, roi_degree))

        df_nc[('corr_degree_' + str(value))] = [nc_degree_array[i, :] for i in range(nc_degree_array.shape[0])]

    # SZ ROI degree
    sz_correlation_array = df_sz['corr_matrix'].values
    for value in threshold_values:
        sz_degree_array = np.array([])

        for i in range(sz_correlation_array.shape[0]):
            corr_array = np.abs(sz_correlation_array[i])
            corr_array = np.where(corr_array >= value, 1, 0)
            roi_degree = np.sum(corr_array, axis=1).reshape(1, -1)

            if sz_degree_array.size == 0:
                sz_degree_array = roi_degree
            else:
                sz_degree_array = np.vstack((sz_degree_array, roi_degree))

        df_sz[('corr_degree_' + str(value))] = [sz_degree_array[i, :] for i in range(sz_degree_array.shape[0])]

    return df_nc, df_sz


def _get_matrix_lower_triangle(matrix, k=-1):
    n, m = matrix.shape
    pair_pos = np.tril_indices(n=n, k=k, m=m)

    return matrix[pair_pos]


def _get_matrix_upper_triangle(matrix, k=1):
    n, m = matrix.shape
    pair_pos = np.triu_indices(n=n, k=k, m=m)

    return matrix[pair_pos]


def _load_subj_mi(df_nc, df_sz):
    nc_mi_array = [load_obj_pkl(file_name=subj + '_mi_' + CONFIG.ts_file_name,
                                dir_path=os.path.join(CONFIG.ts_mi_path, subj)) for subj in df_nc.index]

    df_nc = df_nc.assign(mi_matrix=nc_mi_array)
    df_nc['mi_lower_triangle'] = df_nc['mi_matrix'].map(_get_matrix_lower_triangle)
    df_nc['mi_upper_triangle'] = df_nc['mi_matrix'].map(_get_matrix_upper_triangle)

    sz_mi_array = [load_obj_pkl(file_name=subj + '_mi_' + CONFIG.ts_file_name,
                                dir_path=os.path.join(CONFIG.ts_mi_path, subj)) for subj in df_sz.index]

    df_sz = df_sz.assign(mi_matrix=sz_mi_array)
    df_sz['mi_lower_triangle'] = df_sz['mi_matrix'].map(_get_matrix_lower_triangle)
    df_sz['mi_upper_triangle'] = df_sz['mi_matrix'].map(_get_matrix_upper_triangle)

    return df_nc, df_sz
