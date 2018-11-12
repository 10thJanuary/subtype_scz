# coding=utf-8

import os
import CONFIG
import pandas as pd
from general_io import dump_obj_pkl, load_obj_pkl


def create_smri_df():
    cached_path = os.path.join(os.path.abspath('.'), 'cached_objects')
    if not os.path.exists(cached_path):
        os.mkdir(cached_path)

    cached_objects = ['SZ_FS_sMRI.pkl', 'NC_FS_sMRI.pkl']
    for obj in cached_objects:
        if not os.path.exists(os.path.join(cached_path, obj)):
            _load_subject_smri()
            break

    CONFIG.SZ_FS_sMRI = load_obj_pkl(file_name='SZ_FS_sMRI')
    CONFIG.NC_FS_sMRI = load_obj_pkl(file_name='NC_FS_sMRI')


def _load_subject_smri():
    nc_subjs, sz_subjs = _get_complete_smri_subjects()
    df_nc = pd.DataFrame(index=nc_subjs)
    df_sz = pd.DataFrame(index=sz_subjs)
    pass


def _get_complete_smri_subjects():
    remove_stats_subjects = []
    for subj in os.listdir(CONFIG.fs_stats_base_path):
        for stats_file in CONFIG.fs_stats_files:
            stats_file_path = os.path.join(CONFIG.fs_stats_base_path, subj, stats_file)
            if not os.path.exists(stats_file_path):
                remove_stats_subjects.append(subj)
                break

    reserve_stats_subjects = [subj for subj in os.listdir(CONFIG.fs_stats_base_path)
                              if subj not in remove_stats_subjects]

    nc_subjs, sz_subjs = [], []
    for subj in reserve_stats_subjects:
        if subj.startswith('NC'):
            nc_subjs.append(subj)
        elif subj.startswith('SZ'):
            sz_subjs.append(subj)

    return sorted(nc_subjs), sorted(sz_subjs)
