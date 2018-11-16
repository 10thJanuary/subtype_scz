# coding=utf-8

import os
import CONFIG
import pandas as pd
from general_io import dump_obj_pkl, load_obj_pkl


def create_pgrs_df():
    cached_path = os.path.join(os.path.abspath('.'), 'cached_objects')
    if not os.path.exists(cached_path):
        os.mkdir(cached_path)

    cached_objects = ['SZ_PGRS.pkl', 'NC_PGRS.pkl']
    for obj in cached_objects:
        if not os.path.exists(os.path.join(cached_path, obj)):
            _load_subject_pgrs()
            break

    CONFIG.SZ_PGRS = load_obj_pkl(file_name='SZ_PGRS')
    CONFIG.NC_PGRS = load_obj_pkl(file_name='NC_PGRS')


def _load_subject_pgrs():
    encoding = 'ISO-8859-15'
    index_name = 'FID'
    df = pd.read_csv(CONFIG.pgrs_path, encoding=encoding).set_index(index_name)

    nc_subjs = [subj for subj in df.index if subj.startswith('NC')]
    df_nc = df.loc[nc_subjs]

    sz_subjs = [subj for subj in df.index if subj.startswith('SZ')]
    df_sz = df.loc[sz_subjs]

    dump_obj_pkl(obj=df_nc, file_name='NC_PGRS')
    dump_obj_pkl(obj=df_sz, file_name='SZ_PGRS')
