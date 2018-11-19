# coding=utf-8

import os

base_path = os.path.join(os.path.expanduser('~'), 'data', 'SCZ_973')

# info path
subj_base_path = os.path.join(base_path, 'subj_info')
SZ_info_path = os.path.join(subj_base_path, 'SZ_all.csv')
NC_info_path = os.path.join(subj_base_path, 'NC_all.csv')

# ts_name, ts_path, ts_mi_path
ts_file_name = 'bn_cb_mean.pkl'
rsfMRI_path = os.path.join(base_path, 'rsfMRI', 'ts')
ts_mi_path = os.path.join(base_path, 'rsfMRI', 'ts_mi_analysis')

# VBM features and path
vbm_base_path = os.path.join(base_path, 'vbm_analysis', 'BN')
vbm_features = ['m0wrp1_GM', 'm0wrp2_WM', 'm0wrp3_CSF']

# FreeSurfer stats name and path
fs_stats_base_path = os.path.join(base_path, 'Surface', 'surface_stats')
fs_stats_names = ['aseg.stats', 'wmparc.stats']

# PGRS path
pgrs_path = os.path.join(base_path, 'gene', 'PGRS_973.csv')


# info object
SZ_INFO = None
NC_INFO = None

# rsfMRI object
SZ_fMRI = None
NC_fMRI = None

# VBM object
SZ_VBM = None
NC_VBM = None

# FreeSurfer sMRI object
SZ_FS_sMRI = None
NC_FS_sMRI = None

# PGRS object
SZ_PGRS = None
NC_PGRS = None
