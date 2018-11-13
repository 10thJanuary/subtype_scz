# coding=utf-8
import os

base_path = os.path.join(os.path.expanduser('~'), 'data', 'SCZ_973')

ts_file_name = 'bn_cb_mean.pkl'
rsfMRI_path = os.path.join(base_path, 'rsfMRI', 'ts')

subj_base_path = os.path.join(base_path, 'subj_info')
SZ_info_path = os.path.join(subj_base_path, 'SZ_all.csv')
NC_info_path = os.path.join(subj_base_path, 'NC_all.csv')
HR_info_path = os.path.join(subj_base_path, 'HR_all.csv')

vbm_base_path = os.path.join(base_path, 'vbm_analysis', 'BN')
vbm_features = ['m0wrp1_GM', 'm0wrp2_WM', 'm0wrp3_CSF']

fs_stats_base_path = os.path.join(base_path, 'Surface', 'surface_stats')
fs_stats_files = [
    'aseg.stats',
    'wmparc.stats',
]

# subject info object
SZ_INFO = None
NC_INFO = None

# subject rsfMRI object
SZ_fMRI = None
NC_fMRI = None

# subject VBM object
SZ_VBM = None
NC_VBM = None

# subject fs sMRI object
SZ_FS_sMRI = None
NC_FS_sMRI = None
