# coding=utf-8
import os

base_path = os.path.join(os.path.expanduser('~'), 'data', 'SCZ_973')

ts_file_name = 'bn_cb_mean.pkl'
rsfMRI_path = os.path.join(base_path, 'rsfMRI', 'ts')

# subject info table path
subj_base_path = os.path.join(base_path, 'subj_info')
SZ_info_path = os.path.join(subj_base_path, 'SZ_all.csv')
NC_info_path = os.path.join(subj_base_path, 'NC_all.csv')
HR_info_path = os.path.join(subj_base_path, 'HR_all.csv')

# subject info object
SZ_INFO_RAW = None
NC_INFO_RAW = None

# subject rsfMRI object
SZ_fMRI = None
NC_fMRI = None

