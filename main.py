# coding=utf-8

from data_process import *


if __name__ == '__main__':
    create_info_df(from_cached=False)
    create_fmri_df()
    create_vbm_df()
