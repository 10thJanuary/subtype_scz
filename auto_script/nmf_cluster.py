# coding=utf-8

import argparse
import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF


center_no_strings = ['01', '02', '05_01', '05_02', '07', '09', '10']

cluster_nums = [2, 3, 4, 5, 6, 7, 8, 9, 10]

NMF_parameters = {
    'n_components': cluster_nums,
    'solver': ['cd', 'mu'],
    'beta_loss': ['frobenius', 'kullback-leibler', 'itakura-saito'],
}

demography = [
    'age_research',
    'education',
]

PANSS_P = [
    'delusion',
    'confusion',
    'hallucination',
    'excitement',
    'exaggerate',
    'suspicion',
    'hostility',
]

PANSS_N = [
    'bluntness',
    'flinch',
    'communication_disorder',
    'passive',
    'abstract',
    'initiative',
    'inflexible',
]

PANSS_G = [
    'eg1',
    'eg2',
    'eg3',
    'eg4',
    'eg5',
    'eg6',
    'eg7',
    'eg8',
    'eg9',
    'eg10',
    'eg11',
    'eg12',
    'eg13',
    'eg14',
    'eg15',
    'eg16',
]

PANSS_TOTAL = [
    'PANSS_p',
    'PANSS_n',
    'PANSS_g',
    'PANSS_Total'
]


def dump_multiple_lines(obj, name, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_path = os.path.join(dir_path, name)
    with open(file_path, 'a') as f:
        f.writelines(obj)


def create_feature_array(df_data):
    mi_array = np.array(df_data['mi_lower_triangle'].tolist())
    feature_array = MinMaxScaler().fit_transform(mi_array).T

    return feature_array


def get_subgroup(df_data, nmf_h, group_num):
    all_subj_group_no = np.argmax(nmf_h, axis=0)
    res = {}

    for group_no in range(group_num):
        group_no_subj_iloc = np.where(all_subj_group_no == group_no)[0].tolist()
        group_no_subj_names = df_data.iloc[group_no_subj_iloc].index.tolist()

        res.update({'group_' + str(group_no): group_no_subj_names})

    return res


def group_stats(df_data, subgroup):
    res = {}

    # sex distribution
    res_sex = {}
    for k, v in subgroup.items():
        df_group = df_data.loc[v]
        df_sex_1 = df_group.loc[df_group['sex'] == '1']
        df_sex_2 = df_group.loc[df_group['sex'] == '2']
        res_sex.update({str(k): '%s / %s' % (df_sex_1.shape[0], df_sex_2.shape[0])})
    res.update({'sex': res_sex})

    # stats demography PANSS
    res_stats = {}
    stats_columns = demography + PANSS_TOTAL + PANSS_P + PANSS_N + PANSS_G
    for stats_item in stats_columns:
        res_stats_item, group_values = {}, []

        for k, v in subgroup.items():
            df_group = df_data.loc[v]
            values = df_group[stats_item].values.astype('float32').tolist()
            group_values.append(values)

        if len(group_values) > 2:
            statistic, p_value = stats.f_oneway(*group_values)
            res_stats_item.update({'1-way ANOVA': 's=%s, p=%s' % (statistic, p_value)})

            statistic, p_value = stats.kruskal(*group_values)
            res_stats_item.update({'Kruskal-Wallis H-test': 's=%s, p=%s' % (statistic, p_value)})
        elif len(group_values) == 2:
            statistic, p_value = stats.ttest_ind(group_values[0], group_values[1])
            res_stats_item.update({'T-test': 's=%s, p=%s' % (statistic, p_value)})

        res_stats.update({stats_item: res_stats_item})

    res.update({'stats': res_stats})

    return res


def create_parser():
    parser = argparse.ArgumentParser(
        prog='nmf_cluster',
        description='auto script: nmf cluster stats.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--center_num', type=int, help='center num to train for NMF')

    return parser


if __name__ == '__main__':
    import sys
    sys.path.append('..')

    import CONFIG
    from data_process import create_info_df, create_fmri_df

    args = create_parser().parse_args()
    if args.center_num is not None:
        center_train_num = int(args.center_num)
    else:
        center_train_num = 5

    base_save_dir = os.path.join(os.path.abspath('..'), 'auto_report')
    save_dir = os.path.join(base_save_dir, 'MI', 'center_train_num_' + str(center_train_num))

    create_info_df(cached_path=os.path.join(os.path.abspath('..'), 'cached_objects'))
    create_fmri_df(cached_path=os.path.join(os.path.abspath('..'), 'cached_objects'))
    df = pd.merge(CONFIG.SZ_fMRI, CONFIG.SZ_INFO, how='inner', left_index=True, right_index=True)

    cached_content = []
    for center_train in combinations(center_no_strings, center_train_num):
        cached_content.clear()

        center_train = list(center_train)
        file_name = '-'.join(center_train) + '.txt'

        obj_write = 'train center no: %s\n' % center_train
        print(obj_write)
        cached_content.append(obj_write)

        df_train = df.loc[df['center_no'].isin(center_train)]
        obj_write = 'train set shape: %s of %s\n' % (df_train.shape[0], df.shape[0])
        print(obj_write)
        cached_content.append(obj_write)

        train_feature = create_feature_array(df_train)
        obj_write = 'train feature shape: (%s, %s)\n' % (train_feature.shape[0], train_feature.shape[1])
        print(obj_write)
        cached_content.append(obj_write)

        dump_multiple_lines(obj=cached_content, name=file_name, dir_path=save_dir)

        for n_components in NMF_parameters['n_components']:
            for solver in NMF_parameters['solver']:
                for beta_loss in NMF_parameters['beta_loss']:
                    parameters = {
                        'n_components': n_components,
                        'solver': solver,
                        'beta_loss': beta_loss,
                        'init': 'random',
                        'random_state': 0,
                        'tol': 1e-6,
                        'max_iter': int(1e+6),
                    }
                    try:
                        nmf_model = NMF(**parameters)
                        W = nmf_model.fit_transform(train_feature)
                        H = nmf_model.components_
                    except:
                        continue

                    cached_content.clear()

                    obj_write = ['%s: %s\n' % (k, v) for k, v in parameters.items()]
                    obj_write = ['\n' + '*' * 80] + ['\n[NMF parameters]\n'] + obj_write
                    print(obj_write)
                    cached_content.extend(obj_write)

                    obj_write = 'nmf model n_iter_: %s\n' % nmf_model.n_iter_
                    print(obj_write)
                    cached_content.append(obj_write)

                    res_subgroup = get_subgroup(df_data=df_train, nmf_h=H, group_num=n_components)

                    obj_write = ['%s: %s\n' % (k, len(v)) for k, v in res_subgroup.items()]
                    obj_write = ['\n[stats subgroup]\n'] + obj_write
                    print(obj_write)
                    cached_content.extend(obj_write)

                    stats_result = group_stats(df_data=df_train, subgroup=res_subgroup)

                    obj_write = ['%s: %s\n' % (k, v) for k, v in stats_result['sex'].items()]
                    obj_write = ['\n[stats sex]\n'] + obj_write
                    print(obj_write)
                    cached_content.extend(obj_write)

                    stats_items = demography + PANSS_TOTAL + PANSS_P + PANSS_N + PANSS_G
                    for item in stats_items:
                        obj_write = ['%s: %s\n' % (k, v) for k, v in stats_result['stats'][item].items()]
                        obj_write = ['\n[stats %s]\n' % item] + obj_write
                        print(obj_write)
                        cached_content.extend(obj_write)

                    dump_multiple_lines(obj=cached_content, name=file_name, dir_path=save_dir)
