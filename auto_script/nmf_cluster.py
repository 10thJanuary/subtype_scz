# coding=utf-8

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF


center_train_num = 5
center_no_strings = ['01', '02', '05_01', '05_02', '07', '09', '10']

cluster_nums = [2, 3, 4, 5, 6, 7, 8, 9, 10]

base_save_dir = os.path.join(os.path.abspath('..'), 'auto_report')
save_dir = os.path.join(base_save_dir, 'MI', 'center_train_num_' + str(center_train_num))

NMF_parameters = {
    'n_components': cluster_nums,
    'solver': ['cd', 'mu'],
    'beta_loss': ['frobenius', 'kullback-leibler', 'itakura-saito'],
}

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


def dump_multiple_lines(obj, name, dir_path=save_dir):
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

    # sex
    res_sex = {}
    for k, v in subgroup.items():
        df_group = df_data.loc[v]
        df_sex_1 = df_group.loc[df_group['sex'] == '1']
        df_sex_2 = df_group.loc[df_group['sex'] == '2']
        res_sex.update({str(k): '%s / %s' % (df_sex_1.shape[0], df_sex_2.shape[0])})

    res.update({'sex': res_sex})

    # age
    res_age, group_values = {}, []
    for k, v in subgroup.items():
        df_group = df_data.loc[v]
        age_values = df_group['age_research'].values.astype('float32').tolist()
        group_values.append(age_values)

    statistic, p_value = stats.f_oneway(*group_values)
    res_age.update({'1-way ANOVA': 's=%s, p=%s' % (statistic, p_value)})

    statistic, p_value = stats.kruskal(*group_values)
    res_age.update({'Kruskal-Wallis H-test': 'S=%s, p=%s' % (statistic, p_value)})

    res.update({'age': res_age})

    # education
    res_education, group_values = {}, []
    for k, v in subgroup.items():
        df_group = df_data.loc[v]
        education_values = df_group['education'].values.astype('float32').tolist()
        group_values.append(education_values)

    statistic, p_value = stats.f_oneway(*group_values)
    res_education.update({'1-way ANOVA': 's=%s, p=%s' % (statistic, p_value)})

    statistic, p_value = stats.kruskal(*group_values)
    res_education.update({'Kruskal-Wallis H-test': 's=%s, p=%s' % (statistic, p_value)})

    res.update({'education': res_education})

    # PANSS
    res_panss = {}
    panss_cols = PANSS_TOTAL + PANSS_P + PANSS_N + PANSS_G
    for panss_item in panss_cols:
        res_panss_item, group_values = {}, []
        for k, v in subgroup.items():
            df_group = df_data.loc[v]
            panss_values = df_group[panss_item].values.astype('float32').tolist()
            group_values.append(panss_values)

        statistic, p_value = stats.f_oneway(*group_values)
        res_panss_item.update({'1-way ANOVA': 's=%s, p=%s' % (statistic, p_value)})

        statistic, p_value = stats.kruskal(*group_values)
        res_panss_item.update({'Kruskal-Wallis H-test': 's=%s, p=%s' % (statistic, p_value)})

        res_panss.update({panss_item: res_panss_item})

    res.update({'panss': res_panss})

    return res


if __name__ == '__main__':
    import sys
    sys.path.append('..')

    import CONFIG
    from data_process import create_info_df, create_fmri_df

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

        dump_multiple_lines(obj=cached_content, name=file_name)

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
                    obj_write = ['\n[NMF parameters]\n'] + obj_write
                    print(obj_write)
                    cached_content.extend(obj_write)

                    obj_write = 'nmf model n_iter_: %s\n' % nmf_model.n_iter_
                    print(obj_write)
                    cached_content.append(obj_write)

                    res_subgroup = get_subgroup(df_data=df_train, nmf_h=H, group_num=n_components)

                    obj_write = ['%s: %s\n' % (k, len(v)) for k, v in res_subgroup.items()]
                    obj_write = ['\n[subgroup stats]\n'] + obj_write
                    print(obj_write)
                    cached_content.extend(obj_write)

                    res_stats = group_stats(df_data=df_train, subgroup=res_subgroup)

                    obj_write = ['%s: %s\n' % (k, v) for k, v in res_stats['sex'].items()]
                    obj_write = ['\n[sex stats]\n'] + obj_write
                    print(obj_write)
                    cached_content.extend(obj_write)

                    obj_write = ['%s: %s\n' % (k, v) for k, v in res_stats['age'].items()]
                    obj_write = ['\n[age stats]\n'] + obj_write
                    print(obj_write)
                    cached_content.extend(obj_write)

                    obj_write = ['%s: %s\n' % (k, v) for k, v in res_stats['education'].items()]
                    obj_write = ['\n[education stats]\n'] + obj_write
                    print(obj_write)
                    cached_content.extend(obj_write)

                    panss_items = PANSS_TOTAL + PANSS_P + PANSS_N + PANSS_G
                    for item in panss_items:
                        obj_write = ['%s: %s\n' % (k, v) for k, v in res_stats['panss'][item].items()]
                        obj_write = ['\n[panss stats %s]\n' % item] + obj_write
                        print(obj_write)
                        cached_content.extend(obj_write)

                    dump_multiple_lines(obj=cached_content, name=file_name)
