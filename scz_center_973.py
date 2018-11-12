# coding=utf-8

# PKU6: 北大六院
# HLG: 北京回龙观医院
# ANHUI: 安徽医科大学
# XINXIANG: 新乡医学院二附院
# XIAN: 第四军医大学西京医院
# GUANGZHOU: 广州市精神病医院
# HUBEI: 湖北省人民医院
# ZMD: 驻马店精神病院

# Using: 01, 02, 05_01, 05_02, 07, 09, 10
CENTER_NO_NAME_973 = {
    '01': 'PKU6',
    '02': 'HLG',
    '04': 'AHHUI',
    '05_01': 'XINXIANG_01',
    '05_02': 'XINXIANG_02',
    '07': 'XIAN',
    '08': 'GUANGZHOU',
    '09': 'HUBEI',
    '10': 'ZMD',
}


def get_center_info(subj_name):
    subj_name_split = subj_name.split('_')
    assert len(subj_name_split) >= 3

    subj_type_string = subj_name_split[0]
    center_no_string = subj_name_split[1]
    subj_no_number = int(subj_name_split[2])

    if subj_type_string == 'SZ':
        res = _get_sz_center(center_no_string, subj_no_number)
    elif subj_type_string == 'NC':
        res = _get_nc_center(center_no_string, subj_no_number)
    elif subj_type_string == 'HR':
        res = _get_hr_center(center_no_string, subj_no_number)
    else:
        res = {'center_no_string': '', 'center_name': ''}

    return res


def _get_sz_center(center_no_string, subj_no_number):
    if center_no_string == '05':
        if subj_no_number > 86 or subj_no_number == 42:
            center_no_string += '_02'
        else:
            center_no_string += '_01'

    res = {'center_no_string': center_no_string, 'center_name': CENTER_NO_NAME_973[center_no_string]}

    return res


def _get_nc_center(center_no_string, subj_no_number):
    if center_no_string == '05':
        if subj_no_number > 99:
            center_no_string += '_02'
        else:
            center_no_string += '_01'

    res = {'center_no_string': center_no_string, 'center_name': CENTER_NO_NAME_973[center_no_string]}

    return res


def _get_hr_center(center_no_string, subj_no_number):
    if center_no_string == '05':
        if subj_no_number > 97:
            center_no_string += '_02'
        else:
            center_no_string += '_01'

    res = {'center_no_string': center_no_string, 'center_name': CENTER_NO_NAME_973[center_no_string]}

    return res
