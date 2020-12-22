import os
import pandas as pd


def get_root_dir():
    """
    function to get the root directory
    :return: root dir string
    """
    root_dir_nm = 'C:/Plans3'  # name of root dir
    # check if the main dir is already in disk
    if os.path.exists(root_dir_nm):  # existing condition
        pass
    else:
        os.mkdir(root_dir_nm)
    return root_dir_nm


def get_prj_dirs():
    dct = {'Datasets':'datasets', 'Observed':'observed', 'Projected':'projected',
           'Runbin':'runbin', 'Simulation':'simulation', 'Optimization':'optimization'}
    return dct


def get_prj_dirs_paths(p0='name', wkplc='C:'):
    dirs = get_prj_dirs()
    dir0 = wkplc + '/' + p0
    dir01 = dir0 + '/' + dirs['Datasets']
    dir02 = dir0 + '/' + dirs['Runbin']
    dir011 = dir01 + '/' + dirs['Observed']
    dir012 = dir01 + '/' + dirs['Projected']
    dir021 = dir02 + '/' + dirs['Simulation']
    dir022 = dir02 + '/' + dirs['Optimization']
    def_dct = {'Project': dir0, 'Datasets': dir01, 'Observed': dir011, 'Projected': dir012,
               'Runbin': dir02, 'Simulation': dir021, 'Optimization': dir022}
    return def_dct


def check_exist_project(p0, wkplc='C:'):
    """
    check if the project dir is already in disk
    :param p0: string of project name
    :param wkplc: directory path of workplace
    :return: boolean
    """
    prj_path = wkplc + '/'+ p0
    flag = False
    if os.path.exists(prj_path):  # existing condition
        flag = True
    return flag


def create_new_project(p0, wkplc='C:'):
    """
    create a new empty project
    :param p0: project name
    :param wkplc: workplace (root dir)
    :return: project string path
    """
    # create main dir
    new_prj_path = wkplc + '/' +  p0
    os.mkdir(new_prj_path)
    # create sub directories
    subdirs = get_prj_dirs()
    os.mkdir(new_prj_path + '/' + subdirs['Datasets'])
    os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Observed'])
    os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Projected'])
    os.mkdir(new_prj_path + '/' + subdirs['Runbin'])
    os.mkdir(new_prj_path + '/' + subdirs['Runbin'] + '/' + subdirs['Simulation'])
    os.mkdir(new_prj_path + '/' + subdirs['Runbin'] + '/' + subdirs['Optimization'])
    return new_prj_path


def get_existing_projects(wkplc='C:'):
    """
    load to a dataframe the current projects in the workplace
    :param wkplc: workplace directory
    :return: dataframe
    """
    wkplc_items = os.listdir(wkplc)
    wkplc_dirs = list()
    for i in range(len(wkplc_items)):
        lcl_path = wkplc + '/' + wkplc_items[i]
        if os.path.isdir(lcl_path):
            wkplc_dirs.append(lcl_path)
            #print('{} is dir'.format(lcl_path))
    wkplc_projects_paths = list()
    dct_prj_dirs = get_prj_dirs()
    for i in range(len(wkplc_dirs)):
        lcl_items = os.listdir(wkplc_dirs[i])
        if dct_prj_dirs['Datasets'] in set(lcl_items) and dct_prj_dirs['Runbin'] in set(lcl_items):
            lcl_items = os.listdir(wkplc_dirs[i] + '/' + dct_prj_dirs['Datasets'])
            if dct_prj_dirs['Observed'] in set(lcl_items) and dct_prj_dirs['Projected'] in set(lcl_items):
                lcl_items = os.listdir(wkplc_dirs[i] + '/' + dct_prj_dirs['Runbin'])
                if dct_prj_dirs['Simulation'] in set(lcl_items) and dct_prj_dirs['Optimization'] in set(lcl_items):
                    wkplc_projects_paths.append(wkplc_dirs[i])
        else:
            pass
    wkplc_projects_names = list()
    for i in range(len(wkplc_projects_paths)):
        lcl_name = wkplc_projects_paths[i].split('/')[-1]
        wkplc_projects_names.append(lcl_name)
    def_df = pd.DataFrame({'Name':wkplc_projects_names, 'Path':wkplc_projects_paths})
    return def_df


def get_observed_files():
    files = get_input_files()
    type = list()
    for i in range(len(files)):
        type.append('imported')
    df_input = pd.DataFrame({'File':files, 'Type':type})
    files = get_derived_files()
    type = list()
    for i in range(len(files)):
        type.append('derived')
    df_deriv = pd.DataFrame({'File': files, 'Type': type})
    files = get_calib_files()
    type = list()
    for i in range(len(files)):
        type.append('calibrated')
    df_calib = pd.DataFrame({'File': files, 'Type': type})
    def_df = df_input.append(df_deriv, ignore_index=True).append(df_calib, ignore_index=True)
    return def_df


def get_derived_files():
    files = ('cn.asc', 'grad.asc')
    return files


def get_input2derived():
    dct = {'cn.asc':('lulc.asc', 'lulc_param.txt', 'soil.asc', 'soil_param.txt'),
           'grad.asc':('slope.asc',)}
    return dct


def get_calib_files():
    files = ('calib1.txt', 'calib2.txt')
    return files


def get_input_files():
    files = ('pop.txt', 'wcons.txt', 'qobs.txt', 'pobs.txt', 'tobs.txt', 'dem.asc', 'slope.asc', 'catcha.asc',
             'aoi.asc', 'gaug.asc', 'target.asc', 'lulc.asc', 'lulc_param.txt', 'soil.asc', 'soil_param.txt',
             'ppat.asc', 'tpat.asc', 'conversion.txt', 'operation.txt', 'tariff.txt', 'elasticity.txt',
             'tc_param.txt')
    return files


def verify_observed_files(p0='name', wkplc='C:'):
    files_df = get_observed_files()
    files = files_df['File']
    existing_files = os.listdir(get_prj_dirs_paths(p0=p0, wkplc=wkplc)['Observed'])
    status = list()
    for i in range(len(files)):
        if files[i] in set(existing_files):
            status.append('OK')
        else:
            status.append('missing')
    files_df['Status'] = status
    return files_df


def verify_input2derived(derived, p0='name', wkplc='C:'):
    aux_dct = get_input2derived()
    files = aux_dct[derived]
    existing_files = os.listdir(get_prj_dirs_paths(p0=p0, wkplc=wkplc)['Observed'])
    status = list()
    for i in range(len(files)):
        if files[i] in set(existing_files):
            status.append('OK')
        else:
            status.append('missing')
    files_df = pd.DataFrame({'File':files, 'Status':status})
    return files_df


def check_input2derived(derived, p0='name', wkplc='C:'):
    files_df = verify_input2derived(derived, p0=p0, wkplc=wkplc)
    if 'missing' in set(files_df['Status'].values):
        return True
    else:
        return False


def check_inputfiles(p0='name', wkplc='C:', type='imported'):
    files_df = verify_observed_files(p0=p0, wkplc=wkplc)
    files_input_df = files_df[files_df['Type'] == type]
    status = list(files_input_df['Status'])
    flag = False
    if 'missing' in set(status):
        flag = True
    return flag


def importfile(src, dst):
    from shutil import copyfile
    copyfile(src=src, dst=dst)
