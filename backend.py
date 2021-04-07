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
           'Runbin':'runbin', 'Simulation':'simulation', 'Optimization':'optimization',
           'LULC':'lulc', 'CN':'cn', 'PPat':'ppat', 'TPat':'tpat'}
    return dct


def get_prj_dirs_paths(p0='name', wkplc='C:'):
    """

    :param p0:
    :param wkplc:
    :return:
    """

    dirs = get_prj_dirs()
    dir0 = wkplc + '/' + p0
    dir01 = dir0 + '/' + dirs['Datasets']
    dir02 = dir0 + '/' + dirs['Runbin']
    dir011 = dir01 + '/' + dirs['Observed']
    dir012 = dir01 + '/' + dirs['Projected']
    dir021 = dir02 + '/' + dirs['Simulation']
    dir022 = dir02 + '/' + dirs['Optimization']
    dir0111 = dir011 + '/' + dirs['LULC']
    dir0112 = dir011 + '/' + dirs['PPat']
    dir0113 = dir011 + '/' + dirs['TPat']
    dir0114 = dir011 + '/' + dirs['CN']
    def_dct = {'Project': dir0, 'Datasets': dir01, 'Observed': dir011, 'Projected': dir012,
               'Runbin': dir02, 'Simulation': dir021, 'Optimization': dir022,
               'LULC':dir0111, 'PPat':dir0112, 'TPat':dir0113, 'CN':dir0114}
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
    os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Observed'] + '/' + subdirs['LULC'])
    os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Observed'] + '/' + subdirs['CN'])
    #os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Observed'] + '/' + subdirs['PPat'])  # deprecated
    #os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Observed'] + '/' + subdirs['TPat'])  # deprecated
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
    '''files = get_calib_files()
    type = list()
    for i in range(len(files)):
        type.append('calibrated')
    df_calib = pd.DataFrame({'File': files, 'Type': type})'''
    def_df = df_input.append(df_deriv, ignore_index=True)#.append(df_calib, ignore_index=True)
    return def_df


def get_derived_files():
    files = ('cn_series.txt', 'cn_calib.asc', 'twi.asc', 'lulc_series.txt', 'lulc_areas.txt')
    return files


def get_input2derived():
    dct = {'cn_series.txt':('lulc_series.txt', 'lulc_param.txt', 'soil.asc', 'soil_param.txt'),
           'cn_calib.asc':('cn_series.txt', 'series_calib.txt'),
           'series_calib_month.txt': ('series_calib.txt','aoi.asc'),
           'twi.asc':('slope.asc', 'catcha.asc'),
           'lulc_series.txt':('lulc_input.txt',),
           'lulc_areas.txt': ('lulc_series.txt', 'lulc_param.txt', 'aoi.asc'),
           'ppat_month.txt':('ppat_input.txt',),
           'tpat_month.txt':('tpat_input.txt',)}
    return dct


def get_input2calibhydro():
    files = ('cn_calib.asc', 'twi.asc',  'gaug.asc', 'series_calib.txt', 'hydro_param.txt')
    return files


def get_input2simbhydro():
    files = ('cn_calib.asc', 'twi.asc',  'aoi.asc', 'series_calib.txt', 'hydro_param.txt')
    return files


def get_input_files():
    files = ('pop.txt', 'wcons.txt', 'series_calib.txt', 'dem.asc', 'slope.asc', 'catcha.asc',
             'aoi.asc', 'gaug.asc', 'target.asc', 'lulc_input.txt', 'lulc_param.txt', 'soil.asc', 'soil_param.txt',
             'conversion.txt', 'operation.txt', 'tariff.txt', 'elasticity.txt',
             'tc_param.txt', 'hydro_param.txt')
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


def verify_calibhydro_files(p0='name', wkplc='C:'):
    files_df = get_observed_files()
    files = get_input2calibhydro()
    existing_files = os.listdir(get_prj_dirs_paths(p0=p0, wkplc=wkplc)['Observed'])
    status = list()
    for i in range(len(files)):
        if files[i] in set(existing_files):
            status.append('OK')
        else:
            status.append('missing')
    files_df = pd.DataFrame({'File':files, 'Status':status})
    return files_df


def verify_simhydro_files(p0='name', wkplc='C:'):
    files_df = get_observed_files()
    files = get_input2simbhydro()
    existing_files = os.listdir(get_prj_dirs_paths(p0=p0, wkplc=wkplc)['Observed'])
    status = list()
    for i in range(len(files)):
        if files[i] in set(existing_files):
            status.append('OK')
        else:
            status.append('missing')
    files_df = pd.DataFrame({'File':files, 'Status':status})
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


def check_calibhydro_files(p0='name', wkplc='C:'):
    files_df = verify_calibhydro_files(p0=p0, wkplc=wkplc)
    if 'missing' in set(files_df['Status'].values):
        return True
    else:
        return False


def check_simhydro_files(p0='name', wkplc='C:'):
    files_df = verify_simhydro_files(p0=p0, wkplc=wkplc)
    if 'missing' in set(files_df['Status'].values):
        return True
    else:
        return False



def importfile(src, dst):
    from shutil import copyfile
    copyfile(src=src, dst=dst)


def nowsep(p0='-'):
    import datetime
    def_now = datetime.datetime.now()
    yr = def_now.strftime('%Y')
    mth = def_now.strftime('%m')
    dy = def_now.strftime('%d')
    hr = def_now.strftime('%H')
    mn = def_now.strftime('%M')
    sg = def_now.strftime('%S')
    def_lst = [yr, mth, dy, hr, mn, sg]
    def_s = str(p0.join(def_lst))
    return def_s


def create_rundir(label='', wkplc='C:'):
    dir_nm = wkplc + '/' + label + '_' + nowsep()
    os.mkdir(dir_nm)
    return dir_nm