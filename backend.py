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



