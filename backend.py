''' 
PLANS - Planning Nature-based Solutions

Module description:
This module stores all backend functions specific of plans3. 

Copyright (C) 2022 Iporã Brito Possantti

************ GNU GENERAL PUBLIC LICENSE ************

https://www.gnu.org/licenses/gpl-3.0.en.html

Permissions:
 - Commercial use
 - Distribution
 - Modification
 - Patent use
 - Private use
Conditions:
 - Disclose source
 - License and copyright notice
 - Same license
 - State changes
Limitations:
 - Liability
 - Warranty

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
import os
import pandas as pd


def header(p0='title', p1=70, p2='*', p3=True):
    """
    Returns a title string like:


    ********************** TITLE **********************


    :param p0: body of text
    :param p1: size of title
    :param p2: character of decorations
    :param p3: spaces boolean
    :return: a string
    """
    if len(p0) > p1:
        title_aux0 = ''
    else:
        title_aux0 = str(p2 * int((p1 - (len(p0))) / 2))
    title_s = title_aux0 + ' ' + p0.upper() + ' ' + title_aux0
    if p3:
        title_s = '\n\n\n' + title_s + '\n\n'
    return title_s


def center(p0='text', p1=8):
    """
    return a centered string
    :param p0: text to centralize
    :param p1: full length to center
    :return: centered string
    """
    if len(p0) > p1:  # in case the text is longer than the length param
        s = ' ' + p0 + ' '
    else:
        # symmetry:
        if (p1 - len(p0)) % 2 == 0:
            aux_i1 = int((p1 - len(p0))/2)
            s = (' '*aux_i1) + p0 + (' '*aux_i1)
        #
        else:
            aux_i1 = int(round((p1 - len(p0))/2))
            aux_i2 = int((p1 - len(p0)) - aux_i1)
            s = (' '*aux_i2) + p0 + (' '*aux_i1)
    return s


def header_warp():
    """
    function to built the WARP header message
    :return: string with warp header message
    """
    def_int = 70
    def_str1 = center('UFRGS - Universidade Federal do Rio Grande do Sul', def_int)
    def_str2 = center('IPH - Instituto de Pesquisas Hidráulicas', def_int)
    def_str3 = center('WARP - Research Group in Water Resources Management and Planning', def_int)
    def_str4 = center('https://www.ufrgs.br/warp', def_int)
    def_str5 = center('Porto Alegre, Rio Grande do Sul, Brazil', def_int)
    def_str = '\n\n' + def_str1 + '\n' + def_str2 + '\n' + def_str3 + '\n' + def_str4 + '\n' + def_str5 + '\n\n'
    return def_str


def header_plans():
    """
    built plans 3 header message
    :return: string with plans header msg
    """
    def_str0 = 'plans - planning nature-based solutions'.upper()
    def_str1 = 'Version: 3.0'
    def_str3 = 'This software is under the GNU GPL3.0 license'
    def_str4 = 'Source code repository: https://github.com/ipo-exe/plans3/'
    def_str = def_str0 + '\n' + def_str1 + '\n' + def_str3 + '\n' + def_str4 + '\n\n'
    return def_str


def update_iofiles(infile='iofiles.txt', sep='|', outfile='./docs/iofiles.md'):

    def header1(string):
        return '# ' + string + '\n\n'

    def header2(string):
        return '## ' + string + '\n\n'

    def header2_code(string):
        return '## `' + string + '`\n\n'

    def parag(string):
        return string + '\n'

    def table_list(dataframe):
        def_df = dataframe.copy()
        table = list()
        oldfields = def_df.columns.values
        newfields_str = ' | '.join(oldfields)
        table.append(newfields_str)
        head = list()
        for i in range(len(oldfields)):
            head.append(' :--- ')
        header = ' | '.join(head)
        table.append(header)
        for i in range(len(def_df.values)):
            table.append(' | '.join(def_df.values[i].astype('str')))
        return table

    def example_str(dataframe):
        def_df = dataframe.copy()
        oldfields = def_df.columns.values
        newfields = list()
        for i in range(len(oldfields)):
            if i < len(oldfields) - 1:
                newfields.append(oldfields[i] + ';')
            else:
                newfields.append(oldfields[i])
        concat = list()
        for i in range(len(def_df.index)):
            concat.append(';')
        for i in range(len(oldfields)):
            if i < len(oldfields) - 1:
                if def_df.dtypes[i] == 'float64':
                    def_df[oldfields[i]] = def_df[oldfields[i]].round(4)
                def_df[oldfields[i]] = def_df[oldfields[i]].astype(str)
                def_df[oldfields[i]] = def_df[oldfields[i]].str.cat(concat)
        def_df.columns = newfields
        return def_df

    lcl_df = pd.read_csv(infile, sep='|')
    lcl_df.sort_values(by='ioType', ascending=True, inplace=True)
    iotypes = ('input', 'derived', 'extracted', 'output') #lcl_df['ioType'].unique()
    lines = list()
    lines.append(header1('I/O files of `plans3`'))
    lines.append(parag('This document present the list of all Input / Output files of `plans3`.'))
    lines.append(parag('* `input files` are files of external data provided by the user. Example: observed climatic time series data.'))
    lines.append(parag('* `derived files` are files of data the user may generate using `plans3`. Example: terrain slope map derived from DEM map.'))
    lines.append(parag('* `extracted files` are files of data automatically created by `plans3` for the sake of file name organization. Example: land use maps for a given year.'))
    lines.append(parag('* `output files` are files of computed data generated by `plans3`. Example: simulated flow time series.'))
    # IO types loop
    for i in range(len(iotypes)):
        # title
        lines.append(parag('---'))
        lines.append(header1(iotypes[i] + ' files'))
        type_df = lcl_df[lcl_df['ioType'] == iotypes[i]].copy()
        type_df.sort_values(by='FileName', inplace=True)
        files = type_df['FileName'].values
        formats = type_df['FileFormat'].values
        table_df = type_df[['ioType', 'FileName', 'FileFormat', 'FileType']]
        table_lst = table_list(table_df)
        for i in range(len(table_lst)):
            lines.append(table_lst[i] + '\n')
        lines.append('\n')
        # files loop
        for j in range(len(files)):
            lines.append(header2_code(files[j] + '.' + formats[j]))
            # io
            lines.append(parag('- **I/O**: ' + type_df['ioType'].values[j] + '.'))
            # file type
            lines.append(parag('- **File type**: '+ type_df['FileType'].values[j] + '.'))
            # data type
            lines.append(parag('- **Dataset type**: '+ type_df['DataClass'].values[j] + '.'))
            # data info
            if type_df['DataInfo'].values[j].strip()[0] == '>':
                lines.append(parag('- **Dataset description**:'))
                items = type_df['DataInfo'].values[j].strip().split('>>')[1:]
                # items loop
                for k in range(len(items)):
                    aux_string = '\t' + items[k].replace('"', '`')
                    lines.append(parag(aux_string))
            else:
                lines.append(parag('- **Dataset description**: '+ type_df['DataInfo'].values[j]))
            # data reqs
            if type_df['DataReq'].values[j].strip()[0] == '>':
                lines.append(parag('- **Requirements**:'))
                items = type_df['DataReq'].values[j].strip().split('>>')[1:]
                # items loop
                for k in range(len(items)):
                    aux_string = items[k].replace('"', '`').strip()
                    #print(aux_string)
                    if aux_string[0] == '>':
                        aux_string = aux_string.replace('>', '')
                        aux_string = '\t\t - ' + aux_string
                    else:
                        aux_string = '\t - ' + aux_string
                    lines.append(parag(aux_string))
            else:
                lines.append(parag('- **Requirements**: ' + type_df['DataReq'].values[j]))
            # example
            example = type_df['DataEx'].values[j].strip().replace('.', '')
            example_filetype = type_df['FileType'].values[j].strip().replace('.', '')
            if example == 'none':
                pass
            elif example_filetype == 'raster map':
                lines.append(parag('- **Example**:'))
                fig = type_df['DataEx'].values[j].strip() + '.' + 'PNG' # files[j] + '.' + 'png'
                path = 'https://github.com/ipo-exe/plans3/blob/main/docs/figs/'
                line = '\n![alt text](' + path + fig + ' "' + files[j] + '")\n'
                lines.append(parag(line))
            elif files[j] + '.' + formats[j] in set(os.listdir('./samples')) and example == 'sample file':
                lines.append(parag('- **Example**:'))
                file = './samples/' + files[j] + '.txt'
                examp_df = pd.read_csv(file, sep=';')
                examp_df = example_str(examp_df)
                lines.append('```\n')
                if example_filetype == 'csv data frame':
                    lines.append(examp_df.to_string(index=False))
                    lines.append('\n```\n\n')
                elif example_filetype == 'csv time series':
                    lines.append(examp_df.tail(10).to_string(index=False))
                    lines.append('\n```\n\n')
            else:
                lines.append(parag('- **Example**:'))
                lines.append('```\n')
                lines.append(example)
                lines.append('\n```\n\n')

    # export to file
    expfile = open(outfile, 'w')
    expfile.writelines(lines)
    expfile.close()


def get_root_dir(root='C:/Plans3'):
    """
    function to get the root directory
    :return: root dir string
    """
    root_dir_nm = root  # name of root dir
    # check if the main dir is already in disk
    if os.path.exists(root_dir_nm):  # existing condition
        pass
    else:
        os.mkdir(root_dir_nm)
    return root_dir_nm


def get_prj_dirs():
    dct = {'Datasets':'datasets', 'Observed':'observed', 'Projected':'projected',
           'Runbin':'runbin', 'Simulation':'simulation', 'Optimization':'optimization',
           'ETpat':'etpat', 'Assessment':'assessment'}
    return dct


def get_all_lclvars():
    return 'Prec-Temp-IRA-IRI-PET-D-Cpy-TF-Sfs-R-RSE-RIE-RC-Inf-Unz-Qv-Evc-Evs-Tpun-Tpgw-ET-VSA'


def get_all_vars():
    return 'Prec-Temp-IRA-IRI-PET-D-Cpy-TF-Sfs-R-RSE-RIE-RC-Inf-Unz-Qv-Evc-Evs-Tpun-Tpgw-ET-VSA-Q-Qs-Qb'


def get_all_simvars():
    return 'PET-D-Cpy-TF-Sfs-R-RSE-RIE-RC-Inf-Unz-Qv-Evc-Evs-Tpun-Tpgw-ET-VSA-Q-Qs-Qb'


def get_prj_dirs_paths(p0='name', wkplc='C:'):
    """

    :param p0:
    :param wkplc:
    :return:
    """
    dirs = get_prj_dirs()
    dir0 = wkplc + '/' + p0
    dir01 = dir0 + '/' + dirs['Datasets']
    dir02 = 'C:/bin/gramado' #dir0 + '/' + dirs['Runbin']
    dir011 = dir01 + '/' + dirs['Observed']
    dir012 = dir01 + '/' + dirs['Projected']
    dir021 = dir02 + '/' + dirs['Simulation']
    dir022 = dir02 + '/' + dirs['Optimization']
    dir023 = dir02 + '/' + dirs['Assessment']
    #dir0111 = dir011 + '/' + dirs['LULC']
    #dir0112 = dir011 + '/' + dirs['SHRU']
    dir0113 = dir011 + '/' + dirs['ETpat']
    def_dct = {'Project': dir0,
               'Datasets': dir01,
               'Observed': dir011,
               'Projected': dir012,
               'Runbin': dir02,
               'Simulation': dir021,
               'Optimization': dir022,
               'ETpat':dir0113,
               'Assessment': dir023}
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
    #os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Observed'] + '/' + subdirs['LULC'])
    #os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Observed'] + '/' + subdirs['SHRU'])
    os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Observed'] + '/' + subdirs['ETpat'])
    #os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Observed'] + '/' + subdirs['CN']) # deprecated
    #os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Observed'] + '/' + subdirs['PPat'])  # deprecated
    #os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Observed'] + '/' + subdirs['TPat'])  # deprecated
    os.mkdir(new_prj_path + '/' + subdirs['Datasets'] + '/' + subdirs['Projected'])
    os.mkdir(new_prj_path + '/' + subdirs['Runbin'])
    os.mkdir(new_prj_path + '/' + subdirs['Runbin'] + '/' + subdirs['Simulation'])
    os.mkdir(new_prj_path + '/' + subdirs['Runbin'] + '/' + subdirs['Optimization'])
    os.mkdir(new_prj_path + '/' + subdirs['Runbin'] + '/' + subdirs['Assessment'])
    return new_prj_path


def get_existing_projects(wkplc='C:'):
    """
    load to a dataframe the current projects in the workplace
    :param wkplc: workplace directory
    :return: dataframe
    """
    # list dirs in workplace
    wkplc_items = os.listdir(wkplc)
    wkplc_dirs = list()
    for i in range(len(wkplc_items)):
        lcl_path = wkplc + '/' + wkplc_items[i]
        if os.path.isdir(lcl_path):
            wkplc_dirs.append(lcl_path)
    # list projects in
    wkplc_projects_paths = list()
    dct_prj_dirs = get_prj_dirs()
    for i in range(len(wkplc_dirs)):
        # first level
        lcl_items = os.listdir(wkplc_dirs[i])
        if dct_prj_dirs['Datasets'] in set(lcl_items) and dct_prj_dirs['Runbin'] in set(lcl_items):
            # second level
            lcl_items = os.listdir(wkplc_dirs[i] + '/' + dct_prj_dirs['Datasets'])
            if dct_prj_dirs['Observed'] in set(lcl_items) and dct_prj_dirs['Projected'] in set(lcl_items):
                # third level
                lcl_items = os.listdir(wkplc_dirs[i] + '/' + dct_prj_dirs['Runbin'])
                if dct_prj_dirs['Simulation'] in set(lcl_items) and dct_prj_dirs['Optimization'] in set(lcl_items) \
                        and dct_prj_dirs['Assessment'] in set(lcl_items):
                    wkplc_projects_paths.append(wkplc_dirs[i])
        else:
            pass
    wkplc_projects_names = list()
    for i in range(len(wkplc_projects_paths)):
        lcl_name = wkplc_projects_paths[i].split('/')[-1]
        wkplc_projects_names.append(lcl_name)
    def_df = pd.DataFrame({'Name':wkplc_projects_names, 'Path':wkplc_projects_paths})
    return def_df


def get_observed_files(infile='iofiles.txt', sep='|'):
    # extract data
    lcl_df = pd.read_csv(infile, sep=sep)
    # filter by data class
    lcl_df = lcl_df[lcl_df['DataClass'] == 'observed'].copy()
    lcl_df = lcl_df[lcl_df['ioType'] != 'extracted'].copy()
    lcl_df.sort_values(by='ioType', ascending=False, inplace=True)
    # create new field of data + format
    filenames = lcl_df['FileName'].values
    filesformats = lcl_df['FileFormat'].values
    files = list()
    for i in range(len(filenames)):
        files.append(filenames[i] + '.' + filesformats[i])
    lcl_df['File'] = files
    # filter
    lcl_df = lcl_df[['File', 'ioType']].sort_values(['ioType', 'File'], ascending=[False, True]).copy()
    # rename
    lcl_df.rename(columns={'ioType': 'Type'}, inplace=True)
    return lcl_df


def get_input2derived():
    dct = {'calib_twito.asc':('calib_twi.asc',
                              'calib_fto.asc'),
           'aoi_twito.asc':('calib_twi.asc',
                            'calib_fto.asc'),
           'aoi_canopy_series.txt':('aoi_series.txt',
                                    'aoi_shru_param.txt'),
           'calib_canopy_series.txt': ('calib_series.txt',
                                       'calib_shru_param.txt'),
           'calib_shru.asc': ('calib_lulc.asc',
                              'calib_lulc_param.txt',
                              'calib_soils.asc',
                              'calib_soils_param.txt',
                              'calib_shru_param.txt'),
           'aoi_shru.asc': ('aoi_lulc.asc',
                            'aoi_lulc_param.txt',
                            'aoi_soils.asc',
                            'aoi_soils_param.txt',
                            'aoi_shru_param.txt'),
           'aoi_shru_param.txt':('aoi_lulc_param.txt',
                                 'aoi_soils_param.txt'),
           'calib_shru_param.txt':('calib_lulc_param.txt',
                                   'calib_soils_param.txt'),
           'calib_histograms.txt':('calib_shru_param.txt',
                                   'calib_shru.asc',
                                   'calib_twi.asc'),
           'calib_basin_histograms.txt': ('calib_shru_param.txt',
                                          'calib_shru.asc',
                                          'calib_twi.asc',
                                          'calib_basin.asc'),
           'aoi_histograms.txt': ('aoi_shru_param.txt',
                                  'aoi_shru.asc',
                                  'aoi_twi.asc'),
           'aoi_basin_histograms.txt': ('aoi_shru_param.txt',
                                        'aoi_shru.asc',
                                        'aoi_twi.asc',
                                        'aoi_basin.asc'),
           'aoi_slope.asc':('aoi_dem.asc',),
           'calib_slope.asc':('calib_dem.asc',),
           'aoi_fto.asc': ('aoi_soils.asc',
                           'aoi_soils_param.txt'),
           'calib_fto.asc': ('calib_soils.asc',
                             'calib_soils_param.txt'),
           'calib_etpat_series.txt':('calib_etpat_series_input.txt',),
           'calib_etpat_zmaps.txt':('calib_etpat_series.txt',
                                    'calib_twi.asc',
                                    'calib_shru.asc',
                                    'calib_histograms.txt')}
    return dct


def get_mapid_byfile(filename):
    return filename.split('.')[0].split('_')[1]


def get_input2calibhydro():
    files = ('calib_series.txt',
             'hydro_param.txt',
             'calib_shru_param.txt',
             'calib_histograms.txt',
             'calib_basin_histograms.txt',
             'calib_basin.asc',
             'calib_twi.asc',
             'calib_shru.asc',
             'calib_etpat_zmaps.txt',
             'calib_canopy_series.txt',
             'calib_shru_window.asc',
             'calib_twi_window.asc'
             )
    return files


def get_input2simbhydro(aoi=True):
    if aoi:
        files = ('aoi_series.txt',
                 'hydro_param.txt',
                 'aoi_shru_param.txt',
                 'aoi_histograms.txt',
                 'aoi_basin_histograms.txt',
                 'aoi_basin.asc',
                 'aoi_twi.asc',
                 'aoi_shru.asc',
                 'aoi_canopy_series.txt',
                 'aoi_shru_window.asc',
                 'aoi_twi_window.asc')
    else:
        files = ('calib_series.txt',
                 'hydro_param.txt',
                 'calib_shru_param.txt',
                 'calib_histograms.txt',
                 'calib_basin_histograms.txt',
                 'calib_basin.asc',
                 'calib_twi.asc',
                 'calib_shru.asc',
                 'calib_canopy_series.txt',
                 'calib_etpat_zmaps.txt',
                 'calib_shru_window.asc',
                 'calib_twi_window.asc')
    return files


def get_derived_files():
    lcl_df = pd.read_csv(infile, sep=sep)  # extract
    # extract list
    filenames = lcl_df[lcl_df['ioType'] == 'derived']['FileName'].values
    filesformats = lcl_df[lcl_df['ioType'] == 'derived']['FileFormat'].values
    files = list()
    for i in range(len(filenames)):
        files.append(filenames[i] + '.' + filesformats[i])
    return files


def get_input_files(infile='iofiles.txt', sep='|'):
    lcl_df = pd.read_csv(infile, sep=sep)  # extract
    # extract list
    filenames = lcl_df[lcl_df['ioType'] == 'input']['FileName'].values
    filesformats = lcl_df[lcl_df['ioType'] == 'input']['FileFormat'].values
    files = list()
    for i in range(len(filenames)):
        files.append(filenames[i] + '.' + filesformats[i])
    return files


def verify_observed_files(p0='name', wkplc='C:'):
    files_df = get_observed_files()
    files = files_df['File'].values
    existing_files = os.listdir(get_prj_dirs_paths(p0=p0, wkplc=wkplc)['Observed'])
    status = list()
    for i in range(len(files)):
        if files[i] in set(existing_files):
            status.append('OK')
        else:
            status.append('missing')
    files_df = pd.DataFrame({'File':files, 'Type':files_df['Type'], 'Status':status})
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


def verify_simhydro_files(p0='name', wkplc='C:', aoi=True):
    files_df = get_observed_files()
    files = get_input2simbhydro(aoi=aoi)
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


def check_simhydro_files(p0='name', wkplc='C:', aoi=True):
    files_df = verify_simhydro_files(p0=p0, wkplc=wkplc, aoi=aoi)
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


def get_stringfields(filename):
    """
    get the string fields in a string for dataframe pre processing
    :param filename: string of filename with extension
    :return: string of string fields
    """
    def_str = ''
    if filename == 'calib_shru_param.txt' or filename == 'aoi_shru_param.txt' or filename == 'shru':
        def_str = 'SHRUName,SHRUAlias,LULCName,LULCAlias,CanopySeason,ConvertTo,ColorLULC,SoilName,SoilAlias,ColorSoil'
    elif filename == 'calib_soils_param.txt' or filename =='aoi_soils_param.txt' or filename == 'soils':
        def_str = 'SoilName,SoilAlias,ColorSoil'
    elif filename == 'calib_lulc_param.txt' or filename =='aoi_lulc_param.txt' or filename == 'lulc':
        def_str = 'LULCName,LULCAlias,CanopySeason,ConvertTo,ColorLULC'
    return def_str


def get_mapid(var):
    if var == 'D':
        mapid = 'deficit'
    elif var in set(['Cpy', 'Sfs', 'Unz']):
        mapid = 'stock'
    elif var in set(['R', 'Inf', 'TF', 'IRA', 'IRI', 'Qv', 'Prec']):
        mapid = 'flow'
    elif var in set(['ET', 'Evc', 'Evs', 'Tpun', 'Tpgw']):
        mapid = 'flow_v'
    elif var == 'VSA':
        mapid = 'VSA'
    elif var == 'RC':
        mapid = 'RC'
    else:
        mapid = 'flow'
    return mapid
