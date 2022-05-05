'''
UFRGS - Universidade Federal do Rio Grande do Sul
IPH - Instituto de Pesquisas Hidráulicas
WARP - Research Group in Water Resources Management and Planning
Porto Alegre, Rio Grande do Sul, Brazil

plans - planning nature-based solutions
Version: 3.0

This software is under the GNU GPL3.0 license

Source code repository: https://github.com/ipo-exe/plans3/
Authors: Ipora Possantti: https://github.com/ipo-exe

This file is under LICENSE: GNU General Public License v3.0
Permissions:
    Commercial use
    Modification
    Distribution
    Patent use
    Private use
Limitations:
    Liability
    Warranty
Conditions:
    License and copyright notice
    State changes
    Disclose source
    Same license

Module description:
This module stores a terminal-based user interface for plans3.
Run by calling the main() function.
'''
import pandas as pd
import numpy as np
import backend, tools
from backend import center, header_warp, header_plans
from time import sleep
from tkinter import filedialog
from tkinter import *


def missing():
    print('\nSection to be developed!\n')
    sleep(1)
    ok()

def ok():
    print('\n\t>>> OK\n')
    sleep(0.2)


def warning(wng='Warning!', msg=''):
    print('\n\t{} {}\n'.format(wng, msg))
    sleep(0.3)


def proceed(msg='Enter any key to proceed'):
    string = input('\n>>> ' + msg + ': ').strip()
    ok()


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
    print(title_s)
    sleep(0.2)


def status(msg='Status message', process=True):
    if process:
        print('\t>>> {:60}...'.format(msg))
    else:
        print('\n\t>>> {:60}\n'.format(msg))


def validade_project_name(msg='Enter project name', wng='Warning!',
                          wng1='Invalid name. No spaces allowed',
                          wng2='Invalid name. Max of 20 characters allowed'):
    while True:
        nm = input('>>> ' + msg + ': ').strip().lower()
        nm_lst = list(nm)
        nm_set = set(nm_lst)
        if nm == '':
            pass
        elif ' ' in nm_set:
            warning(wng=wng, msg=wng1)
        elif len(nm) > 20:
            warning(wng=wng, msg=wng2)
        else:
            ok()
            break
    return nm


def slice_series(fseries):
    header('Slicing settings')
    _df = pd.read_csv(fseries, sep=';', dtype=str)
    _len = len(_df)
    #
    # size options
    _opt_1 = '1 / 1  | {:6} days'.format(int(_len))
    _opt_2 = '1 / 2  | {:6} days'.format(int(_len/2))
    _opt_3 = '1 / 3  | {:6} days'.format(int(_len / 3))
    _opt_4 = '1 / 4  | {:6} days'.format(int(_len/4))
    _opt_5 = '1 / 5  | {:6} days'.format(int(_len / 5))
    _opt_6 = '1 / 10  | {:6} days'.format(int(_len / 10))
    _opt_7 = '1 / 20  | {:6} days'.format(int(_len / 20))
    _opt_8 = '1 / 30  | {:6} days'.format(int(_len / 30))
    size_opts = (_opt_1,
                 _opt_2,
                 _opt_3,
                 _opt_4,
                 _opt_5,
                 _opt_6,
                 _opt_7,
                 _opt_8)
    size_opt = menu({'Size menu': size_opts}, title='size options', exit=False)
    #
    # start options
    if size_opt != _opt_1:
        _size = int(size_opt[4:].split('|')[0].strip())
        if _size > 10:
            _size = 10
        start_opts = list()
        _step = int(_len/_size)
        for i in range(0, _len - _step + 1, _step):
            start_opts.append('start at day {:6}'.format(str(i)))
        start_opt = menu({'Start menu': start_opts}, title='start options', exit=False)
        #
        #
        # find the dates
        start_date = _df['Date'].values[int(start_opt.split(' ')[3].strip())]
        end_date = _df['Date'].values[int(start_opt.split(' ')[3].strip()) + int(size_opt.split(' ')[-2].strip()) - 1]
        return '{} to {}'.format(start_date, end_date)
    else:
        return 'all'


def settings_simulation(key):
    header('Simulation settings')
    mapback = True
    integrate = True
    mapraster = False
    mapvar = 'all'
    mapdates = 'all'
    #
    # Mapping options
    simulation_opts = ('Default settings | Mapback & Integrate',
                       'Very light | No Mapback',
                       'Light | Mapback',
                       'Moderate | Mapback & Integrate (default)',
                       'Heavy | Rasterize maps')
    map_opts = menu({key: simulation_opts}, title='Mapping settings', exit=False)
    if map_opts == simulation_opts[0]:
        mapback = True
        integrate = True
        mapraster = False
    elif map_opts == simulation_opts[1]:
        mapback = False
        integrate = False
        mapraster = False
    elif map_opts == simulation_opts[2]:
        mapback = True
        integrate = False
        mapraster = False
    elif map_opts == simulation_opts[3]:
        mapback = True
        integrate = True
        mapraster = False
    elif map_opts == simulation_opts[4]:
        mapback = True
        integrate = True
        mapraster = True
    #
    #
    frametype = 'Skip'
    if mapback:
        frame_opts = ('Export all frames',
                           'Export ET pannel frames',
                           'Export Runoff pannel frames',
                           'Export Recharge pannel frames',
                           'Skip frame export')
        exp_opts = menu({key:frame_opts}, title='Frame settings', exit=False)
        if exp_opts == frame_opts[0]:
            frametype = 'all'
        elif exp_opts == frame_opts[1]:
            frametype = 'ET'
        elif exp_opts == frame_opts[2]:
            frametype = 'R'
        elif exp_opts == frame_opts[3]:
            frametype = 'Qv'
        elif exp_opts == frame_opts[4]:
            frametype = 'Skip'
    #
    out_dct = {'Mapback':mapback,
               'Integrate':integrate,
               'Mapraster':mapraster,
               'Frametype':frametype}
    return out_dct


def menu(options, title='Menu', msg='Chose key', exit=True, exitkey='e', exitmsg='Exit menu',
         keylbl='Keys', wng='Warning!', wngmsg='Key not found', chsn='Chosen'):
    """
    Display a Menu
    :param options: dict with string options
    :param title: title string
    :param msg: message string
    :param keylbl: keys label string
    :param wng: not found warning string
    :param chsn: chosen string
    :return: string of option chosen
    """
    options_prime = list(options.values())[0]
    keys = np.arange(1, len(options_prime) + 1)
    keys_str = keys.astype(str)
    keys_full = list(keys_str)
    keys_full.append(exitkey)
    options[keylbl] = keys_str
    def_df = pd.DataFrame(options)
    if exit:
        values = list(def_df.values)
        rowlen = len(values[0])
        exit_row = list()
        for i in range(rowlen):
            if i == 0:
                exit_row.append(exitmsg)
            elif i == rowlen - 1:
                exit_row.append(exitkey)
            else:
                exit_row.append('')
        values.append(exit_row)
        def_df = pd.DataFrame(values, columns=def_df.columns)
    menu_str = def_df.to_string(index=False)
    lcl_len = len(menu_str.split('\n')[0])
    print(title)
    print('_' * lcl_len)
    print(menu_str + '\n')
    exit_flag = False
    while True:
        chosen = input('>>> ' + msg + ': ').strip()
        if chosen == exitkey:
            exit_flag = True
            break
        elif chosen == '':
            pass
        else:
            if chosen not in set(keys_full):
                warning(wng=wng, msg=wngmsg)
            else:
                lcl_index = int(chosen) - 1
                ok()
                print(chsn + ':\t' + options_prime[lcl_index] + '\n')
                break
    if exit_flag:
        return exitmsg
    else:
        return options_prime[lcl_index]


def pick_dir():
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    root.destroy()
    return folder_selected


def pick_file(p0="txt", p1=".txt file", p2="select file"):
    root = Tk()
    root.withdraw()
    def_aux_str = "*." + p0
    def_filename = filedialog.askopenfile(initialdir="/", title=p2, filetypes=((p1, def_aux_str), ("all files", "*.*")))
    root.destroy()
    #def_filename = str(def_filename)
    if def_filename == None:
        return 'cancel'
    else:
        return def_filename.name


def main(root='default', importing=True):
    # print header
    print(header_warp())  # warp header credentials
    sleep(0.3)
    print(header_plans())  # plans header credentials
    sleep(0.3)
    #
    # get root directory
    if root == 'default':
        rootdir = backend.get_root_dir()  # project standard directory on local machine
    else:
        rootdir = backend.get_root_dir(root=root)
    # load dictionary
    dicionary = pd.read_csv('./dictionary.txt', sep=';', engine='python', encoding='utf-8')
    lng = dicionary.T.values[0]  # array of language strings
    languages = tuple(dicionary.columns)  # tuple of languages
    #
    # enter session loop:
    while True:
        exit_flag = False
        #
        # PLANS Menu loop:
        while True:
            header('PLANS 3')
            print('plans3 workplace: {}\n\n'.format(rootdir))
            session_options = (lng[1], lng[2], lng[3])
            opt = menu({lng[6]:session_options},
                       title=lng[0],
                       exitmsg= lng[4],
                       msg=lng[5],
                       keylbl=lng[7],
                       wng=lng[20],
                       wngmsg=lng[8],
                       chsn=lng[9])
            #
            # open project
            if opt == session_options[0]:
                while True:
                    header(lng[1])
                    projects_df = backend.get_existing_projects(wkplc=rootdir)
                    if len(projects_df.index) == 0:
                        warning(wng=lng[20], msg='No projects found.')
                        project_nm = lng[10]
                        break
                    else:
                        projects_names = tuple(projects_df['Name'])
                        project_nm = menu({lng[6]: projects_names},
                                          exitmsg=lng[10],
                                          title=lng[23],
                                          msg=lng[5],
                                          keylbl=lng[7],
                                          wng=lng[20],
                                          wngmsg=lng[8],
                                          chsn=lng[9])
                        break
                if project_nm == lng[10]:
                    pass
                else:
                    break
            #
            # new project:
            elif opt == session_options[1]:
                header(lng[2])
                while True:
                    project_nm = validade_project_name(msg=lng[21], wng=lng[20], wng1=lng[18], wng2=lng[19])
                    if backend.check_exist_project(p0=project_nm, wkplc=rootdir):
                        warning(wng=lng[20], msg=lng[22])
                    else:
                        break
                backend.create_new_project(p0=project_nm, wkplc=rootdir)
                break
            #
            # reset language
            elif opt == session_options[2]:
                header(lng[3])
                lang_options = list(languages[:])
                opt = menu({lng[6]:lang_options},
                           title=lng[11],
                           exitmsg=lng[10],
                           msg=lng[5],
                           keylbl=lng[7],
                           wng=lng[8],
                           chsn=lng[9])
                if opt == lng[10]:
                    pass
                else:
                    lng_id = languages.index(opt)
                    lng = dicionary.T.values[lng_id]
            #
            # exit program
            elif opt == lng[4]:
                exit_flag = True
                break
        #
        # evaluate exit
        if exit_flag:
            break
        #
        # Project Setup
        projectdirs = backend.get_prj_dirs_paths(p0=project_nm, wkplc=rootdir)  # dictionary of project directories
        #
        # Project Menu loop:
        while True:
            header(lng[12] + ': ' + project_nm)
            print('Project workplace: {}\n'.format(projectdirs['Project']))
            project_options = ['Manage Observed Datasets',
                               'Manage Projected Datasets',
                               'Assessment Tools',
                               'Simulation Tools',
                               'Optimization Tools'] #[lng[14], lng[15], lng[16], lng[17]]
            opt = menu({lng[6]:project_options},
                       title=lng[13],
                       exitmsg=lng[10],
                       msg=lng[5],
                       keylbl=lng[7],
                       wng=lng[20],
                       wngmsg=lng[8],
                       chsn=lng[9])
            #
            # manage observed datasets
            if opt == project_options[0]:
                while True:
                    header(lng[14])
                    files_df = backend.verify_observed_files(project_nm, rootdir)
                    print(files_df.to_string(index=False))
                    print('\n')
                    if importing:
                        observed_options = [lng[24], lng[25]]
                    else:
                        observed_options = [lng[25]]
                    lcl_opt = menu({lng[6]:observed_options},
                                   title='',
                                   exitmsg=lng[10],
                                   msg=lng[5],
                                   keylbl=lng[7],
                                   wng=lng[20],
                                   wngmsg=lng[8],
                                   chsn=lng[9])
                    #
                    # import datasets
                    if lcl_opt == lng[24]:
                        # import menu loop
                        while True:
                            header(lcl_opt)
                            inputfiles_df = files_df[files_df['Type'] == 'input']
                            files_lst = list(inputfiles_df['File'])
                            status_lst = list(inputfiles_df['Status'])
                            lcl_lcl_opt = menu({lng[6]: files_lst,
                                               'Status':status_lst},
                                               title='',
                                               exitmsg=lng[10],
                                               msg=lng[5],
                                               keylbl=lng[7],
                                               wng=lng[20],
                                               wngmsg=lng[8],
                                               chsn=lng[9])
                            # exit menu condition
                            if lcl_lcl_opt == lng[10]:
                                break
                            # import file
                            else:
                                extension = lcl_lcl_opt.split('.')[-1]
                                aux_str = '.' + extension + ' file'
                                src_filenm = pick_file(p0=extension, p1=aux_str)
                                if src_filenm == 'cancel':
                                    pass
                                else:
                                    dst_filenm = projectdirs['Observed'] + '/' + lcl_lcl_opt
                                    # copy and paste
                                    backend.importfile(src_filenm, dst_filenm)
                                    # update database
                                    files_df = backend.verify_observed_files(project_nm, rootdir)
                                    print('\n{}:\n{}\n'.format(lng[30], dst_filenm))
                                    tools.view_imported_input(lcl_lcl_opt, folder=projectdirs['Observed'])
                                    ok()
                    #
                    # derive data
                    elif lcl_opt == lng[25]:
                        while True:
                            header(lcl_opt)
                            derivefiles_df = files_df[files_df['Type'] == 'derived']
                            files_lst = list(derivefiles_df['File'])
                            lcl_lcl_opt = menu({lng[6]: files_lst}, title='', exitmsg=lng[10],
                                               msg=lng[5], keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
                            #
                            # exit menu condition
                            if lcl_lcl_opt == lng[10]:
                                break
                            #
                            # derive file
                            else:
                                #
                                # get input needs
                                filesderiv_df = backend.verify_input2derived(derived=lcl_lcl_opt, p0=project_nm, wkplc=rootdir)
                                #
                                # check missing files
                                if backend.check_input2derived(derived=lcl_lcl_opt, p0=project_nm, wkplc=rootdir):
                                    warning(wng=lng[20], msg=lng[27])
                                    print(filesderiv_df.to_string(index=False))
                                else:
                                    # get files names
                                    filesnames = list(filesderiv_df['File'].values)
                                    files_used_df = pd.DataFrame({'Files used':filesnames})
                                    print(files_used_df.to_string(index=False))
                                    #
                                    lcl_filename = lcl_lcl_opt.split('.')[0]
                                    # get files paths
                                    filesp = list()
                                    for i in range(len(filesnames)):
                                        filesp.append(projectdirs['Observed'] + '/' + filesnames[i])
                                    #
                                    # evaluate options
                                    #
                                    # Derive TWITO
                                    if lcl_lcl_opt == 'calib_twito.asc' or lcl_lcl_opt == 'aoi_twito.asc':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.map_twito(filesp[0],
                                                                      filesp[1],
                                                                      folder=projectdirs['Observed'],
                                                                      filename=lcl_filename)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    # Derive SHRU
                                    elif lcl_lcl_opt == 'calib_shru.asc' or lcl_lcl_opt == 'aoi_shru.asc':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.map_shru(filesp[0], filesp[1], filesp[2],
                                                                     filesp[3], filesp[4],
                                                                     folder=projectdirs['Observed'],
                                                                     filename=lcl_filename)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    # Derive Histograms
                                    elif lcl_lcl_opt == 'calib_histograms.txt' or lcl_lcl_opt == 'aoi_histograms.txt':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.compute_histograms(filesp[0], filesp[1], filesp[2],
                                                                               folder=projectdirs['Observed'],
                                                                               filename=lcl_filename, tui=True)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif lcl_lcl_opt == 'calib_canopy_series.txt' or lcl_lcl_opt == 'aoi_canopy_series.txt':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.canopy_series(filesp[0], filesp[1],
                                                                          folder=projectdirs['Observed'],
                                                                          filename=lcl_filename)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif lcl_lcl_opt == 'calib_basin_histograms.txt' or lcl_lcl_opt == 'aoi_basin_histograms.txt':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.compute_histograms(filesp[0],
                                                                               filesp[1],
                                                                               filesp[2],
                                                                               filesp[3],
                                                                               folder=projectdirs['Observed'],
                                                                               filename=lcl_filename,
                                                                               tui=True)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif lcl_lcl_opt == 'calib_etpat_zmaps.txt':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.compute_zmap_series(fvarseries=filesp[0],
                                                                                ftwi=filesp[1], #r"C:\000_myFiles\myDrive\Plans3\pardinho\datasets\observed\calib_twi_window.asc", #
                                                                                fshru=filesp[2], #r"C:\000_myFiles\myDrive\Plans3\pardinho\datasets\observed\calib_shru_window.asc", #
                                                                                fhistograms=filesp[3],
                                                                                var='etpat',
                                                                                folder=projectdirs['Observed'],
                                                                                filename=lcl_filename,
                                                                                dtype='float32',
                                                                                tui=True)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif lcl_lcl_opt == 'calib_etpat_series.txt':
                                        print('\n' + lng[31] + '...')
                                        normalize = False  # todo dialog menu
                                        derivedfile = tools.import_etpat_series(filesp[0],
                                                                                rasterfolder=projectdirs['ETpat'],
                                                                                folder=projectdirs['Observed'],
                                                                                filename=lcl_filename,
                                                                                rasterfilename='calib_etpat',
                                                                                normalize=normalize,
                                                                                tui=True)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif lcl_lcl_opt == 'aoi_shru_param.txt' or lcl_lcl_opt == 'calib_shru_param.txt':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.get_shru_param(filesp[0], filesp[1],
                                                                           folder=projectdirs['Observed'],
                                                                           filename=lcl_filename)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif lcl_lcl_opt == 'aoi_slope.asc' or lcl_lcl_opt == 'calib_slope.asc':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.map_slope(filesp[0], folder=projectdirs['Observed'],
                                                                      filename=lcl_filename)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif lcl_lcl_opt == 'aoi_fto.asc' or lcl_lcl_opt == 'calib_fto.asc':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.map_fto(filesp[0], filesp[1],
                                                                    folder=projectdirs['Observed'],
                                                                    filename=lcl_filename)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                    #
                    # exit
                    elif lcl_opt == lng[10]:
                        break
            #
            # manage projected datasets
            elif opt == project_options[1]:
                header(opt)
                missing()
            #
            # assessment
            elif opt == project_options[2]:
                header(opt)
                asmt_options = ['CALIB Basin | Hydrology Calibration Assessment',
                                'CALIB Basin | LULC Assessment',
                                'AOI Basin | LULC Assessment',
                                'CALIB Basin | Soils Assessment',
                                'AOI Basin | Soils Assessment',]
                lcl_opt = menu({lng[6]: asmt_options},
                               title='Assessment menu',
                               exitmsg=lng[10],
                               msg=lng[5],
                               keylbl=lng[7],
                               wng=lng[20],
                               wngmsg=lng[8],
                               chsn=lng[9])
                #
                # calibration assessment
                if lcl_opt == asmt_options[0]:
                    header(lcl_opt)
                    #
                    # checker protocol
                    if backend.check_simhydro_files(project_nm, rootdir, aoi=False):
                        warning(wng=lng[20], msg=lng[27])
                        filessim_df = backend.verify_simhydro_files(project_nm, rootdir, aoi=False)
                        print(filessim_df[filessim_df['Status'] == 'missing'].to_string(index=False))
                    else:
                        # get files
                        files_input = backend.get_input2simbhydro(aoi=False)
                        folder = projectdirs['Observed']
                        fseries = folder + '/' + files_input[0]
                        fhydroparam = folder + '/' + files_input[1]
                        fshruparam = folder + '/' + files_input[2]
                        fhistograms = folder + '/' + files_input[3]
                        fbasinhists = folder + '/' + files_input[4]
                        fbasin = folder + '/' + files_input[5]
                        ftwi = folder + '/' + files_input[11] #+ files_input[6]
                        fshru = folder + '/' + files_input[10] # + files_input[7]
                        fcanopy = folder + '/' + files_input[8]
                        fzmaps = folder + '/' + files_input[9]
                        # run slh calib
                        out_dct = tools.hca(fseries=fseries,
                                            fhydroparam=fhydroparam,
                                            fshruparam=fshruparam,
                                            fhistograms=fhistograms,
                                            fbasinhists=fbasinhists,
                                            fbasin=fbasin,
                                            ftwi=ftwi,
                                            fshru=fshru,
                                            fcanopy=fcanopy,
                                            fzmaps=fzmaps,
                                            folder=projectdirs['Assessment'],
                                            wkpl=True,
                                            tui=True,
                                            etpat=True,
                                            label='CALIB')
                # calib LULC assessment
                elif lcl_opt == asmt_options[1]:
                    folder = projectdirs['Observed']
                    fmap = folder + '/calib_lulc.asc'
                    fparams = folder + '/calib_lulc_param.txt'
                    faoi = folder + '/calib_basin.asc'
                    out_dct = tools.qualmap_analyst(fmap=fmap,
                                                    fparams=fparams,
                                                    faoi=faoi,
                                                    kind='lulc',
                                                    label='LULC',
                                                    folder=projectdirs['Assessment'],
                                                    tui=True,
                                                    wkpl=True)
                # AOI LULC assessment
                elif lcl_opt == asmt_options[2]:
                    folder = projectdirs['Observed']
                    fmap = folder + '/aoi_lulc.asc'
                    fparams = folder + '/aoi_lulc_param.txt'
                    faoi = folder + '/aoi_basin.asc'
                    out_dct = tools.qualmap_analyst(fmap=fmap,
                                                    fparams=fparams,
                                                    faoi=faoi,
                                                    kind='lulc',
                                                    label='LULC',
                                                    folder=projectdirs['Assessment'],
                                                    tui=True,
                                                    wkpl=True)
                # calib soils assessment
                elif lcl_opt == asmt_options[3]:
                    folder = projectdirs['Observed']
                    fmap = folder + '/calib_soils.asc'
                    fparams = folder + '/calib_soils_param.txt'
                    faoi = folder + '/calib_basin.asc'
                    out_dct = tools.qualmap_analyst(fmap=fmap,
                                                    fparams=fparams,
                                                    faoi=faoi,
                                                    kind='soils',
                                                    label='SOILS',
                                                    folder=projectdirs['Assessment'],
                                                    tui=True,
                                                    wkpl=True)
                # aoi soils assessment
                elif lcl_opt == asmt_options[4]:
                    folder = projectdirs['Observed']
                    fmap = folder + '/aoi_soils.asc'
                    fparams = folder + '/aoi_soils_param.txt'
                    faoi = folder + '/aoi_basin.asc'
                    out_dct = tools.qualmap_analyst(fmap=fmap,
                                                    fparams=fparams,
                                                    faoi=faoi,
                                                    kind='soils',
                                                    label='SOILS',
                                                    folder=projectdirs['Assessment'],
                                                    tui=True,
                                                    wkpl=True)
            #
            # simulation
            elif opt == project_options[3]:
                while True:
                    header(opt)
                    sim_options = ['CALIB Basin | Observed - Stable LULC Hydrology',
                                   'AOI Basin | Observed - Stable LULC Hydrology',
                                   'AOI Basin | Projected - Stable LULC Hydrolgy']
                    lcl_opt = menu({lng[6]: sim_options},
                                   title='Simulation menu',
                                   exitmsg=lng[10],
                                   msg=lng[5],
                                   keylbl=lng[7],
                                   wng=lng[20],
                                   wngmsg=lng[8],
                                   chsn=lng[9])
                    #
                    # simulate observed policy in calib basin
                    if lcl_opt == sim_options[0]:
                        header(lcl_opt)
                        #
                        # checker protocol
                        if backend.check_simhydro_files(project_nm, rootdir, aoi=False):
                            warning(wng=lng[20], msg=lng[27])
                            filessim_df = backend.verify_simhydro_files(project_nm, rootdir, aoi=False)
                            print(filessim_df[filessim_df['Status'] == 'missing'].to_string(index=False))
                        else:
                            # get files
                            files_input = backend.get_input2simbhydro(aoi=False)
                            folder = projectdirs['Observed']
                            fseries = folder + '/' + files_input[0]
                            fhydroparam = folder + '/' + files_input[1]
                            fshruparam = folder + '/' + files_input[2]
                            fhistograms = folder + '/' + files_input[3]
                            fbasinhists = folder + '/' + files_input[4]
                            fbasin = folder + '/' + files_input[5]
                            ftwi = folder + '/' + files_input[6]
                            fshru = folder + '/' + files_input[7]
                            ftwi_window = folder + '/' + files_input[11]
                            fshru_window = folder + '/' + files_input[10]
                            fcanopy = folder + '/' + files_input[8]
                            #
                            # get settings
                            settings = settings_simulation(lng[6])
                            slicedates = slice_series(fseries=fseries)
                            #
                            # run slh
                            out_dct = tools.slh(fseries=fseries,
                                                fhydroparam=fhydroparam,
                                                fshruparam=fshruparam,
                                                fhistograms=fhistograms,
                                                fbasinhists=fbasinhists,
                                                fbasin=fbasin,
                                                ftwi=ftwi_window,
                                                fshru=fshru_window,
                                                fcanopy=fcanopy,
                                                folder=projectdirs['Simulation'],
                                                integrate=settings['Integrate'],
                                                wkpl=True,
                                                tui=True,
                                                qobs=True,
                                                mapback=settings['Mapback'],
                                                mapraster=settings['Mapraster'],
                                                slicedates=slicedates,
                                                label='CALIB')
                            #
                            # export local pannels
                            if settings['Frametype'] != 'Skip':
                                tools.export_local_pannels(ftwi_window, fshru_window,
                                                           folder=out_dct['Folder'],
                                                           frametype=settings['Frametype'],
                                                           tui=True)
                    #
                    # simulate observed policy in AOI basin
                    elif lcl_opt == sim_options[1]:
                        header(lcl_opt)
                        #
                        # checker protocol
                        if backend.check_simhydro_files(project_nm, rootdir, aoi=True):
                            warning(wng=lng[20], msg=lng[27])
                            filessim_df = backend.verify_simhydro_files(project_nm, rootdir, aoi=True)
                            print(filessim_df[filessim_df['Status'] == 'missing'].to_string(index=False))
                        else:
                            #
                            # get files
                            files_input = backend.get_input2simbhydro(aoi=True)
                            folder = projectdirs['Observed']
                            fseries = folder + '/' + files_input[0]
                            fhydroparam = folder + '/' + files_input[1]
                            fshruparam = folder + '/' + files_input[2]
                            fhistograms = folder + '/' + files_input[3]
                            fbasinhists = folder + '/' + files_input[4]
                            fbasin = folder + '/' + files_input[5]
                            ftwi = folder + '/' + files_input[6]
                            fshru = folder + '/' + files_input[7]
                            fcanopy = folder + '/' + files_input[8]
                            ftwi_window = folder + '/' + files_input[10]
                            fshru_window = folder + '/' + files_input[9]
                            #
                            # get settings
                            settings = settings_simulation(lng[6])
                            slicedates = slice_series(fseries=fseries)
                            #
                            #
                            #
                            # run slh
                            out_dct = tools.slh(fseries=fseries,
                                                fhydroparam=fhydroparam,
                                                fshruparam=fshruparam,
                                                fhistograms=fhistograms,
                                                fbasinhists=fbasinhists,
                                                fbasin=fbasin,
                                                ftwi=ftwi,
                                                fshru=fshru,
                                                fcanopy=fcanopy,
                                                folder=projectdirs['Simulation'],
                                                integrate=settings['Integrate'],
                                                wkpl=True,
                                                tui=True,
                                                mapback=settings['Mapback'],
                                                mapraster=settings['Mapraster'],
                                                slicedates=slicedates,
                                                label='AOI',
                                                aoi=True)
                            #
                            # export local pannels
                            if settings['Frametype'] != 'Skip':
                                tools.export_local_pannels(ftwi_window, fshru_window,
                                                           folder=out_dct['Folder'],
                                                           frametype=settings['Frametype'],
                                                           tui=True)
                    #
                    # simulate projected policy
                    elif lcl_opt == sim_options[2]:
                        # todo list available scenarios to choose
                        header(lcl_opt)
                        #
                        # checker protocol
                        if backend.check_simhydro_files(project_nm, rootdir, aoi=True):
                            warning(wng=lng[20], msg=lng[27])
                            filessim_df = backend.verify_simhydro_files(project_nm, rootdir, aoi=True)
                            print(filessim_df[filessim_df['Status'] == 'missing'].to_string(index=False))
                        else:
                            #
                            # get files
                            # todo load files from the scenario chosen
                            files_input = backend.get_input2simbhydro(aoi=True)
                            folder = projectdirs['Observed']
                            fseries = folder + '/' + files_input[0]
                            fhydroparam = folder + '/' + files_input[1]
                            fshruparam = folder + '/' + files_input[2]
                            fhistograms = 'C:/000_myFiles/myDrive/Plans3/pardinho/datasets/projected/scn__lulc_predc/histograms.txt'
                            fbasinhists = 'C:/000_myFiles/myDrive/Plans3/pardinho/datasets/projected/scn__lulc_predc/basin_histograms.txt'
                            fbasin = folder + '/' + files_input[5]
                            ftwi = folder + '/' + files_input[6]
                            fshru = 'C:/000_myFiles/myDrive/Plans3/pardinho/datasets/projected/scn__lulc_predc/shru.asc'
                            fcanopy = folder + '/' + files_input[8]
                            ftwi_window = folder + '/' + files_input[10]
                            fshru_window = folder + '/' + files_input[9]
                            #
                            # get settings
                            settings = settings_simulation(lng[6])
                            slicedates = slice_series(fseries=fseries)
                            #
                            #
                            #
                            # run slh
                            out_dct = tools.slh(fseries=fseries,
                                                fhydroparam=fhydroparam,
                                                fshruparam=fshruparam,
                                                fhistograms=fhistograms,
                                                fbasinhists=fbasinhists,
                                                fbasin=fbasin,
                                                ftwi=ftwi,
                                                fshru=fshru,
                                                fcanopy=fcanopy,
                                                folder=projectdirs['Simulation'],
                                                integrate=settings['Integrate'],
                                                wkpl=True,
                                                tui=True,
                                                mapback=settings['Mapback'],
                                                mapraster=settings['Mapraster'],
                                                slicedates=slicedates,
                                                label='predc_AOI',
                                                aoi=True)
                            # todo post a checker here
                            #
                            # export local pannels
                            if settings['Frametype'] != 'Skip':
                                tools.export_local_pannels(ftwi_window, fshru_window,
                                                           folder=out_dct['Folder'],
                                                           frametype=settings['Frametype'],
                                                           tui=True)
                    #
                    # exit
                    elif lcl_opt == lng[10]:
                        break
            #
            # optimization
            elif opt == project_options[4]:
                while True:
                    header(opt)
                    optimize_options = ('Calibrate CALIB Basin Hydrology',
                                        'Calibrate Population Model',
                                        'Optimize LULC Policy')
                    lcl_opt = menu({'Model': optimize_options},
                                   exitmsg=lng[10],
                                   msg=lng[5],
                                   keylbl=lng[7],
                                   wng=lng[20],
                                   wngmsg=lng[8],
                                   chsn=lng[9])
                    # exit menu condition
                    if lcl_opt == lng[10]:
                        break
                    #
                    # calibrate hydrology:
                    elif lcl_opt == optimize_options[0]:
                        # checker protocol
                        if backend.check_calibhydro_files(project_nm, rootdir):
                            warning(wng=lng[20], msg=lng[27])
                            filesclib_df = backend.verify_calibhydro_files(project_nm, rootdir)
                            print(filesclib_df[filesclib_df['Status'] == 'missing'].to_string(index=False))
                        else:
                            while True:
                                header(lcl_opt)
                                type_options = ('Flow and ET', 'Flow only')
                                runtype = menu({'Type': type_options},
                                                  title='Calibration Menu',
                                                  exitmsg=lng[10],
                                                  msg=lng[5],
                                                  keylbl=lng[7],
                                                  wng=lng[20],
                                                  wngmsg=lng[8],
                                                  chsn=lng[9])
                                # exit menu condition
                                if runtype == lng[10]:
                                    break
                                metrics_options = ('NSE',
                                                   'NSElog',
                                                   'RMSE',
                                                   'RMSElog',
                                                   'KGE',
                                                   'KGElog',
                                                   'PBias',
                                                   'RMSE-CFC',
                                                   'RMSElog-CFC')
                                likelihood = menu({'Likelihood': metrics_options},
                                                  title='Flow Likelihood Menu',
                                                  exitmsg=lng[10],
                                                  msg=lng[5],
                                                  keylbl=lng[7],
                                                  wng=lng[20],
                                                  wngmsg=lng[8],
                                                  chsn=lng[9])
                                # exit menu condition
                                if likelihood == lng[10]:
                                    break
                                else:
                                    files_input = backend.get_input2calibhydro()
                                    folder = projectdirs['Observed']
                                    fseries = folder + '/' + files_input[0]
                                    fhydroparam = folder + '/' + files_input[1]
                                    fshruparam = folder + '/' + files_input[2]
                                    fhistograms = folder + '/' + files_input[3]
                                    fbasinhists = folder + '/' + files_input[4]
                                    fbasin = folder + '/' + files_input[5]
                                    ftwi = folder + '/' + files_input[11] # + files_input[6]
                                    fshru = folder + '/'  + files_input[10] # + files_input[7]
                                    fetpatzmaps = folder + '/' + files_input[8]
                                    fcanopy = folder + '/' + files_input[9]
                                    #
                                    if runtype == 'Flow Only':
                                        etpat = False
                                    elif runtype == 'Flow and ET':
                                        etpat = True
                                    else:
                                        etpat = False
                                    #
                                    aux_str = 'calib_hydro' + '_' + likelihood
                                    #
                                    size_opts = ('Very Small - Size:10 Gens:3',
                                                 'Small - Size:25 Gens:5',
                                                 'Medium - Size:250 Gens:5',
                                                 'Large - Size:500 Gens:10',
                                                 'Very Large - Size:2000 Gens:10')
                                    scale = menu({'Scale': size_opts},
                                                 exitkey='d',
                                                 title='Genetic Algorithm Scale',
                                                 exitmsg='Use default (Small)',
                                                 msg=lng[5],
                                                 keylbl=lng[7],
                                                 wng=lng[20],
                                                 wngmsg=lng[8],
                                                 chsn=lng[9])
                                    popsize = 10
                                    generations = 3
                                    if scale == size_opts[0]:
                                        popsize = 10
                                        generations = 3
                                    elif scale == size_opts[1]:
                                        popsize = 25
                                        generations = 5
                                    elif scale == size_opts[2]:
                                        popsize = 250
                                        generations = 5
                                    elif scale == size_opts[3]:
                                        popsize = 500
                                        generations = 10
                                    elif scale == size_opts[4]:
                                        popsize = 2000
                                        generations = 10
                                    #
                                    #
                                    # run calibration tool
                                    calibfiles = tools.calibrate(fseries=fseries,
                                                                 fhydroparam=fhydroparam,
                                                                 fshruparam=fshruparam,
                                                                 fhistograms=fhistograms,
                                                                 fbasinhists=fbasinhists,
                                                                 fbasin=fbasin,
                                                                 fetpatzmaps=fetpatzmaps,
                                                                 ftwi=ftwi,
                                                                 fshru=fshru,
                                                                 fcanopy=fcanopy,
                                                                 folder=projectdirs['Optimization'],
                                                                 label='calib',
                                                                 generations=generations,
                                                                 popsize=popsize,
                                                                 likelihood=likelihood,
                                                                 tui=True,
                                                                 normalize=False,
                                                                 etpat=etpat,
                                                                 cutdatef=0.2,
                                                                 tail=True)
                                    print('\nRun files sucessfully created at:\n{}\n'.format(calibfiles['Folder']))
                                    ok()
                    #
                    # population
                    elif lcl_opt == optimize_options[1]:
                        header(lcl_opt)
                        missing()
                    #
                    # optimize lulc polilcy
                    elif lcl_opt == optimize_options[2]:
                        header(lcl_opt)
                        missing()
            #
            # exit
            elif opt == lng[10]:
                break
        #
        # evaluate exit
        if exit_flag:
            break
