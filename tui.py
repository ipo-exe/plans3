import pandas as pd
import numpy as np
import backend, tools
from backend import center, header_warp, header_plans
from time import sleep
from tkinter import filedialog
from tkinter import *


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


def menu(options, title='Menu', msg='Chose key', exit=True, exitkey='e', exitmsg='Exit menu',
         keylbl='Keys', wng='Warning!', wngmsg='Key not found', chsn='Chosen'):
    """
    Display a Menu
    :param options: iterable with string options
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
                index = int(chosen) - 1
                ok()
                print(chsn + ':\t' + options_prime[index] + '\n')
                break

    if exit_flag:
        return exitmsg
    else:
        return options_prime[index]


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
    #
    # load dictionary
    dicionary = pd.read_csv('./dictionary.txt', sep=';', engine='python', encoding='utf-8')
    lng = dicionary.T.values[0]  # array of language strings
    languages = tuple(dicionary.columns)  # tuple of languages
    #
    # enter session loop:
    while True:
        exit_flag = False
        # PLANS Menu loop:
        while True:
            header('PLANS 3')
            session_options = (lng[1], lng[2], lng[3])
            opt = menu({lng[6]:session_options}, title=lng[0], exitmsg= lng[4], msg=lng[5], keylbl=lng[7],
                       wng=lng[20], wngmsg=lng[8], chsn=lng[9])
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
                        project_nm = menu({lng[6]: projects_names}, exitmsg=lng[10], title=lng[23], msg=lng[5],
                                          keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
                        break
                if project_nm == lng[10]:
                    pass
                else:
                    break
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
            # reset language
            elif opt == session_options[2]:
                header(lng[3])
                lang_options = list(languages[:])
                opt = menu({lng[6]:lang_options}, title=lng[11],  exitmsg=lng[10], msg=lng[5],
                       keylbl=lng[7], wng=lng[8], chsn=lng[9])
                if opt == lng[10]:
                    pass
                else:
                    lng_id = languages.index(opt)
                    lng = dicionary.T.values[lng_id]
            # exit program
            elif opt == lng[4]:
                exit_flag = True
                break
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
            print(projectdirs['Project'] + '\n')
            project_options = [lng[14], lng[15], lng[16], lng[17]]
            opt = menu({lng[6]:project_options}, title=lng[13], exitmsg=lng[10], msg=lng[5],
                       keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
            #
            # observed datasets
            if opt == project_options[0]:
                while True:
                    header(lng[14])
                    files_df = backend.verify_observed_files(project_nm, rootdir)
                    print(files_df.to_string(index=False))
                    print('\n')
                    if importing:
                        observed_options = [lng[24], lng[25], 'Analyse data', lng[26]]
                    else:
                        observed_options = [lng[25], 'Analyse data', lng[26]]
                    opt = menu({lng[6]:observed_options}, title='', exitmsg=lng[10], msg=lng[5],
                           keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
                    #
                    # import datasets
                    if opt == lng[24]:
                        # import menu loop
                        while True:
                            header(lng[24])
                            inputfiles_df = files_df[files_df['Type'] == 'input']  #
                            files_lst = list(inputfiles_df['File'])
                            status_lst = list(inputfiles_df['Status'])
                            opt = menu({lng[6]: files_lst, 'Status':status_lst}, title='', exitmsg=lng[10],
                                       msg=lng[5], keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
                            # exit menu condition
                            if opt == lng[10]:
                                break
                            # import file
                            else:
                                extension = opt.split('.')[-1]
                                aux_str = '.' + extension + ' file'
                                src_filenm = pick_file(p0=extension, p1=aux_str)
                                if src_filenm == 'cancel':
                                    pass
                                else:
                                    dst_filenm = projectdirs['Observed'] + '/' + opt
                                    # copy and paste
                                    backend.importfile(src_filenm, dst_filenm)
                                    # update database
                                    files_df = backend.verify_observed_files(project_nm, rootdir)
                                    print('\n{}:\n{}\n'.format(lng[30], dst_filenm))
                                    tools.view_imported(opt, folder=projectdirs['Observed'])
                                    ok()
                    #
                    # derive data
                    elif opt == lng[25]:
                        while True:
                            header(lng[25])
                            derivefiles_df = files_df[files_df['Type'] == 'derived']
                            files_lst = list(derivefiles_df['File'])
                            opt = menu({lng[6]: files_lst}, title='', exitmsg=lng[10],
                                       msg=lng[5], keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
                            #
                            # exit menu condition
                            if opt == lng[10]:
                                break
                            #
                            # derive file
                            else:
                                #
                                # get input needs
                                filesderiv_df = backend.verify_input2derived(derived=opt, p0=project_nm, wkplc=rootdir)
                                #
                                # check missing files
                                if backend.check_input2derived(derived=opt, p0=project_nm, wkplc=rootdir):
                                    warning(wng=lng[20], msg=lng[27])
                                    print(filesderiv_df.to_string(index=False))
                                else:
                                    # get files names
                                    filesnames = list(filesderiv_df['File'].values)
                                    files_used_df = pd.DataFrame({'Files used':filesnames})
                                    print(files_used_df.to_string(index=False))
                                    #
                                    # get files paths
                                    filesp = list()
                                    for i in range(len(filesnames)):
                                        filesp.append(projectdirs['Observed'] + '/' + filesnames[i])
                                    #
                                    # evaluate options
                                    #
                                    # Derive TWI
                                    if opt == 'calib_twi.asc' or opt == 'aoi_twi.asc':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.map_twi(filesp[0], filesp[1], filesp[2],
                                                                    folder=projectdirs['Observed'],
                                                                    filename=opt.split('.')[0])
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif opt == 'calib_shru.asc':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.map_shru(filesp[0], filesp[1], filesp[2],
                                                                     filesp[3], filesp[4],
                                                                     folder=projectdirs['Observed'],
                                                                     filename=opt.split('.')[0])
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif opt == 'calib_histograms.txt':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.compute_histograms(filesp[0], filesp[1], filesp[2],
                                                                               filesp[3], folder=projectdirs['Observed'],
                                                                               filename=opt.split('.')[0], tui=True)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif opt == 'calib_etpat_zmaps.txt':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.compute_zmap_series(filesp[0], filesp[1], filesp[2],
                                                                                filesp[3], filesp[4],
                                                                                folder=projectdirs['Observed'],
                                                                                filename=opt.split('.')[0], tui=True)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif opt == 'aoi_lulc_series.txt':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.import_map_series(filesp[0],
                                                                               rasterfolder=projectdirs['LULC'],
                                                                               folder=projectdirs['Observed'],
                                                                               filename=opt.split('.')[0],
                                                                               rasterfilename='aoi_lulc')
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif opt == 'calib_etpat_series.txt':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.import_map_series(filesp[0],
                                                                              rasterfolder=projectdirs['ETpat'],
                                                                              folder=projectdirs['Observed'],
                                                                              filename=opt.split('.')[0],
                                                                              rasterfilename='calib_etpat')
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif opt == 'aoi_shru_series.txt':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.import_shru_series(filesp[0], filesp[1], filesp[2],
                                                                               filesp[3], filesp[4],
                                                                               rasterfolder=projectdirs['SHRU'],
                                                                               folder=projectdirs['Observed'],
                                                                               filename=opt.split('.')[0],
                                                                               suff='aoi', tui=True)
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif opt == 'aoi_shru_param.txt' or opt == 'calib_shru_param.txt':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.shru_param(filesp[0], filesp[1],
                                                                       folder=projectdirs['Observed'],
                                                                       filename=opt.split('.')[0])
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif opt == 'aoi_slope.asc' or opt == 'calib_slope.asc':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.map_slope(filesp[0], folder=projectdirs['Observed'],
                                                                      filename=opt.split('.')[0])
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                                    elif opt == 'aoi_fto.asc' or opt == 'calib_fto.asc':
                                        print('\n' + lng[31] + '...')
                                        derivedfile = tools.map_fto(filesp[0], filesp[1],
                                                                    folder=projectdirs['Observed'],
                                                                    filename=opt.split('.')[0])
                                        print('\n{}:\n{}\n'.format(lng[30], derivedfile))
                                        ok()
                    #
                    # analyse data
                    elif opt == 'Analyse data':
                        while True:
                            header('Analyse data')
                            analysis = ('LULC Analysis | AOI', 'Soil Analysis | AOI', 'SHRU Analysis | AOI',
                                        'DEM Analysis | AOI', 'Series Analysis | AOI', 'LULC Analysis | CALIB',
                                        'Soil Analysis | CALIB', 'SHRU Analysis | CALIB', 'DEM Analysis | CALIB',
                                        'Series Analysis | CALIB')
                            opt = menu({lng[6]: analysis}, title='Analysis options', exitmsg=lng[10],
                                       msg=lng[5], keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
                            if opt == lng[10]:
                                break
                            # lulc AOI
                            elif opt == analysis[0]:
                                print('Develop code!')
                    # calibrate models
                    elif opt == lng[26]:
                        while True:
                            header(lng[26])
                            calib_options = ('Calibrate Hydrology', 'Calibrate Population')
                            opt = menu({'Model':calib_options}, exitmsg=lng[10], msg=lng[5], keylbl=lng[7], wng=lng[20],
                                       wngmsg=lng[8], chsn=lng[9])
                            # exit menu condition
                            if opt == lng[10]:
                                break
                            # calibrate hydrology:
                            elif opt == calib_options[0]:
                                # checker protocol
                                if backend.check_calibhydro_files(project_nm, rootdir):
                                    warning(wng=lng[20], msg=lng[27])
                                    filesclib_df = backend.verify_calibhydro_files(project_nm, rootdir)
                                    print(filesclib_df[filesclib_df['Status'] == 'missing'].to_string(index=False))
                                else:
                                    while True:
                                        header(calib_options[0])
                                        metrics_options = ('NSE', 'NSElog', 'RMSE', 'RMSElog', 'KGE', 'KGElog', 'PBias',
                                                           'RMSE-CFC', 'RMSElog-CFC')
                                        metric = menu({'Metric':metrics_options}, exitmsg=lng[10], msg=lng[5],
                                                      keylbl=lng[7],
                                                   wng=lng[20], wngmsg=lng[8], chsn=lng[9])
                                        # exit menu condition
                                        if metric == lng[10]:
                                            break
                                        else:
                                            files_input = backend.get_input2calibhydro()
                                            folder = projectdirs['Observed']
                                            fseries = folder + '/' + files_input[0]
                                            fhydroparam = folder + '/' + files_input[1]
                                            fshruparam = folder + '/' + files_input[2]
                                            fhistograms = folder + '/' + files_input[3]
                                            fbasin = folder + '/' + files_input[4]
                                            fetpatzmaps = folder + '/' + files_input[5]
                                            aux_str = 'calib_hydro' + '_' + metric
                                            dst_dir = backend.create_rundir(label=aux_str,
                                                                            wkplc=projectdirs['Optimization'])
                                            size_opts = ( 'Very Small - Size:12 Gens:2', 'Small - Size:25 Gens:5', 'Medium - Size:50 Gens:50',
                                                             'Large - Size:100 Gens:200')
                                            scale = menu({'Scale': size_opts}, exitkey='d',
                                                         exitmsg='Use default (Small)', msg=lng[5],
                                                        keylbl=lng[7],wng=lng[20], wngmsg=lng[8], chsn=lng[9])
                                            popsize = 12
                                            generations = 3
                                            if scale == size_opts[0]:
                                                popsize = 12
                                                generations = 3
                                            elif scale == size_opts[1]:
                                                popsize = 25
                                                generations = 5
                                            elif scale == size_opts[2]:
                                                popsize = 100
                                                generations = 50
                                            elif scale == size_opts[3]:
                                                popsize = 500
                                                generations = 200
                                            calibparam = tools.calib_hydro(fseries=fseries, fhydroparam=fhydroparam,
                                                                           fshruparam=fshruparam,
                                                                           fhistograms=fhistograms, fbasin=fbasin,
                                                                           fetpatzmaps=fetpatzmaps , folder=dst_dir,
                                                                           generations=generations, popsize=popsize,
                                                                           metric=metric, tui=True)
                                            print('\n{}:\n{}\n'.format(lng[30], calibparam))
                                            ok()
                            elif opt == calib_options[1]:
                                header(calib_options[1])
                                print('missing code!')
                    #
                    # exit
                    elif opt == lng[10]:
                        break
            #
            # projected datasets
            elif opt == project_options[1]:
                header(lng[15])
                print('Projected datasets')
            #
            # simulate nbs policy
            elif opt == project_options[2]:
                while True:
                    header(lng[16])
                    sim_options = ['Simulate CALIB hydrology', 'AOI observed policy', 'AOI projected policy']
                    opt = menu({lng[6]: sim_options}, title='Simulation menu', exitmsg=lng[10], msg=lng[5],
                               keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
                    # simulate observed policy
                    if opt == sim_options[0]:
                        header(opt)
                        # simulate hydrology
                        if opt == 'Simulate CALIB hydrology':
                            # checker protocol
                            header(lng[32])
                            if backend.check_simhydro_files(project_nm, rootdir, aoi=False):
                                warning(wng=lng[20], msg=lng[27])
                                filessim_df = backend.verify_simhydro_files(project_nm, rootdir, aoi=False)
                                print(filessim_df[filessim_df['Status'] == 'missing'].to_string(index=False))
                            else:
                                files_input = backend.get_input2simbhydro(aoi=False)
                                folder = projectdirs['Observed']
                                fseries = folder + '/' + files_input[0]
                                fhydroparam = folder + '/' + files_input[1]
                                fshruparam = folder + '/' + files_input[2]
                                fhistograms = folder + '/' + files_input[3]
                                fbasin = folder + '/' + files_input[4]
                                dst_dir = backend.create_rundir(label='sim_hydro', wkplc=projectdirs['Simulation'])
                                files = tools.stable_lulc_hydro(fseries=fseries, fhydroparam=fhydroparam,
                                                                fshruparam=fshruparam, fhistograms=fhistograms,
                                                                fbasin=fbasin,mapvar='D-ET-R-Inf-Tpun-Tpgw', qobs=True,
                                                                folder=dst_dir, tui=True, mapback=True)
                                files_analyst = tools.obs_sim_analyst(fseries=files[0], fld_obs='Qobs', fld_sim='Q',
                                                                      folder=dst_dir, tui=True)
                    # simulate observed policy
                    elif opt == sim_options[1]:
                        header(opt)
                        print('develop code')
                    # simulate projected policy
                    elif opt == sim_options[1]:
                        header(opt)
                        print('develop code')
                    elif opt == lng[10]:
                        break
            #
            # optimize nbs policy
            elif opt == project_options[3]:
                header(opt)
                print('develop code')
            elif opt == lng[10]:
                break
        if exit_flag:
            break
