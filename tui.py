import pandas as pd
import numpy as np
import backend, tools
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
    built plans 2 header message
    :return: string with plans header msg
    """
    def_str0 = 'plans - planning nature-based solutions'.upper()
    def_str1 = 'Version: 3.0'
    def_str3 = 'This software is under the GNU GPL3.0 license'
    def_str4 = 'Source code repository: https://github.com/ipo-exe/plans/'
    def_str = def_str0 + '\n' + def_str1 + '\n' + def_str3 + '\n' + def_str4 + '\n\n'
    return def_str


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
            index = int(chosen) - 1
            if chosen in keys_str:
                ok()
                print(chsn + ':\t' + options_prime[index] + '\n')
                break
            else:
                warning(wng=wng, msg=wngmsg)
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


def main():
    # print header
    print(header_warp())  # warp header credentials
    sleep(0.3)
    print(header_plans())  # plans header credentials
    sleep(0.3)
    #
    # get root directory
    rootdir = backend.get_root_dir()  # project standard directory on local machine
    #
    # load dictionary
    dicionary = pd.read_csv('./dictionary.txt', sep=';')
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
            # observed datasets
            if opt == project_options[0]:
                while True:
                    header(lng[14])
                    files_df = backend.verify_observed_files(project_nm, rootdir)
                    print(files_df.to_string(index=False))
                    print('\n')
                    observed_options = [lng[24], lng[25], lng[26]]
                    opt = menu({lng[6]:observed_options}, title='', exitmsg=lng[10], msg=lng[5],
                           keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
                    # import datasets
                    if opt == observed_options[0]:
                        # import menu loop
                        while True:
                            header(observed_options[0])
                            inputfiles_df = files_df[files_df['Type'] == 'imported']
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
                    # derive data
                    elif opt == observed_options[1]:
                        while True:
                            header(observed_options[1])
                            derivefiles_df = files_df[files_df['Type'] == 'derived']
                            files_lst = list(derivefiles_df['File'])
                            opt = menu({lng[6]: files_lst}, title='', exitmsg=lng[10],
                                       msg=lng[5], keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
                            # exit menu condition
                            if opt == lng[10]:
                                break
                            # derive file
                            else:
                                filesderiv_df = backend.verify_input2derived(derived=opt, p0=project_nm, wkplc=rootdir)

                                if backend.check_input2derived(derived=opt, p0=project_nm, wkplc=rootdir):
                                    warning(wng=lng[20], msg=lng[27])
                                    print(filesderiv_df.to_string(index=False))
                                else:
                                    # get paths
                                    filesnames = list(filesderiv_df['File'].values)
                                    print(filesnames)
                                    filesp = list()
                                    for i in range(len(filesnames)):
                                        filesp.append(projectdirs['Observed'] + '/' + filesnames[i])
                                    # evaluate options
                                    if opt == 'cn.asc':
                                        derivedfile = tools.map_cn(filesp[0], filesp[1], filesp[2], filesp[3],
                                                                   folder=projectdirs['Observed'])
                                        print('\nFile sucessfully created at: {}'.format(derivedfile))
                                        ok()
                                    elif opt == 'grad.asc':
                                        derivedfile = tools.map_grad(filesp[0], folder=projectdirs['Observed'])
                                        print('\nFile sucessfully created at: {}'.format(derivedfile))
                                        ok()
                                    elif opt == 'lulc_series.txt':
                                        derivedfile = tools.import_lulc_series(filesp[0],
                                                                               rasterfolder=projectdirs['LULC'],
                                                                               folder=projectdirs['Observed'])
                                        print('\nFile sucessfully created at: {}'.format(derivedfile))
                                        ok()
                                    elif opt == 'lulc_areas.txt':
                                        derivedfile = tools.lulc_areas(filesp[0], filesp[1], filesp[2],
                                                                       folder=projectdirs['Observed'])
                                        print('\nFile sucessfully created at: {}'.format(derivedfile))
                                        ok()
                    # calibrate models
                    elif opt == observed_options[2]:
                        # check if all input data is present
                        check1 = backend.check_inputfiles(project_nm, rootdir, 'imported')
                        check2 = backend.check_inputfiles(project_nm, rootdir, 'derived')
                        if check1 or check2:
                            files_df = backend.verify_observed_files(project_nm, rootdir)
                            warning(wng=lng[20], msg=lng[27])
                            qdf = files_df.query('Type =="imported" or Type =="derived" and Status == "missing"')
                            print('>>> ' + lng[28] + ':')
                            print(qdf.to_string(index=False))
                            proceed(msg=lng[29])
                        else:
                            header(observed_options[2])
                            print('\n>>> RUN calibration')
                            calib_options = ('Hydrology model', '')

                    # exit
                    elif opt == lng[10]:
                        break
            # projected datasets
            elif opt == project_options[1]:
                header(lng[15])
                print('Projected datasets')
            # simulate policy
            elif opt == project_options[2]:
                header(lng[16])
                print('Simulate policy')
            # optimize policy
            elif opt == project_options[3]:
                header(lng[17])
                print('Optimize policy')
            elif opt == lng[10]:
                break
        if exit_flag:
            break
