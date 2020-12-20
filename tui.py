import pandas as pd
import numpy as np
import backend
from time import sleep
from tkinter import filedialog
from tkinter import *


def ok():
    print('\n\t>>> OK\n')
    sleep(0.2)


def warning(wng='Warning!', msg=''):
    print('\n\t{} {}\n'.format(wng, msg))
    sleep(0.3)


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


def menu(options, title='Menu', msg='Chose key', optlbl='Options',
         keylbl='Keys', wng='Warning!', wngmsg='Key not found', chsn='Chosen'):
    """
    Display a Menu
    :param options: iterable with string options
    :param title: title string
    :param msg: message string
    :param optlbl: option label string
    :param keylbl: keys label string
    :param wng: not found warning string
    :param chsn: chosen string
    :return: string of option chosen
    """
    keys = np.arange(1, len(options) + 1)
    keys_str = keys.astype(str)
    def_df = pd.DataFrame({optlbl:options, keylbl:keys})
    menu_str = def_df.to_string(index=False)
    lcl_len = len(menu_str.split('\n')[0])
    print(title)
    print('_' * lcl_len)
    print(menu_str + '\n')
    while True:
        chosen = input('>>> ' + msg + ': ').strip()
        index = int(chosen) - 1
        if chosen in keys_str:
            ok()
            print(chsn + ':\t' + options[index] + '\n')
            break
        else:
            warning(wng=wng, msg=wngmsg)
    return options[index]


def pick_dir():
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected


def main():
    # print header
    print(header_warp())
    sleep(0.3)
    print(header_plans())
    sleep(0.3)
    #
    # get root directory
    rootdir = backend.get_root_dir()
    #
    # load dictionary
    dicionary = pd.read_csv('./dictionary.txt', sep=';')
    lng = dicionary.T.values[0]
    languages = tuple(dicionary.columns)
    #
    # enter session loop:
    while True:
        exit_flag = False
        # PLANS Menu:
        while True:
            header('PLANS 3')
            session_options = (lng[1], lng[2], lng[3], lng[4])
            opt = menu(session_options, title=lng[0], msg=lng[5], optlbl=lng[6],
                       keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
            # open project
            if opt == session_options[0]:
                header(lng[1])
                projects_df = backend.get_existing_projects(wkplc=rootdir)
                projects_names = tuple(projects_df['Name'])
                project_nm = menu(projects_names, title=lng[23], msg=lng[5], optlbl=lng[6],
                                  keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
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
                lang_options.append(lng[10])
                opt = menu(lang_options, title=lng[11], msg=lng[5], optlbl=lng[6],
                       keylbl=lng[7], wng=lng[8], chsn=lng[9])
                if opt == lng[10]:
                    pass
                else:
                    lng_id = languages.index(opt)
                    lng = dicionary.T.values[lng_id]
            # exit program
            elif opt == session_options[3]:
                exit_flag = True
                break
        # evaluate exit
        if exit_flag:
            break
        # Project Setup
        header(lng[12] + ': ' + project_nm)
        projectdirs = backend.get_prj_dirs_paths(p0=project_nm, wkplc=rootdir)
        print(projectdirs['Project'])
        # Project Menu:
        while True:
            project_options = [lng[14], lng[15], lng[16], lng[17], lng[10]]
            opt = menu(project_options, title=lng[13], msg=lng[5], optlbl=lng[6],
                       keylbl=lng[7], wng=lng[20], wngmsg=lng[8], chsn=lng[9])
            # observed datasets
            if opt == project_options[0]:
                header(lng[14])
                print('Observed datasets')
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
            elif opt == project_options[4]:
                break

        if exit_flag:
            break
