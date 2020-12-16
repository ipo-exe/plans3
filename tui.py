import pandas as pd
import numpy as np

def ok():
    print('>>> OK\t\t')

def header(title='Title'):
    print('\n\n')
    print('*' * 20 + ' ' + title.upper() + ' ' + '*' * 20 )
    print('\n\n')

def menu(options, title='Menu', msg='Chose key', optlbl='Options',
         keylbl='Keys', wng='Warning! Key not found', chsn='Chosen'):
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
    print(menu_str)
    print('')
    while True:
        chosen = input('>>> ' + msg + ': ').strip()
        index = int(chosen) - 1
        if chosen in keys_str:
            ok()
            print(chsn + ':\t' + options[index])
            print('')
            break
        else:
            print(wng)
            print('')
    return options[index]


def main():
    # print header

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
                       keylbl=lng[7], wng=lng[8], chsn=lng[9])
            if opt == session_options[0]:
                print('Open')
            elif opt == session_options[1]:
                print('New')
            elif opt == session_options[2]:
                # reset language
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
            elif opt == session_options[3]:
                exit_flag = True
                break

        # Project Menu:


        if exit_flag:
            break

main()
