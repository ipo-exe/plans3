'''
PLANS - Planning Nature-based Solutions

Output routines.

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
import numpy as np
import pandas as pd


def asc_raster(array, meta, folder, filename, dtype='float32'):
    """
    Function for exporting an .ASC raster file.
    :param array: 2d numpy array
    :param meta: dicitonary of metadata. Example:

    {'ncols': 366,
     'nrows': 434,
      'xllcorner': 559493.087150689564,
       'yllcorner': 6704832.279550871812,
        'cellsize': 28.854232957826,
        'NODATA_value': -9999}

    :param folder: string of directory path
    :param filename: string of file without extension
    :param dtype: string code of data type
    :return: full file name (path and extension) string
    """
    meta_lbls = ('ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value')
    ndv = float(meta['NODATA_value'])
    exp_lst = list()
    for i in range(len(meta_lbls)):
        line = '{}    {}\n'.format(meta_lbls[i], meta[meta_lbls[i]])
        exp_lst.append(line)
    # print(exp_lst)
    #
    # data constructor loop:
    def_array = np.array(array, dtype=dtype)
    for i in range(len(def_array)):
        # replace np.nan to no data values
        lcl_row_sum = np.sum((np.isnan(def_array[i])) * 1)
        if lcl_row_sum > 0:
            #print('Yeas')
            for j in range(len(def_array[i])):
                if np.isnan(def_array[i][j]):
                    def_array[i][j] = int(ndv)
        str_join = ' ' + ' '.join(np.array(def_array[i], dtype='str')) + '\n'
        exp_lst.append(str_join)

    flenm = folder + '/' + filename + '.asc'
    fle = open(flenm, 'w+')
    fle.writelines(exp_lst)
    fle.close()
    return flenm


def zmap(zmap, twibins, shrubins, index_lbl='TWI\SHRU', folder='C:/bin', filename='zmap'):
    """
    Export a zmap file
    :param zmap: 2d numpy array of zmap
    :param twibins: 1d numpy array of TWI classes
    :param shrubins: 1d numpy array of SHRU classes
    :param index_lbl: string of index label
    :param folder: output folder
    :param filename: file name without extension
    :return: output file path
    """
    exp_file = folder + '/' + filename + '.txt'
    lcl_exp_df = pd.DataFrame(zmap, index=twibins, columns=shrubins)
    lcl_exp_df.to_csv(exp_file, sep=';', index_label=index_lbl)
    return exp_file


def export_report(report_lst, filename='report', folder='C:/bin', tui=False):
    # todo docstring
    from backend import header_plans, header
    filepath = folder + '/' + filename + '.txt'
    fle = open(filepath, 'w+')
    header = header('output report')
    report_lst.insert(0, header)
    header = header_plans()
    report_lst.insert(0, header)
    fle.writelines(report_lst[:])
    fle.close()
    if tui:
        print('\n\n')
        for e in report_lst[1:]:
            print(e)
        print('\n\n')
