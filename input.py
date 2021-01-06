import numpy as np
import pandas as pd

def asc_raster(file):
    """
    A function to import .ASC raster files
    :param file: string of file path with the '.asc' extension
    :return: metadata dictionary and numpy 2d array
    """
    def_f = open(file)
    def_lst = def_f.readlines()
    def_f.close()
    #
    # get metadata constructor loop
    meta_lbls = ('ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value')
    meta_format = ('int', 'int', 'float', 'float', 'float', 'float')
    meta_dct = dict()
    for i in range(6):
        lcl_lst = def_lst[i].split(' ')
        lcl_meta_str = lcl_lst[len(lcl_lst) - 1].split('\n')[0]
        if meta_format[i] == 'int':
            meta_dct[meta_lbls[i]] = int(lcl_meta_str)
        else:
            meta_dct[meta_lbls[i]] = float(lcl_meta_str)
    #
    # array constructor loop:
    array_lst = list()
    for i in range(6, len(def_lst)):
        lcl_lst = def_lst[i].split(' ')[1:]
        lcl_lst[len(lcl_lst) - 1] = lcl_lst[len(lcl_lst) - 1].split('\n')[0]
        array_lst.append(lcl_lst)
    def_array = np.array(array_lst, dtype='float64')
    #
    # replace NoData value by np.nan
    ndv = float(meta_dct['NODATA_value'])
    for i in range(len(def_array)):
        lcl_row_sum = np.sum((def_array[i] == ndv) * 1)
        if lcl_row_sum > 0:
            for j in range(len(def_array[i])):
                if def_array[i][j] == ndv:
                    def_array[i][j] = np.nan
    return meta_dct, def_array


def asc_raster_list(file, filefield='File', sep=';'):
    """
    batch imput of asc raster
    :param file: string filepath of batch txt file
    :param filefield: string of File path field
    :param sep: string data separator
    :return: list of metadata dictionaries and list of 2d numpy arrays
    """
    lcl_df = pd.read_csv(file, sep=sep)
    lcl_files = lcl_df[filefield].values
    array_lst = list()
    meta_lst = list()
    for i in range(len(lcl_files)):
        #print(i)
        lcl_meta, lcl_array = asc_raster(lcl_files[i])
        meta_lst.append(meta_lst)
        array_lst.append(lcl_array)
    return meta_lst, array_lst