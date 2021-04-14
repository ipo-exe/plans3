import numpy as np
import pandas as pd


def dataframe_prepro(dataframe, strfields='Field1,Field2', strf=True, date=False, datefield='Date'):
    """
    Convenience function for pre processing dataframes
    :param dataframe: pandas dataframe object
    :param strfields: iterable of string fields
    :return: pandas dataframe
    """
    lcl_df = dataframe.copy()
    lcl_df.columns = lcl_df.columns.str.strip()
    if strf:
        fields_lst = strfields.split(',')
        for i in range(len(fields_lst)):
            lcl_df[fields_lst[i]] = lcl_df[fields_lst[i]].str.strip()
    if date:
        lcl_df[datefield] = pd.to_datetime(lcl_df[datefield])
    return lcl_df


def asc_raster(file):
    """
    A function to import .ASC raster files
    :param file: string of file path with the '.asc' extension
    :return: 1) metadata dictionary and 2) numpy 2d array
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
    def_array = np.array(array_lst, dtype='float32')
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
        print(i)
        lcl_meta, lcl_array = asc_raster(lcl_files[i])
        meta_lst.append(meta_lst)
        array_lst.append(lcl_array)
    return meta_lst, array_lst


def zmap(file, yfield='TWI\CN'):
    """
    Import a zmap to a 2d numpy array and respective 1d arrays of histogram values (it is a 2D histogram)
    :param file: string filepath to zmap file
    :param yfield: string of Y (vertical) variable
    :return: 3 returns:
    1) 2d numpy array of ZMAP
    2) 1d numpy array of Y (vertical) variable histogram values
    3) 1d numpy array of X (horizontal) variable histogram values
    """
    lcl_df = pd.read_csv(file, sep=';')
    yhist = lcl_df[yfield].values
    #print(yhist)
    xhist = np.array(lcl_df.columns[1:], dtype='float32')
    #print(xhist)
    zmap = lcl_df.values[:, 1:]
    #print(zmap)
    return zmap, yhist, xhist