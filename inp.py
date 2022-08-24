'''
PLANS - Planning Nature-based Solutions

Input routines

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


def asc_raster(file, nan=False, dtype='int16'):
    """
    A function to import .ASC raster files
    :param file: string of file path with the '.asc' extension
    :param nan: boolean to convert nan values to np.nan
    :param dtype: string code to data type. Options: 'int16', 'int32', 'float32' etc
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
    def_array = np.array(array_lst, dtype=dtype)
    #
    # replace NoData value by np.nan
    if nan:
        ndv = float(meta_dct['NODATA_value'])
        for i in range(len(def_array)):
            lcl_row_sum = np.sum((def_array[i] == ndv) * 1)
            if lcl_row_sum > 0:
                for j in range(len(def_array[i])):
                    if def_array[i][j] == ndv:
                        def_array[i][j] = np.nan
    return meta_dct, def_array


def asc_raster_meta(file):
    """
    A function to import .ASC raster files
    :param file: string of file path with the '.asc' extension
    :return: metadata dictionary
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
    return meta_dct


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


def zmap(file, yfield='TWI\SHRU', nodata=-1):
    """
    Import a zmap to a 2d numpy array and respective 1d arrays of histogram values (it is a 2D histogram)
    :param file: string filepath to zmap file
    :param yfield: string of Y (vertical) variable
    :return: 3 returns:
    1) 2d numpy array of ZMAP
    2) 1d numpy array of Y (vertical) variable histogram values
    3) 1d numpy array of X (horizontal) variable histogram values
    """
    dataframe = pd.read_csv(file, sep=';')
    dataframe = dataframe_prepro(dataframe, strf=False)
    # set first column as index
    dataframe = dataframe.set_index(dataframe.columns[0])
    # get shru bins
    shru_bins = dataframe.columns.astype('int')
    # get twi bins
    twi_bins = dataframe.index.values
    # extract ZMAP
    zmap = (dataframe.values * (dataframe.values != nodata)) + (nodata * (dataframe.values == nodata))
    return zmap, twi_bins, shru_bins


def hydroparams(fhydroparam):
    """
    Import the hydrology reference parameters to a dictionary.
    :param fhydroparam: hydro_param txt filepath
    :return: dictionary of dictionaries of parameters Set, Min and Max and pandas dataframe
    """
    hydroparam_df = pd.read_csv(fhydroparam, sep=';')
    hydroparam_df = dataframe_prepro(hydroparam_df, 'Parameter')
    #
    fields = ('Set', 'Min', 'Max', 'AOI_f')
    params = ('m', 'lamb', 'qo', 'cpmax', 'sfmax', 'erz', 'ksat', 'c', 'lat', 'k', 'n')
    # built dict
    hydroparams_dct = dict()
    for p in params:
        lcl_dct = dict()
        for f in fields:
            lcl_dct[f] = hydroparam_df[hydroparam_df['Parameter'] == p][f].values[0]
        hydroparams_dct[p] = lcl_dct
    return hydroparams_dct, hydroparam_df


def histograms(fhistograms):
    """
    Extract the histogram data
    :param fhistograms: string filepath to the txt file of histograms
    :return: 2d numpy array of count matrix, 1d numpy array of twi bins and 1d array of SHRU bins
    """
    dataframe = pd.read_csv(fhistograms, sep=';')
    dataframe = dataframe_prepro(dataframe, strf=False)
    dataframe = dataframe.set_index(dataframe.columns[0])
    shru_ids = dataframe.columns.astype('int')
    twi_bins = dataframe.index.values
    count_matrix = dataframe.values
    return count_matrix, twi_bins, shru_ids


def lulcparam(flulcparam):
    """
    Import the lulc parameters dataframe
    :param flulcparam: string filepath
    :return: pandas dataframe
    """
    from backend import get_stringfields
    _lulc_df = pd.read_csv(flulcparam, sep=';', engine='python')
    _lulc_df = dataframe_prepro(_lulc_df, strfields=get_stringfields(filename='lulc'))
    return _lulc_df


def soilsparam(fsoilsparam):
    """
    Import the soils parameters dataframe
    :param flulcparam: string filepath
    :return: pandas dataframe
    """
    from backend import get_stringfields
    _soils_df = pd.read_csv(fsoilsparam, sep=';', engine='python')
    _soils_df = dataframe_prepro(_soils_df, strfields=get_stringfields(filename='soils'))
    return _soils_df


def shruparam(fshruparam):
    """
    Import the SHRU parameters dataframe
    :param flulcparam: string filepath
    :return: pandas dataframe
    """
    from backend import get_stringfields
    _shru_df = pd.read_csv(fshruparam, sep=';', engine='python')
    _shru_df = dataframe_prepro(_shru_df, strfields=get_stringfields(filename='shru'))
    return _shru_df
