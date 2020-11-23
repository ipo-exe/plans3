import numpy as np


def asc_raster(array, meta, folder, filename):
    """
    Function for exporting an .ASC raster file.
    :param array: 2d numpy array
    :param meta: dicitonary of metadata. Example:

    {'ncols': '366',
     'nrows': '434',
      'xllcorner': '559493.087150689564',
       'yllcorner': '6704832.279550871812',
        'cellsize': '28.854232957826',
        'NODATA_value': '-9999'}

    :param folder: string of directory path
    :param filename: string of file without extension
    :return: full file name (path and extension) string
    """
    meta_lbls = ('ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value')
    ndv = float(meta['NODATA_value'])
    exp_lst = list()
    for i in range(len(meta_lbls)):
        line = '{}    {}\n'.format(meta_lbls[i], meta[meta_lbls[i]])
        exp_lst.append(line)
    print(exp_lst)
    #
    # data constructor loop:
    for i in range(len(array)):
        # replace np.nan to no data values
        lcl_row_sum = np.sum((np.isnan(array[i])) * 1)
        if lcl_row_sum > 0:
            print('Yeas')
            for j in range(len(array[i])):
                if np.isnan(array[i][j]):
                    array[i][j] = int(ndv)
        str_join = ' ' + ' '.join(np.array(array[i], dtype='str')) + '\n'
        exp_lst.append(str_join)

    flenm = folder + '/' + filename + '.asc'
    fle = open(flenm, 'w+')
    fle.writelines(exp_lst)
    fle.close()
    return flenm
