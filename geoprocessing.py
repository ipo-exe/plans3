import numpy as np


def stddem():
    """
    Utility function for code development
    :return: asc file metadata dictionary and 2d numpy array of standard dem
    """
    lst = [(60, 75, 80, 75, 65, 50, 65, 75, 80, 75, 70),
           (70, 65, 70, 85, 72, 55, 70, 80, 70, 65, 75),
           (80, 75, 40, 75, 70, 62, 65, 62, 60, 68, 70),
           (78, 85, 65, 40, 60, 65, 62, 40, 62, 70, 65),
           (40, 60, 70, 60, 50, 45, 50, 55, 60, 50, 60),
           (55, 45, 50, 45, 40, 50, 55, 60, 55, 45, 55),
           (60, 65, 45, 18, 20, 45, 50, 45, 40, 48, 50),
           (65, 60, 55, 12, 15, 40, 30, 35, 50, 55, 35),
           (70, 45, 35, 10, 5, 20, 25, 45, 40, 35, 30),
           (65, 40, 45, 50, 45, 35, 15, 20, 45, 42, 35),
           (50, 55, 50, 40, 50, 55, 45, 10, 20, 30, 40),]
    dct = {'ncols': '11',
           'nrows': '11',
           'xllcorner': '559490.0',
           'yllcorner': '6704830.0',
           'cellsize': '100',
           'NODATA_value': '-9999'}
    return dct, np.array(lst)


def find_sinks(array):
    """
    Find sink cells on a dem
    :param array: 2d numpy array of dem
    :return: 2d numpy array of boolean values (1 = is sink)
    """
    rowmaxid = np.shape(array)[0] - 1
    colmaxid = np.shape(array)[1] - 1
    sinks_array = array * 0.0  # create the blank boolean array
    # loop inside raster bulk:
    for i in range(1, rowmaxid - 1):
        for j in range(1, colmaxid - 1):
            lcl_value = array[i][j]
            window = array[i - 1: i + 2, j - 1: j + 2]
            lcl_min = np.min(window)
            if lcl_value == lcl_min:
                sinks_array[i][j] = 1.0  # replace boolean True value
    return sinks_array


def fill_sinks(array):
    """
    Fill sinks algorithm
    :param array: 2d numpy array of dem
    :return: 2d numpy array of filled dem
    """
    rowmaxid = np.shape(array)[0] - 1
    colmaxid = np.shape(array)[1] - 1
    fill_array = array.copy()
    while True:
        sink_counter = 0
        # loop inside raster bulk:
        for i in range(1, rowmaxid):
            for j in range(1, colmaxid):
                lcl_value = fill_array[i][j]
                window = fill_array[i - 1: i + 2, j - 1: j + 2]
                lcl_edge_val = (window[0][0], window[0][1], window[0][2],
                                window[1][2], window[1][0],
                                window[2][0], window[2][1], window[2][2],)
                lcl_min = np.min(window)
                if lcl_value == lcl_min:
                    # print('\nSink Found')
                    fill_array[i][j] =  (np.max(lcl_edge_val) + np.min(lcl_edge_val)) / 2
                    sink_counter = sink_counter + 1
        print(sink_counter)
        if sink_counter == 0:
            break
    return fill_array


def flow_dir(array):
    """
    Flow direction algorithm - based on the lowest neighboring cell

    Direction convetion:

    2   3   4
    9   1   5
    8   7   6

    :param array: 2d numpy array of dem
    :return: 2d numpy array of flow direction
    """
    rowmaxid = np.shape(array)[0] - 1
    colmaxid = np.shape(array)[1] - 1
    flowdir_array = array * 0.0
    directions = np.array(((2, 3, 4), (9, 1, 5), (8, 7, 6)))
    for i in range(len(array)):
        for j in range(len(array[i])):
            lcl_value = array[i][j]
            #
            # find the window and directions
            if i == 0 and j == 0:
                # print('First corner')
                window = array[:i + 2, : j + 2]
                windir = directions[1:, 1:]
            elif i == 0 and j == colmaxid:
                # print('Second corner')
                window = array[:2, j - 1:]
                windir = directions[1:, :2]
            elif i == rowmaxid and j == 0:
                # print('Third corner')
                window = array[i - 1:, :j + 2]
                windir = directions[:2, 1:]
            elif i == rowmaxid and j == colmaxid:
                # print('Forth corner')
                window = array[i - 1:, j - 1:]
                windir = directions[:2, :2]
            elif i == 0 and 0 < j < colmaxid:
                # print('Upper edge')
                window = array[:i + 2, j - 1: j + 2]
                windir = directions[1:, :]
            elif i == rowmaxid and 0 < j < colmaxid:
                # print('Lower edge')
                window = array[i - 1:, j - 1: j + 2]
                windir = directions[:2, :]
            elif 0 < i < rowmaxid and j == 0:
                # print('Left edge')
                window = array[i - 1: i + 2, :j + 2]
                windir = directions[:, 1:]
            elif 0 < i < rowmaxid and j == colmaxid:
                # print('Right edge')
                window = array[i - 1: i + 2, j - 1:]
                windir = directions[:, :2]
            else:
                # print('Bulk')
                window = array[i - 1: i + 2, j - 1: j + 2]
                windir = directions
                # print(window)
            lcl_min = np.min(window)
            if lcl_value == lcl_min:
                flowdir_array[i][j] = directions[1, 1]
            else:
                mask = ((window == lcl_min) * 1) * windir
                lcl_direction = np.max(mask)
                flowdir_array[i][j] = lcl_direction
    return flowdir_array


def slope(array, cellsize, degree=True):
    """
    Slope algorithm based on gradient built in functions of numpy
    :param array: 2d numpy array of dem
    :param cellsize: float value of cellsize (delta x = delta y)
    :param degree: boolean to control output units. Defalt = True. If False output units are in radians
    :return: 2d numpy array of slope
    """
    grad = np.gradient(array)
    gradx = grad[0] / cellsize
    grady = grad[1] / cellsize
    gradv = np.sqrt((gradx * gradx) + (grady * grady))
    slope_array = np.arctan(gradv)
    if degree:
        slope_array = slope_array * 360 / (2 * np.pi)
    return slope_array
