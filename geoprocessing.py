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
    :return: 2d numpy array of sink classes (1 = is a pit sink, 2 = is a plain sink)
    """
    rowmaxid = np.shape(array)[0] - 1
    colmaxid = np.shape(array)[1] - 1
    sinks_array = array * 0.0  # create the blank boolean array
    # loop inside raster bulk:
    for i in range(1, rowmaxid - 1):
        for j in range(1, colmaxid - 1):
            lcl_value = array[i][j]
            window = array[i - 1: i + 2, j - 1: j + 2]
            lcl_edge_val_lst = (window[0][0], window[0][1], window[0][2], window[1][2], window[1][0],
                                window[2][0], window[2][1], window[2][2],)
            lcl_edge_val = np.array(lcl_edge_val_lst)
            # classic sink condition
            edge_min = np.min(lcl_edge_val)
            if lcl_value < edge_min:  # pit sink condition
                # print('Pit Sink')
                sinks_array[i][j] = 1.0  # replace boolean True value
            elif lcl_value == edge_min:  # plain sink condition
                # print('Plain Sink')
                sinks_array[i][j] = 2.0  # replace boolean True value

    return sinks_array


def fill_sinks(array, status=False):
    """
    Fill sinks algorithm
    :param array: 2d numpy array of dem
    :return: 2d numpy array of filled dem
    """
    import time
    rowmaxid = np.shape(array)[0] - 1
    colmaxid = np.shape(array)[1] - 1
    fill_array = array.copy()
    iter = 1
    start = time.time()
    while True:
        sink_counter = 0
        # loop inside raster bulk:
        for i in range(1, rowmaxid):
            for j in range(1, colmaxid):
                lcl_value = fill_array[i][j]
                window = fill_array[i - 1: i + 2, j - 1: j + 2]
                lcl_edge_val_lst = (window[0][0], window[0][1], window[0][2], window[1][2], window[1][0],
                                    window[2][0], window[2][1], window[2][2],)
                lcl_edge_val = np.array(lcl_edge_val_lst)
                edge_min = np.min(lcl_edge_val)
                edge_max = np.max(lcl_edge_val)
                edge_avg = (edge_max + edge_min) / 2
                if lcl_value < edge_min:
                    # pit sink condition
                    fill_array[i][j] = edge_avg  # replace by the edge average
                    sink_counter = sink_counter + 1
                elif lcl_value == edge_min:
                    # plain sink condition
                    if lcl_value == edge_max:
                        # open plain condition
                        pass
                    else:
                        fill_array[i][j] = edge_avg  # replace by the edge average
                        sink_counter = sink_counter + 1
        if status:
            deltat = time.time() - start
            print('Fill Sinks -- iteration # {:<6}\t\tSinks left:\t{:8}'
                  '\t\tEnlapsed time: {:8.1f} s'.format(iter, sink_counter, deltat))
        iter = iter + 1
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

# todo
# def catchment_area(array, cellsize)
# def extract_points()


def fill_sinks_wang(array, status=True):
    import time

    def find_edge_mask(array):
        rowmaxid = np.shape(array)[0] - 1
        colmaxid = np.shape(array)[1] - 1
        edge_mask = array.copy() * 0.0
        for i in range(len(array)):
            for j in range(len(array[i])):
                lcl_value = array[i][j]
                if lcl_value == 1:
                    pass
                else:
                    # find the window
                    if i == 0 and j == 0:
                        # print('First corner')
                        window = array[:i + 2, : j + 2]
                    elif i == 0 and j == colmaxid:
                        # print('Second corner')
                        window = array[:2, j - 1:]
                    elif i == rowmaxid and j == 0:
                        # print('Third corner')
                        window = array[i - 1:, :j + 2]
                    elif i == rowmaxid and j == colmaxid:
                        # print('Forth corner')
                        window = array[i - 1:, j - 1:]
                    elif i == 0 and 0 < j < colmaxid:
                        # print('Upper edge')
                        window = array[:i + 2, j - 1: j + 2]
                    elif i == rowmaxid and 0 < j < colmaxid:
                        # print('Lower edge')
                        window = array[i - 1:, j - 1: j + 2]
                    elif 0 < i < rowmaxid and j == 0:
                        # print('Left edge')
                        window = array[i - 1: i + 2, :j + 2]
                    elif 0 < i < rowmaxid and j == colmaxid:
                        # print('Right edge')
                        window = array[i - 1: i + 2, j - 1:]
                    else:
                        # print('Bulk')
                        window = array[i - 1: i + 2, j - 1: j + 2]
                    # print(window)
                    # print(np.sum(window))
                    if np.sum(window) > 0:
                        # print('Edge cell')
                        edge_mask[i][j] = 1.0
        return edge_mask

    def first_bulk(array):
        rowmaxid = np.shape(array)[0] - 1
        colmaxid = np.shape(array)[1] - 1
        bulk = (array.copy() * 0.0) + 1.0
        for i in range(len(array)):
            for j in range(len(array[i])):
                # find the window
                if i == 0:
                    # print('Upper edge')
                    bulk[i][j] = 0.0
                elif i == rowmaxid:
                    # print('Lower edge')
                    bulk[i][j] = 0.0
                elif j == 0:
                    # print('Left edge')
                    bulk[i][j] = 0.0
                elif j == colmaxid:
                    # print('Right edge')
                    bulk[i][j] = 0.0
        return bulk

    def window_mask(array, i=1, j=1):
        rowmaxid = np.shape(array)[0] - 1
        colmaxid = np.shape(array)[1] - 1
        mask = (array.copy() * 0.0)
        # find the window
        if i == 0 and j == 0:
            # print('First corner')
            mask[:i + 2, : j + 2] = 1
        elif i == 0 and j == colmaxid:
            # print('Second corner')
            mask[:2, j - 1:] = 1
        elif i == rowmaxid and j == 0:
            # print('Third corner')
            mask[i - 1:, :j + 2] = 1
        elif i == rowmaxid and j == colmaxid:
            # print('Forth corner')
            mask[i - 1:, j - 1:] = 1
        elif i == 0 and 0 < j < colmaxid:
            # print('Upper edge')
            mask[:i + 2, j - 1: j + 2] = 1
        elif i == rowmaxid and 0 < j < colmaxid:
            # print('Lower edge')
            mask[i - 1:, j - 1: j + 2] = 1
        elif 0 < i < rowmaxid and j == 0:
            # print('Left edge')
            mask[i - 1: i + 2, :j + 2] = 1
        elif 0 < i < rowmaxid and j == colmaxid:
            # print('Right edge')
            mask[i - 1: i + 2, j - 1:] = 1
        else:
            # print('Bulk')
            mask[i - 1: i + 2, j - 1: j + 2] = 1
        return mask

    iter = 0
    rowmaxid = np.shape(array)[0] - 1
    colmaxid = np.shape(array)[1] - 1
    fill_array = array.copy()
    # get the first bulk:
    bulk_mask = first_bulk(fill_array)
    '''plt.imshow(bulk_mask, cmap='viridis')
    plt.title('bulk')
    plt.show()'''
    bulk_size = np.sum(bulk_mask)
    start_size = bulk_size
    start = time.time()
    while bulk_size > 0:
        edge_mask = find_edge_mask(bulk_mask)
        edge = fill_array * edge_mask
        edge_max = np.max(edge)
        edge = ((edge == 0) * edge_max) + edge
        '''plt.imshow(edge, cmap='viridis')
        plt.title('edge')
        plt.show()'''
        outlet_val = np.min(edge)
        #print(outlet_val)
        outlet_mask = (edge == outlet_val) * 1.0
        ind_cols = outlet_mask.argmax(axis=1)
        for i in range(len(ind_cols)):
            #print('row:{}\tcol:{}\tvalue:{}'.format(i, ind_cols[i], outlet_mask[i][ind_cols[i]]))
            if outlet_mask[i][ind_cols[i]] == 1:
                outlet_i = i
                outlet_j = ind_cols[i]
                break
        window = window_mask(fill_array, i=outlet_i, j=outlet_j)
        '''plt.imshow(window, cmap='viridis')
        plt.title('window')
        plt.show()'''
        # find fill mask
        fill_mask = window * bulk_mask * fill_array
        '''plt.imshow(fill_mask, cmap='viridis')
        plt.title('fill mask')
        plt.show()'''
        condition = np.sum((fill_mask < outlet_val) * (fill_mask > 0))
        if condition > 0:
            # print('Sink found')
            # update fill mask
            fill_mask = (fill_mask > 0) * (((fill_mask >= outlet_val) * fill_mask) + ((fill_mask < outlet_val) * outlet_val))
            '''plt.imshow(fill_mask, cmap='viridis')
            plt.title('fill mask updated')
            plt.show()'''
            # update fill array:
            fill_array = (fill_array * ((fill_mask == 0) * 1)) + fill_mask
            '''plt.imshow(fill_array, cmap='viridis')
            plt.title('filled dem')
            plt.show()'''
        # update bulk
        bulk_mask = bulk_mask - (fill_mask > 0)
        '''plt.imshow(bulk_mask, cmap='viridis')
        plt.title('new bulk')
        plt.show()'''
        bulk_size = np.sum(bulk_mask)
        #print('Bulk size: {}\t\t'.format(bulk_size))
        if status:
            deltat = time.time() - start
            process = (start_size - bulk_size) * 100 / start_size
            print('Fill Sinks -- iteration # {:<6}\t\tImage processing:\t{:8.1f}%'
                  '\t\tEnlapsed time: {:8.1f} s'.format(iter, process, deltat))
        iter = iter + 1
    return fill_array
