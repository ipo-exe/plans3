import numpy as np
import matplotlib.pyplot as plt


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


def find_sinks(dem):
    """
    Find sink cells on a dem
    :param dem: 2d numpy array of dem
    :return: 2d numpy array of sink classes (1 = is a pit sink, 2 = is a plain sink)
    """
    rowmaxid = np.shape(dem)[0] - 1
    colmaxid = np.shape(dem)[1] - 1
    sinks_array = dem * 0.0  # create the blank boolean array
    # loop inside raster bulk:
    for i in range(1, rowmaxid - 1):
        for j in range(1, colmaxid - 1):
            lcl_value = dem[i][j]
            window = dem[i - 1: i + 2, j - 1: j + 2]
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


def fill_sinks(dem, status=False):
    """
    Fill sinks NAIVE algorithm
    :param dem: 2d numpy array of dem
    :return: 2d numpy array of filled dem
    """
    import time
    rowmaxid = np.shape(dem)[0] - 1
    colmaxid = np.shape(dem)[1] - 1
    fill_array = dem.copy()
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


def flow_dir(dem):
    """
    Flow direction algorithm - based on the lowest neighboring cell

    D8 - Direction convetion:

    4   3   2
    5   0   1
    6   7   8

    :param dem: 2d numpy array of dem
    :return: 2d numpy array of flow direction
    """
    rowmaxid = np.shape(dem)[0] - 1
    colmaxid = np.shape(dem)[1] - 1
    #
    # load the flow dir array
    flowdir_array = np.zeros(shape=np.shape(dem))
    #
    # load the flow dir convention window:
    directions = np.array(((4, 3, 2), (5, 0, 1), (6, 7, 8)))
    #
    # moving window scanning loop:
    for i in range(len(dem)):
        for j in range(len(dem[i])):
            lcl_value = dem[i][j]
            #
            # find the window and directions
            if i == 0 and j == 0:
                # print('First corner')
                window = dem[:i + 2, : j + 2]
                windir = directions[1:, 1:]
            elif i == 0 and j == colmaxid:
                # print('Second corner')
                window = dem[:2, j - 1:]
                windir = directions[1:, :2]
            elif i == rowmaxid and j == 0:
                # print('Third corner')
                window = dem[i - 1:, :j + 2]
                windir = directions[:2, 1:]
            elif i == rowmaxid and j == colmaxid:
                # print('Forth corner')
                window = dem[i - 1:, j - 1:]
                windir = directions[:2, :2]
            elif i == 0 and 0 < j < colmaxid:
                # print('Upper edge')
                window = dem[:i + 2, j - 1: j + 2]
                windir = directions[1:, :]
            elif i == rowmaxid and 0 < j < colmaxid:
                # print('Lower edge')
                window = dem[i - 1:, j - 1: j + 2]
                windir = directions[:2, :]
            elif 0 < i < rowmaxid and j == 0:
                # print('Left edge')
                window = dem[i - 1: i + 2, :j + 2]
                windir = directions[:, 1:]
            elif 0 < i < rowmaxid and j == colmaxid:
                # print('Right edge')
                window = dem[i - 1: i + 2, j - 1:]
                windir = directions[:, :2]
            else:
                # print('Bulk')
                window = dem[i - 1: i + 2, j - 1: j + 2]
                windir = directions
                # print(window)
            #
            # lowest elevation method:
            lcl_min = np.min(window)
            #
            # sink conditionals
            if lcl_value == lcl_min:
                lcl_direction = directions[1, 1]
            else:
                mask = ((window == lcl_min) * 1) * windir
                lcl_direction = np.max(mask)
            #
            # insert on flow dir array
            flowdir_array[i][j] = lcl_direction
            #
    return flowdir_array


def flow_acc(flowdir):
    """
    Flow accumulation algorithm of Zhou et al (2019)

    D8 - Direction convetion:

    4   3   2
    5   0   1
    6   7   8

    Reverse direction convention:

    8   7   6
    1   0   5
    2   3   4


    :param flowdir: 2d numpy array of flow direction using the above convention
    :return: 2d numpy array of flow accum
    """

    def nidp(flowdir):
        rowmaxid = np.shape(flowdir)[0] - 1
        colmaxid = np.shape(flowdir)[1] - 1
        #
        # load the flow dir array
        local_accum = np.zeros(shape=np.shape(flowdir))
        #
        # load the flow dir convention window:
        reverse_directions = np.array(((8, 7, 6), (1, 0, 5), (2, 3, 4)))
        #
        # moving window scanning loop:
        for i in range(len(flowdir)):
            for j in range(len(flowdir[i])):
                #
                # find the window and directions
                if i == 0 and j == 0:
                    #print('First corner')
                    window = flowdir[:i + 2, : j + 2]
                    windir = reverse_directions[1:, 1:]
                elif i == 0 and j == colmaxid:
                    #print('Second corner')
                    window = flowdir[:2, j - 1:]
                    windir = reverse_directions[1:, :2]
                elif i == rowmaxid and j == 0:
                    #print('Third corner')
                    window = flowdir[i - 1:, :j + 2]
                    windir = reverse_directions[:2, 1:]
                elif i == rowmaxid and j == colmaxid:
                    #print('Forth corner')
                    window = flowdir[i - 1:, j - 1:]
                    windir = reverse_directions[:2, :2]
                elif i == 0 and 0 < j < colmaxid:
                    #print('Upper edge')
                    window = flowdir[:i + 2, j - 1: j + 2]
                    windir = reverse_directions[1:, :]
                elif i == rowmaxid and 0 < j < colmaxid:
                    #print('Lower edge')
                    window = flowdir[i - 1:, j - 1: j + 2]
                    windir = reverse_directions[:2, :]
                elif 0 < i < rowmaxid and j == 0:
                    #print('Left edge')
                    window = flowdir[i - 1: i + 2, :j + 2]
                    windir = reverse_directions[:, 1:]
                elif 0 < i < rowmaxid and j == colmaxid:
                    #print('Right edge')
                    window = flowdir[i - 1: i + 2, j - 1:]
                    windir = reverse_directions[:, :2]
                else:
                    #print('Bulk')
                    window = flowdir[i - 1: i + 2, j - 1: j + 2]
                    windir = reverse_directions
                    # print(window)
                # count number of poiting neighbor cells
                lcl_count = np.sum(1.0 * (window == windir))
                # insert on flow dir array
                local_accum[i][j] = lcl_count
            #
        return local_accum


    def downstream_coordinates(dir, x, y):
        """
        Compute x and y donwstream cell coordinates based on cell flow direction

        D8 - Direction convetion:

        4   3   2
        5   0   1
        6   7   8

        :param dir: int flow direction code
        :param x: int x (row) array index
        :param y: int y (column) array index
        :return: x and y downstream cell array indexes
        """
        if dir == 1:
            x = x
            y = y + 1
        elif dir == 2:
            x = x - 1
            y = y + 1
        elif dir == 3:
            x = x - 1
            y = y
        elif dir == 4:
            x = x - 1
            y = y - 1
        elif dir == 5:
            x = x
            y = y - 1
        elif dir == 6:
            x = x + 1
            y = y - 1
        elif dir == 7:
            x = x + 1
            y = y
        elif dir == 8:
            x = x + 1
            y = y + 1
        elif dir == 0:
            x = x
            y = y
        return x, y

    #
    # get the initiation grid
    nidp_array = nidp(flowdir)
    #
    # load the flow acc array
    flow_accum = np.ones(shape=np.shape(flowdir))
    #
    # moving window scanning loop in the NIDP array:
    for i in range(len(nidp_array)):
        for j in range(len(nidp_array[i])):
            print('{}-{}'.format(i, j))
            #
            # get local value of NIDP
            lcl_nidp = nidp_array[i][j]
            #
            # source cell condition:
            if lcl_nidp == 0:
                #
                # start tracing procedure:
                x = i
                y = j
                while True:
                    # todo this is messed up. fix tracing procedure.
                    #
                    # get current flow accum value:
                    local_flow_acc = flow_accum[x][y]
                    #
                    # get flow direction
                    direction = flowdir[x][y]
                    #
                    # move to downstream cell -- get next x and y
                    x, y = downstream_coordinates(dir=direction, x=x, y=y)
                    #
                    # update flow accumulation array
                    downstream_flow_accum = flow_accum[x][y]
                    flow_accum[x][y] = downstream_flow_accum + local_flow_acc
                    print(flow_accum[x][y])
                    # check stop condition
                    if nidp_array[x][y] > 1 or flowdir[x][y] == 0:
                        nidp_array[x][y] = 1
                        break
            plt.imshow(flow_accum)
            plt.show()
    return flow_accum




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
    # todo optimze and revise

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


def cn(lulc, soils, cnvalues, lulcclasses, soilclasses):
    """
    derive the CN map based on LULC and Soils groups
    :param lulc: lulc 2d array
    :param soils: soils 2d array
    :param cnvalues: array of CN values for A, B, C and D soils (2d array) in the order of lulc classes
    :param lulcclasses: array of lulc classes values
    :param soilclasses: array of soil classses values
    :return: 2d array of CN
    """
    soilclasses = soilclasses * 100
    cn_class = (100 * soils) + lulc
    cn_map = lulc * 0.0
    for i in range(len(soilclasses)):
        for j in range(len(lulcclasses)):
            lcl_class = soilclasses[i] + lulcclasses[j]
            lcl_cn = cnvalues[i][j]
            cn_map = cn_map + (cn_class == lcl_class) * lcl_cn
    return cn_map


def grad(slope):
    """
    derive the topographical gradient tan(B) from the slope in degrees
    :param array: slope in degrees 2d array
    :return:
    """
    slope_rad = np.pi * 2 * slope / 360
    grad = np.tan(slope_rad)
    return grad


def areas(array, cellsize, values, factor=1):
    """
    derive a list of areas in array based on a list of values
    :param array: 2d array
    :param cellsize: cellsize float
    :param values: sequence of values to lookup
    :return: array of areas in cellsize squared units
    """
    areas = list()
    for i in range(len(values)):
        lcl_val = values[i]
        lcl_bool = (array == lcl_val) * 1
        lcl_pixsum = np.sum(lcl_bool)
        lcl_area = lcl_pixsum * cellsize * cellsize
        areas.append(lcl_area)
    return np.array(areas)/(factor * factor)


def mask(array, mask):
    masked = np.copy(array)
    masked[mask == 0] = np.nan
    return masked


def twi(catcha, grad, cellsize, gradmin=0.0001):
    """
    Derive the Topographical Wetness Index of TOPMODEL (Beven & Kirkby, 1979)

    :param catcha: cathment area 2d array in meters
    :param grad: gradient of terrain 2d array (tangent of slope)
    :param cellsize: cell size in meters
    :param gradmin: minimun gradient threshold
    :return: Topographical Wetness Index 2d array
    """

    return np.log(catcha / (cellsize * (grad + gradmin)))


def flatten_clear(array, mask):
    masked = np.copy(array)
    masked[mask == 0] = np.nan
    flatten = masked.flatten()
    cleared = masked[~np.isnan(masked)]
    return cleared


def reclassify(array, upvalues, classes):
    new = array * 0.0
    for i in range(len(upvalues)):
        if i == 0:
            new = new + ((array <= upvalues[i]) * classes[i])
        else:
            new = new + ((array > upvalues[i - 1]) * (array <= upvalues[i]) * classes[i])
    return new