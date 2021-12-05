'''
UFRGS - Universidade Federal do Rio Grande do Sul
IPH - Instituto de Pesquisas Hidráulicas
WARP - Research Group in Water Resources Management and Planning
Porto Alegre, Rio Grande do Sul, Brazil

plans - planning nature-based solutions
Version: 3.0

This software is under the GNU GPL3.0 license

Source code repository: https://github.com/ipo-exe/plans3/
Authors: Ipora Possantti: https://github.com/ipo-exe

This file is under LICENSE: GNU General Public License v3.0
Permissions:
    Commercial use
    Modification
    Distribution
    Patent use
    Private use
Limitations:
    Liability
    Warranty
Conditions:
    License and copyright notice
    State changes
    Disclose source
    Same license

Module description:
This is the PLANS 3 geoprocessing raster module.
'''
import numpy as np


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


def xmap(map1, map2, map1ids, map2ids, map1f=100, map2f=1):
    """
    Generalized crossing map function
    :param map1: 2d array of map 1
    :param map2: 2d array of map 1
    :param map1ids: 1d array of map 1 ids
    :param map2ids: 1d array of map 2 ids
    :param map1f: int - factor of map 1 in map algebra
    :param map2f: int - factor of map 2 in map algebra
    :return: 2d array of crossed map
    """
    map1_values = map1ids * map1f
    map2_values = map2ids * map2f
    xmap = map1 * 0.0
    for i in range(len(map1_values)):
        for j in range(len(map2_values)):
            xmap_value = map1_values[i] + map2_values[j]
            lcl_xmap = (map1 == map1ids[i]) * (map2 == map2ids[j]) * xmap_value
            xmap = xmap + lcl_xmap
    return xmap


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


def extract_grid_centroids(array, meta, byvalue=False, value=1):
    """
    extract centroid of array cells
    :param array: 2d numpy array
    :param meta: metadata dictionary with 'xllcorner', 'yllcorner' and 'cellsize'
    :return: dictionary of X, Y and Z values
    """
    shape = np.shape(array)
    nrows = shape[0]
    ncols = shape[1]
    if byvalue:
        size = int(np.sum(1.0 * (array == value)))
    else:
        size = int(nrows * ncols)

    xll = float(meta['xllcorner'])
    yll = float(meta['yllcorner'])
    cellsize = float(meta['cellsize'])

    x_long = np.zeros(shape=size)
    y_lat = np.zeros(shape=size)
    z_val = np.zeros(shape=size)
    # get relative coordinates
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            if byvalue:
                if array[i][j] == value:
                    z_val[count] = array[i][j]
                    x_long[count] = (j * cellsize)
                    y_lat[count] = ((nrows - i - 1) * cellsize)
                    count = count + 1
            else:
                z_val[count] = array[i][j]
                x_long[count] = (j * cellsize)
                y_lat[count] = ((nrows - i - 1) * cellsize)
                count = count + 1
    # get absolute coordinates
    x_long = x_long + xll + (cellsize / 2)
    y_lat = y_lat + yll + (cellsize / 2)
    return {'X':x_long, 'Y':y_lat, 'Z':z_val}


def extract_grid_nodes(array, meta):
    shape = np.shape(array)
    nrows = shape[0] + 1
    ncols = shape[1] + 1
    size = int(nrows * ncols)

    xll = float(meta['xllcorner'])
    yll = float(meta['yllcorner'])
    cellsize = float(meta['cellsize'])

    x_long = np.zeros(shape=size)
    y_lat = np.zeros(shape=size)
    # get relative coordinates
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            x_long[count] = (j * cellsize)
            y_lat[count] = ((nrows - i - 2) * cellsize)
            count = count + 1
    # get absolute coordinates
    x_long = x_long + xll
    y_lat = y_lat + yll + cellsize
    return {'X': x_long, 'Y': y_lat}


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


def flatten_clear(array, mask):
    """
    Flatten and clear a 2d numpy array using a mask pseudo-boolean array
    :param array: field 2d numpy array
    :param mask: mask pseudo-boolean 2d numpy array
    :return: 1d numpy array of cleared values
    """
    masked = np.copy(array)
    masked[mask == 0] = np.nan
    flatten = masked.flatten()
    cleared = masked[~np.isnan(masked)]
    return cleared


def flow_acc(flowdir, status=False):
    """
    Flow accumulation algorithm

    D8 - Direction convetion:

    4   3   2
    5   0   1
    6   7   8

    Reverse direction convention:

    8   7   6
    1   0   5
    2   3   4


    :param flowdir: 2d numpy array of flow direction using the above convention
    :return: 2d numpy array of flow accum (number of accumulated cells)
    """
    #
    # get the initiation grid
    nidp_array = local_flowacc(flowdir)
    if status:
        c = 1
        cmax = np.sum(1.0 * (nidp_array == 0))
    #
    # load the flow acc array
    flow_accum = np.ones(shape=np.shape(flowdir))
    #
    # moving window scanning loop in the NIDP array:
    for i in range(len(nidp_array)):
        for j in range(len(nidp_array[i])):
            #
            # get local value of NIDP
            lcl_nidp = nidp_array[i][j]
            #
            # source cell condition:
            if lcl_nidp == 0:
                if status:
                    print('Flow Accumulation\t{:.3f} %'.format(100 * c / cmax))
                    c = c + 1
                #
                # start tracing procedure:
                x = i
                y = j
                accumulate = True
                while True:
                    #
                    # get current flow accum value:
                    if accumulate:
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
                    # propagate
                    flow_accum[x][y] = downstream_flow_accum + local_flow_acc
                    if downstream_flow_accum > 1:
                        accumulate = False
                    #
                    # check stop condition
                    if flowdir[x][y] == 0:
                        break
    return flow_accum


def flow_dir(dem, status=False):
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
    cmax = np.shape(dem)[0] * np.shape(dem)[1]
    c = 1
    # moving window scanning loop:
    for i in range(len(dem)):
        for j in range(len(dem[i])):
            if status:
                print('Flow Direction\t{:.3f} %'.format(100 * c / cmax))
                c = c + 1
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


def fuzzy_transition(array, a, b, ascending=True, type='senoid'):
    """

    Fuzzify a numpy array by a transition from a to b values

    :param array: numpy array
    :param a: float initial threshold
    :param b: float terminal threshold
    :param ascending: boolean
    :param type: string type of fuzzy. options: 'senoid' and 'linear' (trapezoid)
    :return: numpy array
    """
    if ascending:
        if type == 'senoid':
            transition = (array >= b) + (array > a) * (array < b) * (-0.5 * np.cos(np.pi * (array - a)/(b - a)) + 0.5 )
        if type == 'linear':
            transition = (array >= b) + (array > a) * (array < b) * (( array / (b - a)) - (a / (b - a)))
    else:
        if type == 'senoid':
            transition = (array <= a) + (array > a) * (array < b) * (0.5 * np.cos(np.pi * (array - a)/(b - a)) + 0.5)
        if type == 'linear':
            transition = (array <= a) + (array > a) * (array < b) * ((- array / (b - a)) + (b / (b - a)))
    return transition


def landforms(slope, tpimicro, tpimacro, v1=-1, v2=1, v3=-1, v4=1, v5=4):
    """
    landform TPI-based classification (Weiss, 2001)

    :param slope: 2d numpy array of slope
    :param tpimicro: 2d numpy array of TPI of micro topography
    :param tpimacro: 2d numpy array of TPI of macro topography
    :param v1: float of tpi micro lower threshold in tpi units
    :param v2: float of tpi micro upper threshold in tpi units
    :param v3: float of tpi micro lower threshold in tpi units
    :param v4: float of tpi macro upper threshold in tpi units
    :param v5: float of slope threshold in slope units
    :return: numpy array of 10 landform classes:

    11 - valley bottoms
    12 - valleys
    13 - valley borders
    21 - Mesas
    22 - Gentle hillslopes
    23 - Ridges
    24 - Plains
    31 - Hill thalweg
    32 - Hill tops
    33 - Peaks
    """
    slope_rcl = 1 * (slope <= v5)
    tpi_micro_rcl = (1 * (tpimicro <= v1)) + (2 * (tpimicro > v1) * (tpimicro <= v2)) + (3 * (tpimicro > v2))
    tpi_macro_rcl = (1 * (tpimacro <= v3)) + (2 * (tpimacro > v3) * (tpimacro <= v4)) + (3 * (tpimacro > v4))
    # cross tpi
    lf9 = tpi_micro_rcl + (tpi_macro_rcl * 10)
    # define plains
    lf10 = ((slope_rcl * (lf9 == 22)) * 2) + lf9
    return lf10


def local_flowacc(flowdir):
    """
    compute the local flow accumulation

    D8 - Direction convetion:

    4   3   2
    5   0   1
    6   7   8

    Reverse direction convention:

    8   7   6
    1   0   5
    2   3   4

    :param flowdir: 2d numpy array of flow direction
    :return: 2d numpy array of local flow direction
    """
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
                # print('First corner')
                window = flowdir[:i + 2, : j + 2]
                windir = reverse_directions[1:, 1:]
            elif i == 0 and j == colmaxid:
                # print('Second corner')
                window = flowdir[:2, j - 1:]
                windir = reverse_directions[1:, :2]
            elif i == rowmaxid and j == 0:
                # print('Third corner')
                window = flowdir[i - 1:, :j + 2]
                windir = reverse_directions[:2, 1:]
            elif i == rowmaxid and j == colmaxid:
                # print('Forth corner')
                window = flowdir[i - 1:, j - 1:]
                windir = reverse_directions[:2, :2]
            elif i == 0 and 0 < j < colmaxid:
                # print('Upper edge')
                window = flowdir[:i + 2, j - 1: j + 2]
                windir = reverse_directions[1:, :]
            elif i == rowmaxid and 0 < j < colmaxid:
                # print('Lower edge')
                window = flowdir[i - 1:, j - 1: j + 2]
                windir = reverse_directions[:2, :]
            elif 0 < i < rowmaxid and j == 0:
                # print('Left edge')
                window = flowdir[i - 1: i + 2, :j + 2]
                windir = reverse_directions[:, 1:]
            elif 0 < i < rowmaxid and j == colmaxid:
                # print('Right edge')
                window = flowdir[i - 1: i + 2, j - 1:]
                windir = reverse_directions[:, :2]
            else:
                # print('Bulk')
                window = flowdir[i - 1: i + 2, j - 1: j + 2]
                windir = reverse_directions
                # print(window)
            # count number of poiting neighbor cells
            lcl_count = np.sum(1.0 * (window == windir))
            # insert on flow dir array
            local_accum[i][j] = lcl_count
    return local_accum


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


def grad(slope):
    """
    derive the topographical gradient tan(B) from the slope in degrees
    :param array: slope in degrees 2d array
    :return: gradient (m/m) of terrain
    """
    slope_rad = np.pi * 2 * slope / 360
    grad = np.tan(slope_rad)
    return grad


def least_cost_path(cost_map, x0, y0, xf, yf,
                    popsize=100,
                    generations=30,
                    pathf=2.0,
                    offsfrac=1,
                    mutrate=0.95,
                    puremutrate=0.9,
                    cutfrac=0.2,
                    tracefrac=1,
                    tui=False):
    import pandas as pd
    import evolution
    from datetime import datetime
    from sys import getsizeof
    if tui:
        from tui import status
    #
    # deploy random state
    seed = int(str(datetime.now())[-6:])
    np.random.seed(seed)
    #
    # set target to zero cost
    _solution = cost_map.copy()
    _solution[yf][xf] = 0
    #
    # deploy nucletides for 2 genes
    _nucleo = ((1, 0, -1), (1, 0, -1))
    #
    # get path size
    _dy = yf - y0
    _dx = xf - x0
    _euclid_dist = np.sqrt((_dx * _dx) + (_dy * _dy))
    print(_euclid_dist)
    _path_size = int(pathf * _euclid_dist)
    print(_path_size)
    #
    # get gene sizes
    _gene_size = (_path_size, _path_size)
    #
    #
    #
    # -------- Generate candidate paths --------
    if tui:
        status('generating initial population')
    parents = evolution.generate_population(nucleotides=_nucleo,
                                            genesizes=_gene_size,
                                            popsize=popsize)
    if tui:
        status('generating candidate paths')
    pop_dct = dict()  # population dictionary
    trace_lst = list()
    g = 0
    while True:
        if tui:
            print('\n')
            status('running generation: {}'.format(g))
        # REPRODUCE the fitting population
        if g == 0:
            population = parents
        else:
            # get offstring
            offsize = int(offsfrac * len(parents))
            if tui:
                status('generating offspring')
            population = evolution.generate_offspring(parents,
                                                      offsize=offsize,
                                                      nucleotides=_nucleo,
                                                      mutrate=mutrate,
                                                      puremutrate=puremutrate,
                                                      cutfrac=cutfrac)
        if tui:
            status('population size: {} DNAs'.format(len(population)))
            status('population size in memory: {} KB'.format(getsizeof(population)))
        #
        #
        # 4) FIT new population
        ids_lst = list()
        ids_set_lst = list()
        scores_lst = list()
        # loop in individuals
        for i in range(len(population)):
            # get local score and id:
            lcl_dna = population[i]  # local dna
            lcl_set_id = str(lcl_dna)  # id is the string conversion of DNA
            #
            # express genes
            _lcl_x_path = evolution.express_path(gene=lcl_dna[0], p0=x0)
            _lcl_y_path = evolution.express_path(gene=lcl_dna[1], p0=y0)
            #
            # compute distance score:
            _dx = xf - _lcl_x_path[-1]
            _dy = yf - _lcl_y_path[-1]
            _prox = np.sqrt((_dx * _dx) + (_dy * _dy))
            #
            # compure DNA score:
            lcl_dna_score = _prox
            #
            # tag DNA
            lcl_dna_id = 'G' + str(g) + '-' + str(i)
            # store in retrieval system:
            pop_dct[lcl_dna_id] = lcl_dna
            #
            ids_lst.append(lcl_dna_id)
            ids_set_lst.append(lcl_set_id)
            scores_lst.append(lcl_dna_score)
        #
        # RECRUIT:
        if g == 0:
            df_parents_rank = pd.DataFrame({'Id':ids_lst,
                                            'SetIds':ids_set_lst[:],
                                            'Score':scores_lst[:]
                                            })
        else:
            df_offspring_rank = pd.DataFrame({'Id':ids_lst,
                                              'SetIds':ids_set_lst[:],
                                              'Score':scores_lst[:],
                                              })
            if tui:
                print('offspring:')
                print(df_offspring_rank[['Id', 'Score']].head(5).to_string())
            # append to existing dataframe
            df_parents_rank = df_parents_rank.append(df_offspring_rank, ignore_index=True)
        #
        # RANK population
        df_parents_rank.drop_duplicates(subset='SetIds', inplace=True, ignore_index=True)
        df_parents_rank.sort_values(by='Score', ascending=True, inplace=True, ignore_index=True)
        #
        # SELECT mating pool
        df_parents_rank = df_parents_rank.head(len(parents))
        if tui:
            print('new parents:')
            print(df_parents_rank[['Id', 'Score']].head(5).to_string())
        #
        # retrieve parents DNAs to a list:
        parents_ids = df_parents_rank['Id'].values  # numpy array of string IDs
        parents_lst = list()
        for i in range(len(parents_ids)):
            parents_lst.append(pop_dct[parents_ids[i]])
        # recicle index
        pop_dct = dict()
        for i in range(len(parents_ids)):
            pop_dct[parents_ids[i]] = parents_lst[i]
        # Set new parents
        parents = tuple(parents_lst)  # parents DNAs
        if np.min(df_parents_rank['Score'].values) <= 30:
            break
        else:
            g = g + 1
    #
    #
    #
    # -------- Generate least cost paths --------
    #
    if tui:
        status('generating least cost paths')
    trace = list()
    for g in range(generations):
        if tui:
            print('\n')
            status('running generation: {}'.format(g))
        # REPRODUCE the fitting population
        if g == 0:
            population = parents
        else:
            # get offstring
            offsize = int(offsfrac * len(parents))
            if tui:
                status('generating offspring')
            population = evolution.generate_offspring(parents,
                                                      offsize=offsize,
                                                      nucleotides=_nucleo,
                                                      mutrate=mutrate,
                                                      puremutrate=puremutrate,
                                                      cutfrac=cutfrac)
        if tui:
            status('population size: {} DNAs'.format(len(population)))
            status('population size in memory: {} KB'.format(getsizeof(population)))
        #
        #
        # 4) FIT new population
        ids_lst = list()
        ids_set_lst = list()
        scores_lst = list()
        prox_lst = list()
        cost_lst = list()
        # loop in individuals
        for i in range(len(population)):
            # get local score and id:
            lcl_dna = population[i]  # local dna
            lcl_set_id = str(lcl_dna)  # id is the string conversion of DNA
            #
            # express genes
            _lcl_x_path = evolution.express_path(gene=lcl_dna[0], p0=x0)
            _lcl_y_path = evolution.express_path(gene=lcl_dna[1], p0=y0)
            #
            # compute distance score:
            _dx = xf - _lcl_x_path[-1]
            _dy = yf - _lcl_y_path[-1]
            _prox = np.sqrt((_dx * _dx) + (_dy * _dy))
            # compure path score:
            _cost = 0
            # accumulate cost
            for t in range(len(_lcl_x_path) - 1):
                _cost = _cost + _solution[int(_lcl_y_path[t])][int(_lcl_x_path[t])]
            #
            # compure DNA score:
            _w_prox = 100
            _w_cost = 1
            lcl_dna_score = np.sqrt((_w_prox * (_prox * _prox)) + (_w_cost * (_cost * _cost)))
            #
            # tag DNA
            lcl_dna_id = 'G' + str(g) + '-' + str(i)
            # store in retrieval system:
            pop_dct[lcl_dna_id] = lcl_dna
            #
            ids_lst.append(lcl_dna_id)
            ids_set_lst.append(lcl_set_id)
            scores_lst.append(lcl_dna_score)
            prox_lst.append(_prox)
            cost_lst.append(_cost)
        #
        # RECRUIT:
        if g == 0:
            df_parents_rank = pd.DataFrame({'Id':ids_lst,
                                            'SetIds':ids_set_lst[:],
                                            'Score':scores_lst[:],
                                            'Prox': prox_lst[:],
                                            'Cost': cost_lst[:]
                                            })
        else:
            df_offspring_rank = pd.DataFrame({'Id':ids_lst,
                                              'SetIds':ids_set_lst[:],
                                              'Score':scores_lst[:],
                                              'Prox': prox_lst[:],
                                              'Cost': cost_lst[:]
                                              })
            if tui:
                print('offspring:')
                print(df_offspring_rank[['Id', 'Score', 'Prox', 'Cost']].head(5).to_string())
            # append to existing dataframe
            df_parents_rank = df_parents_rank.append(df_offspring_rank, ignore_index=True)
        #
        # RANK population
        df_parents_rank.drop_duplicates(subset='SetIds', inplace=True, ignore_index=True)
        df_parents_rank.sort_values(by='Score', ascending=True, inplace=True, ignore_index=True)
        #
        # SELECT mating pool
        df_parents_rank = df_parents_rank.head(len(parents))
        if tui:
            print('new parents:')
            print(df_parents_rank[['Id', 'Score', 'Prox', 'Cost']].head(5).to_string())
        #
        # retrieve parents DNAs to a list:
        parents_ids = df_parents_rank['Id'].values  # numpy array of string IDs
        parents_scores = df_parents_rank['Score'].values  # numpy array of float scores
        parents_prox = df_parents_rank['Prox'].values
        parents_cost = df_parents_rank['Cost'].values
        parents_lst = list()
        for i in range(len(parents_ids)):
            parents_lst.append(pop_dct[parents_ids[i]])
        # recicle index
        pop_dct = dict()
        for i in range(len(parents_ids)):
            pop_dct[parents_ids[i]] = parents_lst[i]
        # Set new parents
        parents = tuple(parents_lst)  # parents DNAs
        #
        #
        tr_len = int(len(df_parents_rank) * tracefrac)
        print('>>> {}'.format(tr_len))
        #
        # trace parents
        trace.append({'DNAs': parents[:tr_len],
                      'Ids': parents_ids[:tr_len],
                      'Scores': parents_scores[:tr_len],
                      'Prox': parents_prox[:tr_len],
                      'Cost': parents_cost[:tr_len]
                      })
        if tui:
            print('Trace size: {} KB'.format(getsizeof(trace)))
            print('Index size: {} KB'.format(getsizeof(pop_dct)))
            print('Trace len: {}'.format(len(trace)))

    return trace



def local_load(avg_load, proxy, aoi):
    """
    compute a local load map for a loading variable

    load_i = avg_load * proxy_i / avg_proxy

    :param avg_load: float average load
    :param proxy: 2d numpy array of proxy map
    :param aoi: 2d numpy array of AOI (pseudo-boolean of 0 and 1)
    :return: local load map
    """
    # average inside aoi
    _avg_proxy = np.sum(proxy * aoi) / np.sum(aoi)
    # if no process occurs, load is zero
    if _avg_proxy == 0:
        avg_load = 0
        _avg_proxy = 1
    # weights
    _weights = proxy / _avg_proxy
    return avg_load * _weights * aoi


def local_loads(loads, zones_ids, zones, proxy):
    """
    Compute the local loads considering average loads per zone type and a proxy map
    :param loads: 1d numpy array of average loads per zones
    :param zones_ids: 1d numpy array of zones ids values
    :param zones: 2d numpy array of zones map
    :param proxy: 2d numpy array proxy map
    :return:
    """
    # deploy a new array:
    _loads = np.zeros(shape=np.shape(proxy))
    for i in range(len(zones_ids)):
        _lcl_aoi = 1.0 * (zones == zones_ids[i])
        if np.sum(_lcl_aoi) == 0: # zone id not found
            _lcl_local_load = np.zeros(shape=np.shape(proxy))
        else:
            _lcl_local_load = local_load(avg_load=loads[i], proxy=proxy, aoi=_lcl_aoi)
        _loads = _loads + _lcl_local_load
    return _loads


def local_stats(maps, tui=False):
    """
    Compute the local stats of a pile of maps
    :param maps: 3d numpy array with the 2d maps
    :return: dictionary with the local stats maps (2d numpy arrays)
    """
    if tui:
        size = len(maps) * np.shape(maps[0])[0] * np.shape(maps[0])[1]
        print('Size: {}'.format(size))
        _count = 0
    # declare stats arrays
    _median = np.zeros(shape=np.shape(maps[0]))
    _mean = np.zeros(shape=np.shape(maps[0]))
    _max = np.zeros(shape=np.shape(maps[0]))
    _min = np.zeros(shape=np.shape(maps[0]))
    _5th_ptile = np.zeros(shape=np.shape(maps[0]))
    _50th_ptile = np.zeros(shape=np.shape(maps[0]))
    _95th_ptile = np.zeros(shape=np.shape(maps[0]))
    _range_90 = np.zeros(shape=np.shape(maps[0]))
    _range_minmax = np.zeros(shape=np.shape(maps[0]))
    # scan the arrays
    for i in range(len(maps[0])):
        if tui:
            _status = 100 * _count / size
            if _status % 5 < 0.1:
                print('Status: {:.3f}%'.format(_status))
        for j in range(len(maps[0][0])):
            # built a cross array for every cell
            _x_values = np.zeros(shape=len(maps))
            for k in range(len(_x_values)):
                _x_values[k] = maps[k][i][j]
                if tui:
                    _count = _count + 1
            #
            # sample stas
            _median[i][j] = np.median(_x_values)
            _mean[i][j] = np.mean(_x_values)
            _min[i][j] = np.min(_x_values)
            _max[i][j] = np.max(_x_values)
            _5th_ptile[i][j] = np.percentile(_x_values, q=5)
            _50th_ptile[i][j] = np.percentile(_x_values, q=50)
            _95th_ptile[i][j] = np.percentile(_x_values, q=95)
    _range_90 = _95th_ptile - _50th_ptile
    _range_minmax = _max - _min
    return {'Median':_median,
            'Mean': _mean,
            'Min': _min,
            'Max': _max,
            '5th_p': _5th_ptile,
            '50th_p': _5th_ptile,
            '95th_p': _95th_ptile,
            'Range_90': _range_90,
            'Range_minmax': _range_minmax}


def mask(array, mask):
    """
    utility function for masking an array
    :param array: 2d numpy array
    :param mask: 2d numpy pseudo-boolean array (1 and 0)
    :return: masked 2d numpy array (nan assigned to 0 values on mask)
    """
    masked = np.copy(array)
    masked[mask == 0] = np.nan
    return masked


def ndvi(nir, red):
    """
    Normalized Difference Vegetation Index - NDVI
    :param nir: 2d numpy array of NIR (Near Infrared) - Band 5 of Landsat 8
    :param red: 2d numpy array of Red - Band 4 of Landsat 8
    :return: 2d numpy array of NDVI
    """
    return (nir - red) / (nir + red)


def ndwi_v(nir, swir):
    """
    Normalized Difference Wetness Index of Vegetation - NDWIv
    :param nir: 2d numpy array of NIR (Near Infrared) - Band 5 of Landsat 8
    :param swir: 2d numpy array of SWIR (Shortwave Infrared) - Band 6 of Landsat 8
    :return: 2d numpy array of NDWI_v
    """
    return (nir - swir) / (nir + swir)


def ndwi_w(green, nir):
    """
    Normalized Difference Wetness Index of Water - NDWIw
    :param green: 2d numpy array of Green - Band 3 of Landsat 8
    :param nir: 2d numpy array of NIR (Near Infrared) - Band 5 of Landsat 8
    :return: 2d numpy array of NDWIw
    """
    return (green - nir) / (green + nir)


def reclassify(array, upvalues, classes):
    """
    utility function -
    Reclassify array based on list of upper values and list of classes values

    :param array: 2d numpy array to reclassify
    :param upvalues: 1d numpy array of upper values
    :param classes: 1d array of classes values
    :return: 2d numpy array reclassified
    """
    new = array * 0.0
    for i in range(len(upvalues)):
        if i == 0:
            new = new + ((array <= upvalues[i]) * classes[i])
        else:
            new = new + ((array > upvalues[i - 1]) * (array <= upvalues[i]) * classes[i])
    return new


def rusle_l(slope, cellsize):
    """
    RUSLE L Factor (McCool et al. 1989; USDA, 1997)

    L = (x / 22.13) ^ m

    where:
    m = β / (1 + β)
    and:
    β = (sinθ ⁄ 0.0896) ⁄ (3.0⋅(sinθ)^0.8 + 0.56)

    x is the plot lenght taken as 1.4142 * cellsize  (diagonal length of cell)

    :param slope: slope in degrees of terrain 2d array
    :param cellsize: cell size in meters
    :return: RUSLE L factor 2d array
    """
    slope_rad = np.pi * 2 * slope / 360
    lcl_grad = np.sin(slope_rad)
    beta = (lcl_grad / 0.0896) / ((3 * np.power(lcl_grad, 0.8)) + 0.56)
    m = beta / (1.0 + beta)
    return np.power(np.sqrt(2) * cellsize / 22.13, m)


def rusle_s(slope):
    """
    RUSLE S Factor (McCool et al. 1987; USDA, 1997)

    S = 10.8 sinθ + 0.03     sinθ < 0.09
    S = 16.8 sinθ - 0.5      sinθ >= 0.09

    :param slope: slope in degrees of terrain 2d array
    :return: RUSLE S factor 2d array
    """
    slope_rad = np.pi * 2 * slope / 360
    lcl_grad = np.sin(slope_rad)
    lcl_s = ((10.8 * lcl_grad + 0.03 ) * (lcl_grad < 0.09)) + ((16.8 * lcl_grad - 0.5 ) * (lcl_grad >= 0.09))
    return lcl_s


def slope(dem, cellsize, degree=True):
    """
    Slope algorithm based on gradient built in functions of numpy
    :param dem: 2d numpy array of dem
    :param cellsize: float value of cellsize (delta x = delta y)
    :param degree: boolean to control output units. Default = True. If False output units are in radians
    :return: 2d numpy array of slope
    """
    grad = np.gradient(dem)
    gradx = grad[0] / cellsize
    grady = grad[1] / cellsize
    gradv = np.sqrt((gradx * gradx) + (grady * grady))
    slope_array = np.arctan(gradv)
    if degree:
        slope_array = slope_array * 360 / (2 * np.pi)
    return slope_array


def soils_landforms(landforms):
    """
    Landform-based soil classification
    :param landforms: numpy array of 10 landform classes:

    11 - valley bottoms
    12 - valleys
    13 - valley borders
    21 - Mesas
    22 - Gentle hillslopes
    23 - Ridges
    24 - Plains
    31 - Hill thalweg
    32 - Hill tops
    33 - Peaks

    :return: numpy array of soil classification

    1 - Alluvial
    2 - Colluvial
    3 - Residual
    4 - Neosols

    """

    alluvial = 1 * (landforms == 12) # valley
    colluvial = 2 * ((landforms == 11) + (landforms == 13)) # valley bottom and valley border
    residuals = 3 * ((landforms == 31) + (landforms == 32) + (landforms == 21) + (landforms == 22) + (landforms == 23) + (landforms == 24))
    neosols = 4 * (landforms == 33)  # peaks

    soils_land = alluvial + colluvial + residuals + neosols
    return soils_land


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


def stdmask():
    """
    Utility function for code development
    :return: asc file metadata dictionary and 2d numpy array of standard dem
    """
    lst = [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0),
           (0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0),
           (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0),
           (0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0),
           (0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0),
           (0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0),
           (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0),
           (0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0),
           (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    dct = {'ncols': '11',
           'nrows': '11',
           'xllcorner': '559490.0',
           'yllcorner': '6704830.0',
           'cellsize': '100',
           'NODATA_value': '-9999'}
    return dct, np.array(lst)


def twi(catcha, grad, fto, cellsize, gradmin=0.0001):
    """
    Derive the Topographical Wetness Index of TOPMODEL (Beven & Kirkby, 1979)

    TWI =  ln ( a / To tanB )

    Where a is the specific catchment area, To is the local transmissivity and tanB is the terrain gradient.

    :param catcha: 2d array - cathment area in m2
    :param grad: 2d array - gradient of terrain (tangent of slope) unitless
    :param fto: 2d array - local transmissivity factor of basin average - unitless
    :param cellsize: cell size in meters
    :param gradmin: minimun gradient threshold
    :return: Topographical Wetness Index 2d array
    """
    return np.log(catcha / (cellsize * fto * (grad + gradmin)))


def twito(twi, fto):
    """
    Derive the complete TWI raster map by inserting the To (transmissivity) term to the TWI formula.

    If TWI =  ln ( a / To tanB ), then
    TWI =  ln ( a / tanB ) + ln ( 1 / To )

    :param twi: 2d numpy array of TWI  map computed without the To term, i.e., only ln ( a / tanB )
    :param fto: 2d numpy array of fTo raster map
    :return: 2d numpy array of TWIto map
    """
    return twi + np.log(1 / fto)


def twi_hand_long(catcha, slope, hand, cellsize, hand_hi=15.0, hand_lo=0.0, hand_w=1, twi_w=1, gradmin=0.0001):
    """

    HAND enhanced TWI map method.

    :param catcha: 2d array - cathment area in m2
    :param slope: 2d array - slope in degrees
    :param fto: 2d array - local transmissivity factor of basin average - unitless
    :param hand: 2d array - Height Above Nearest Drainage
    :param cellsize: float - cell size in meters
    :param hand_hi: float - HAND higher threshold
    :param hand_lo: float - HAND lower threshold
    :param hand_w: float - HAND weight factor
    :param twi_w: float - TWI weight factor
    :param gradmin: float - minimun gradient threshold
    :return: 2d numpy array of HAND-enhanced TWI map
    """
    # get grad
    grad_map = grad(slope)
    # get twi without fto heterogeneity:
    fto = 1.0 * (grad_map * 0.0)
    twi_map = twi(catcha, grad_map, fto, cellsize=cellsize)
    #
    # fuzify twi
    twi_lo = np.min(twi_map)
    twi_hi = np.max(twi_map)
    twi_fuzz = fuzzy_transition(twi_map, a=twi_lo, b=twi_hi, ascending=True)
    #
    # fuzify hand
    hand_fuzz = fuzzy_transition(hand, a=hand_lo, b=hand_hi, ascending=False)
    #
    # compound twi with hand:
    twi_comp = hand_w * hand_fuzz + twi_w * twi_fuzz
    #
    # fuzify again to restore twi range
    twi_hand_map = twi_hi * fuzzy_transition(twi_comp, a=np.min(twi_comp), b=np.max(twi_comp))
    return twi_hand_map


def twi_hand_short(twi, hand, hand_hi=15.0, hand_lo=0.0, hand_w=1, twi_w=1):
    """

    HAND enhanced TWI map method. Short method.

    :param twi: 2d array - TWI - Topographical Wetness Index
    :param hand: 2d array - HAND - Height Above Nearest Drainage
    :param cellsize: float - cell size in meters
    :param hand_hi: float - HAND higher threshold
    :param hand_lo: float - HAND lower threshold
    :param hand_w: float - HAND weight factor
    :param twi_w: float - TWI weight factor
    :return: 2d numpy array of HAND-enhanced TWI map
    """
    # fuzify twi
    twi_lo = 0.0 # np.min(twi)
    twi_hi = np.max(twi)
    twi_fuzz = fuzzy_transition(twi, a=twi_lo, b=twi_hi, ascending=True )
    #
    # fuzify hand
    hand_fuzz = fuzzy_transition(hand, a=hand_lo, b=hand_hi, ascending=False )
    # OLD CODE
    # compound twi:
    twi_comp = hand_w * hand_fuzz + twi_w * twi_fuzz
    #
    # fuzify again to restore twi range
    htwi = np.max(twi) * fuzzy_transition(twi_comp, a=np.min(twi_comp), b=np.max(twi_comp), ascending=True, type='linear')
    return htwi


def usle_l(slope, cellsize):
    """
    Wischmeier & Smith (1978) L factor

    L = (x / 22.13) ^ m

    where:

    m = 0.2 when sinθ < 0.01;
    m = 0.3 when 0.01 ≤ sinθ ≤ 0.03;
    m = 0.4 when 0.03 < sinθ < 0.05;
    m = 0.5 when sinθ ≥ 0.05

    x is the plot lenght taken as 1.4142 * cellsize  (diagonal length of cell)

    :param slope: slope in degrees of terrain 2d array
    :param cellsize: cell size in meters
    :return: Wischmeier & Smith (1978) L factor 2d array
    """
    slope_rad = np.pi * 2 * slope / 360
    lcl_grad = np.sin(slope_rad)
    m = reclassify(lcl_grad, upvalues=(0.01, 0.03, 0.05, np.max(lcl_grad)), classes=(0.2, 0.3, 0.4, 0.5))
    return np.power(np.sqrt(2) * cellsize / 22.13, m)


def usle_s(slope):
    """
    Wischmeier & Smith (1978) S factor

    S = 65.41(sinθ)^2 + 4.56sinθ + 0.065

    :param slope: slope in degrees of terrain 2d array
    :return:
    """
    slope_rad = np.pi * 2 * slope / 360
    lcl_grad = np.sin(slope_rad)
    return (65.41 * np.power(lcl_grad, 2)) + (4.56 * lcl_grad) + 0.065


def usle_m_a(q, prec, r, k, l, s, c, p, cellsize=30):
    """

    USLE-M Annual Soil Loss (Kinnell & Risse, 1998)

    :param q: 2d numpy array of annual runoff in mm / year
    :param prec: 2d numpy array or float of annual precipitation in mm / year
    :param r: 2d numpy array or float of rain erosivity in MJ mm h-1 ha-1 year-1
    :param k: 2d numpy array or float of erodibility K factor in ton h MJ-1 mm-1
    :param l: 2d numpy array or float of USLE/RUSLE L factor
    :param s: 2d numpy array or float of USLE/RUSLE S factor
    :param c: 2d numpy array or float of C_UM factor
    :param p: 2d numpy array or float of USLE/RUSLE P factor
    :param cellsize: float of grid cell size in meters
    :return: 2d numpy array of Annual Soil Loss in ton / year
    """
    return (q / prec) * r * k * l * s * c * p * (cellsize * cellsize / (100 * 100))


def write_wkt_points(x_long, y_lat):
    """
    Writes on a tuple Points WKT geometry strings
    :param x_long: 1d array of x - longitude vertex coordinates
    :param y_lat: 1d array of y - latitude vertex coordinates
    :return: tuple of Points WKT geometry strings
    """

    wkt_lst = list()
    for i in range(len(x_long)):
        wkt_lst.append('Point ({} {})'.format(x_long[i], y_lat[i]))
    return tuple(wkt_lst)


def write_wkt_linestring(x_long, y_lat):
    """
    Writes a string of WKT LineString geometry
    :param x_long: 1d array of x - longitude vertex coordinates
    :param y_lat: 1d array of y - latitude vertex coordinates
    :return: string of WKT LineString geometry
    """
    vertex_lst = list()
    for i in range(len(x_long)):
        vertex_lst.append('{} {}'.format(x_long[i], y_lat[i]))
    argument = vertex_lst[0]
    for i in range(1, len(vertex_lst)):
        argument = argument + ', ' + vertex_lst[i]
    return 'LineString ({})'.format(argument)


def write_wkt_polygon(x_long, y_lat):
    """
    Writes a string of WKT Polygon geometry
    :param x_long: 1d array of x - longitude vertex coordinates
    :param y_lat: 1d array of y - latitude vertex coordinates
    :return: a string of WKT Polygon geometry
    """
    vertex_lst = list()
    for i in range(len(x_long)):
        vertex_lst.append('{} {}'.format(x_long[i], y_lat[i]))
    argument = vertex_lst[0]
    for i in range(1, len(vertex_lst)):
        argument = argument + ', ' + vertex_lst[i]
    return 'Polygon (({}))'.format(argument)


def zonalstats(field, zones, tui=False):
    """
    Performs a zonal statistics over a 2d numpy array
    :param field: 2d numpy array map of the field variable
    :param zones: 2d numpy array map of the zones
    :return: dictionary of the zonal statistics
    """
    # internal aux function
    def lcl_flatten_clear(array, mask):
        """
        Flatten and clear a 2d numpy array using a mask pseudo-boolean array
        :param array: field 2d numpy array
        :param mask: mask pseudo-boolean 2d numpy array
        :return: 1d numpy array of cleared values
        """
        masked = np.copy(array)
        masked[mask == 0] = np.nan
        flatten = masked.flatten()
        cleared = masked[~np.isnan(masked)]
        return cleared
    #
    # load lists
    cnt_lst = list()
    sum_lst = list()
    avg_lst = list()
    max_lst = list()
    min_lst = list()
    std_lst = list()
    med_lst = list()
    #
    # loop in zones
    zone_values = np.unique(zones)
    for i in range(len(zone_values)):
        if tui:
            print('Zone #:{}'.format(zone_values[i]))
        #
        # compute mask
        lcl_mask = (zones == zone_values[i]) * 1.0
        #
        # compute cleared array
        lcl_cleared = lcl_flatten_clear(array=field, mask=lcl_mask)
        if len(lcl_cleared) == 0:
            pass
        else:
            #
            # load stats to lists
            cnt_lst.append(len(lcl_cleared))
            sum_lst.append(np.sum(lcl_cleared))
            avg_lst.append(np.mean(lcl_cleared))
            max_lst.append(np.max(lcl_cleared))
            min_lst.append(np.min(lcl_cleared))
            std_lst.append(np.std(lcl_cleared))
            med_lst.append(np.median(lcl_cleared))
    # load to dict
    out_dct = {'Zones':zone_values,
               'Count':cnt_lst,
               'Sum':sum_lst,
               'Mean':avg_lst,
               'Max':max_lst,
               'Min':min_lst,
               'SD':std_lst,
               'Median':med_lst
                }
    return out_dct

