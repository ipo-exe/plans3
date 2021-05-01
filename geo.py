import numpy as np
'''
This is the PLANS 3 geoprocessing module

'''


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

    :param catcha: 2d array - cathment area in m2
    :param grad: 2d array - gradient of terrain (tangent of slope) unitless
    :param fto: 2d array - local transmissivity factor of basin average - unitless
    :param cellsize: cell size in meters
    :param gradmin: minimun gradient threshold
    :return: Topographical Wetness Index 2d array
    """
    return np.log(catcha / (cellsize * fto * (grad + gradmin)))


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


