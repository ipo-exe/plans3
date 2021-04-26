"""
Deprecated code bin

"""

# deprecated:
def topmodel_s0max(cn, a):
    """
    Plans 3 Model of S0max as a function of CN

    s0max = a * (100 - CN + b)

    :param cn: float or nd array of CN
    :param a: scaling parameter in mm/CN
    :return: float or nd array of S0max
    """
    lcl_s0max = a * (100 - cn + 0.1)
    return lcl_s0max


# deprecated:
def topmodel_hist(twi, cn, aoi, twibins=20, cnbins=10):
    """
    2D histogram for TOPMODEL in PLANS 3. Crossed variables: TWI and CN
    :param twi: 2d numpy array of TWI
    :param cn: 2d numpy array of CN
    :param aoi: 2d numpy array of AOI (Area of Interest)
    :param twibins: int number of bins in TWI histogram
    :param cnbins: int number of bins in CN histogram
    :return:
    1) 2d histogram/count (rows - TWI, columns - CN) matrix
    2) tuple of histogram of TWI: fist: bins - 1d numpy array, second: count - 1d numpy array
    3) tuple of histogram of CN: fist: bins - 1d numpy array, second: count - 1d numpy array
    """
    countmatrix, twi_hist, cn_hist = count_matrix(array2d1=twi, array2d2=cn, bins1=twibins, bins2=cnbins, aoi=aoi)
    return countmatrix, twi_hist, cn_hist

# deprecated:
def topmodel_sim_deprec(series, twihist, cnhist, countmatrix, lamb, ksat, m, qo, a, c, lat, qt0, k, n,
                 mapback=False, mapvar='R-ET-S1-S2', tui=False, qobs=False):
    """

    PLANS 3 TOPMODEL simulation procedure

    :param series: Pandas DataFrame of input series.
    Required fields: 'Date', 'Prec', 'Temp'. Optional: 'Q' (Q obs, in mm)
    :param twihist: tuple of histogram of TWI
    :param cnhist: tuple of histogram of CN
    :param countmatrix: 2D histogram of TWI and CN
    :param lamb: positive float - average TWI value of the AOI
    :param ksat: positive float - effective saturated hydraulic conductivity in mm/d
    :param m: positive float - effective transmissivity decay coefficient in mm
    :param qo: positive float - max baseflow when d=0 in mm/d
    :param a: positive float - scaling parameter for S0max model
    :param c: positive float - scaling parameter for PET model in Celcius
    :param lat: float - latitude in degrees for PET model
    :param qt0: positive float - baseflow at t=0 in mm/d
    :param k: positive float - Nash Cascade residence time in days
    :param n: positive float - equivalent number of reservoirs in Nash Cascade
    :param mapback: boolean control to map back variables
    :param mapvar: string code of variables to map back. Available variables:
    'TF', Qv', 'R', 'ET', 'S1', 'S2', 'Inf', 'Tp', 'Ev', 'Tpgw' (see below the relation)
    :param tui: boolean to control terminal messages
    :return: Pandas DataFrame of simulated variables:

    'Date': date (from input)
    'Prec': precipitation (from input), mm
    'Temp': temperature (from input), deg C
    'PET': simulated potential evapotranspiration, mm
    'S1': simulated water in S1 stock (canopy interceptation), mm
    'TF': simulated throughfall, mm
    'Ev': simulated evaporation from canopy, mm
    'S2': simulated water in S2 stock (unsaturated zone), mm
    'Inf': simulated infiltration, mm
    'R': simulated overland runoff, mm
    'Tp': simulated transpiration from unsaturated zone (S2), mm
    'Tpgw': simulated transpiration from the saturated zone, mm
    'ET': simulated actual evapotranspiration, mm
    'D': simulated soil water deficit, mm
    'Qb': simualted baseflow, mm
    'Qv':simulated recharge, mm
    'Qs':simulated surface flow, mm
    'Q': simulated streamflow, mm  (Q = Qb + Qs)
    'VSA': simulated variable source area (saturated areas), in %

    And
    if mapback=True:
    Dictionary of encoded 2d numpy arrays maps
    Keys to access maps: 'TF', Qv', 'R', 'ET', 'S1', 'S2', 'Inf', 'Tp', 'Ev', 'Tpgw'
    Each key stores an array of 2d numpy arrays (i.e., 3d array) in the ascending order of the time series.

    """
    #
    # extract data input
    ts_prec = series['Prec'].values
    ts_temp = series['Temp'].values
    size = len(ts_prec)
    #
    # compute PET
    days = series['Date'].dt.dayofyear
    ts_days = days.values
    lat = lat * np.pi / 180  # convet lat to radians
    ts_pet = pet_oudin(temperature=ts_temp, day=ts_days, latitude=lat, k1=c)
    #
    # set initial conditions
    d0 = topmodel_d0(qt0=qt0, qo=qo, m=m)
    #
    twi_bins = twihist[0]
    twi_count = twihist[1]
    s0max_bins = topmodel_s0max(cn=cnhist[0], a=a)  # convert cn to S0max
    #print(s0max_bins)
    s0max_count = cnhist[1]
    shape = np.shape(countmatrix)
    rows = shape[0]
    cols = shape[1]
    #
    # set 2d count parameter arrays
    s1maxi = 0.2 * s0max_bins * np.ones(shape=shape, dtype='float32')  # canopy water
    s2maxi = 0.8 * s0max_bins * np.ones(shape=shape, dtype='float32')  # rootzone
    rzdi = s2maxi.copy()
    #
    # get local Lambda
    lambi = np.reshape(twi_bins, (rows, 1)) * np.ones(shape=shape, dtype='float32')
    #
    # set 2d count variable arrays
    preci = ts_prec[0] * np.ones(shape=shape)
    peti = ts_pet[0] * np.ones(shape=shape)
    s1i = np.zeros(shape=shape)  # initial condition
    tfi = np.zeros(shape=shape)
    evi = np.zeros(shape=shape)
    qri = np.zeros(shape=shape)
    #
    s2i = np.zeros(shape=shape)  # initial condition
    infi = np.zeros(shape=shape)
    ri = np.zeros(shape=shape)
    tpi = np.zeros(shape=shape)
    tpgwi = np.zeros(shape=shape)
    eti = evi + tpi + tpgwi
    #
    #s3i = np.zeros(shape=shape)  # initial condition
    di = topmodel_di(d=d0, twi=lambi, m=m, lamb=lamb)
    vsai = topmodel_vsai(di=di)
    qvi = np.zeros(shape=shape)
    #
    # set stocks time series arrays and initial conditions
    ts_s1 = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_s1[0] = avg_2d(var2d=s1i, weight=countmatrix)
    ts_s2 = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_s2[0] = avg_2d(var2d=s2i, weight=countmatrix)
    ts_d = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_d[0] = d0
    ts_qv = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_qv[0] = avg_2d(var2d=qvi, weight=countmatrix)
    ts_qb = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_qb[0] = qt0
    ts_vsa = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_vsa[0] = np.sum(vsai * countmatrix) / np.sum(countmatrix)
    #
    # set flows time series arrays
    ts_ev = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_tp = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_tpgw = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_et = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_r = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_inf = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_tf = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    #
    # Trace setup
    if mapback:
        mapvar_lst = mapvar.split('-')
        map_dct = dict()
        for e in mapvar_lst:
            map_dct[e] = np.zeros(shape=(size, rows, cols), dtype='float32')
    #
    if tui:
        print('Soil moisture accounting simulation...')
    # ESMA loop
    for t in range(1, size):
        #if tui:
        #   print('Step {}'.format(t))
        #
        # update S1 - Canopy storage - interceptation
        s1i = s1i - evi - tfi + preci
        ts_s1[t] = avg_2d(s1i, countmatrix)
        #
        # compute current EV - Evaporation from canopy storage
        peti = ts_pet[t] * np.ones(shape=shape)  # update PET
        evi = ((peti) * (s1i >= peti)) + (s1i * (s1i < peti))
        ts_ev[t] = avg_2d(evi, countmatrix)
        #
        # compute current TF - Throughfall (or "effective preciputation")
        preci = ts_prec[t] * np.ones(shape=shape)  # update PREC
        tfi = ((preci + s1i - evi - s1maxi) * ((preci + s1i - evi) >= s1maxi))
        ts_tf[t] = avg_2d(tfi, countmatrix)
        #
        # update S2 - Unsaturated soil water - vadoze zone
        s2i = s2i - qvi - tpi + infi
        ts_s2[t] = avg_2d(s2i, countmatrix)
        #
        # compute TP - Plant transpiration from vadoze zone
        peti = peti - evi  # update peti
        di_aux = di + ((np.max(di) + 3) * (di <= 0.0))  # auxiliar Di to replace zero values by a higher positive value to avoid division by zero
        ptpi = ((s2i * (rzdi >= di)) + ((s2i * rzdi / di_aux) * (rzdi < di))) * (di > 0.0)  # compute potential TP
        tpi = (ptpi * (peti >= ptpi)) + (peti * (peti < ptpi))
        ts_tp[t] = avg_2d(tpi, countmatrix)
        #
        # compute QV - Vertical flow to saturated zone - Water table recharge
        pqvi = (ksat * s2i / di_aux) * (di > 0.0)  # potential QV
        qvi = ((pqvi) * (s2i - tpi >= pqvi)) + ((s2i - tpi) * (s2i - tpi < pqvi))
        ts_qv[t] = avg_2d(qvi, countmatrix)
        #
        # compute Inf - Infiltration from surface. Soil has a potential infiltration capacity equal to the rootzone
        pinfi = (rzdi * (tfi >= rzdi)) + (tfi * (tfi < rzdi))  # potential infiltration -- infiltration capacity
        infi = ((di - s2i + tpi + qvi) * (pinfi >= (di - s2i + tpi + qvi))) + (pinfi * (pinfi < (di - s2i + tpi + qvi)))
        ts_inf[t] = avg_2d(infi, countmatrix)
        #
        # compute R - Runoff water
        ri = tfi - infi
        ts_r[t] = avg_2d(ri, countmatrix)
        #
        # compute TP-GW - Plant transpiration directly from the water table (if within reach of the root zone)
        peti = peti - tpi  # update peti
        ptpgwi = (rzdi - di) * (di < rzdi)  # potential local TP
        #ptpgwi = (s2maxi - di) * (di < s2maxi)  # potential local TP
        tpgwi = (peti * (ptpgwi > peti)) + (ptpgwi * (ptpgwi <= peti))
        ts_tpgw[t] = avg_2d(tpgwi, countmatrix)
        #
        # compute ET - Actual Evapo-transpiration
        eti = evi + tpi + tpgwi
        ts_et[t] = avg_2d(eti, countmatrix)
        #
        # update D water balance
        ts_d[t] = ts_d[t - 1] + ts_qb[t - 1] - ts_qv[t - 1] + ts_tpgw[t - 1]
        #
        # compute Qb - Baseflow
        ts_qb[t] = topmodel_qb(d=ts_d[t], qo=qo, m=m)
        #
        # Update Di
        di = topmodel_di(d=ts_d[t], twi=lambi, m=m, lamb=lamb)
        #
        # compute VSA
        vsai = topmodel_vsai(di=di)
        ts_vsa[t] = np.sum(vsai * countmatrix) / np.sum(countmatrix)
        #
        # trace section
        if mapback:
            dct = {'TF':tfi, 'Qv':qvi, 'R': ri, 'ET':eti, 'S1':s1i, 'S2':s2i, 'Inf':infi, 'Tp':tpi, 'Ev':evi,
                   'Tpgw':tpgwi}
            for e in mapvar_lst:
                map_dct[e][t] = dct[e]
    #
    # RUNOFF ROUTING by Nash Cascade of linear reservoirs
    if tui:
        print('Runoff routing...')
    ts_qs = nash_cascade(ts_r, k=k, n=n)
    #
    # compute full discharge Q = Qb + Qs
    ts_q = ts_qb + ts_qs
    #
    # export data
    exp_df = pd.DataFrame({'Date': series['Date'].values,
                           'Prec': series['Prec'].values,
                           'Temp': series['Temp'].values,
                           'PET': ts_pet,
                           'S1': np.round(ts_s1, 3), 'TF': np.round(ts_tf, 3), 'Ev': np.round(ts_ev, 3),
                           'S2': ts_s2, 'Inf': ts_inf, 'R': ts_r, 'Tp': ts_tp, 'Tpgw': ts_tpgw, 'ET': ts_et,
                           'D': ts_d, 'Qb': ts_qb, 'Qv': ts_qv, 'Qs': ts_qs, 'Q': ts_q, 'VSA': ts_vsa})
    if qobs:
        exp_df['Qobs'] = series['Q'].values
    #
    if mapback:
        return exp_df, map_dct
    else:
        return exp_df


# deprecated:
def map_cn(flulc, flulcparam, fsoils, fsoilsparam, folder='C:/bin', filename='cn'):
    """
    derive the CN map based on LULC and Soils groups
    :param flulc: string file path to lulc .asc raster file
    :param flulcparam: string file path to lulc parameters .txt file. Separator = ;
    Example:

     Value     Name  NBS  CN-A  CN-B  CN-C  CN-D
     1        Water    0   100   100   100   100
     2        Urban    0    77    85    90    92
     3       Forest    0    30    55    70    77
     4      Pasture    0    68    79    86    89
     5        Crops    0    72    81    88    91
     6   Forest NBS    1    36    60    74    80
     7  Pasture NBS    1    39    61    74    80
     8    Crops NBS    1    62    71    78    81

    :param fsoils: string path to soils.asc raster file
    :param fsoilsparam: string path to soils parameters .txt file. Separator = ;

     Value Group
     1     A
     2     B
     3     C
     4     D

    Sep=';' and must have a 'Value' field enconding the raster soil values
    :param folder: string path to destination folder
    :param filename: string name of file
    :return: string file path
    """
    #
    # import data
    metalulc, lulc = input.asc_raster(flulc)
    metasoils, soils = input.asc_raster(fsoils)
    lulc_param_df = pd.read_csv(flulcparam, sep=';\s+', engine='python')
    soils_param_df = pd.read_csv(fsoilsparam, sep=';\s+', engine='python')
    lulc_classes = lulc_param_df['Value'].values
    soils_classes = soils_param_df['Value'].values
    cn_a = lulc_param_df['CN-A'].values
    cn_b = lulc_param_df['CN-B'].values
    cn_c = lulc_param_df['CN-C'].values
    cn_d = lulc_param_df['CN-D'].values
    cn_values = (cn_a, cn_b, cn_c, cn_d)
    #
    #
    # process data
    cn_map = geo.cn(lulc=lulc, soils=soils, cnvalues=cn_values, lulcclasses=lulc_classes, soilclasses=soils_classes)
    #
    # export data
    export_file = output.asc_raster(cn_map, metalulc, folder, filename)
    return export_file

# deprecated:
def map_grad(fslope, folder='C:/bin', filename='grad'):
    """
    derive the topographical gradient tan(B) from the slope in degrees
    :param fslope: string path to slope in degrees raster .asc file
    :param folder: string path to destination folder
    :param filename: string of file name
    :return: string path to file
    """
    # import data
    meta, slope = input.asc_raster(fslope)
    #
    # process data
    grad = geo.grad(slope)
    #
    # export data
    export_file = output.asc_raster(grad, meta, folder, filename)
    return export_file


# deprecated:
def import_climpat(fclimmonth, rasterfolder='C:/bin', folder='C:/bin', filename='clim_month', alias='p'):
    """
    #### Possibly deprecated function !! ####
    :param fclimmonth: string filepath to climate raster monthly pattern series txt file
    :param rasterfolder: string filepath to raster folder
    :param folder: string filepath to folder
    :param filename: string of filename
    :param alias: string climate alies
    :return: string filepath
    """
    from shutil import copyfile

    # import data
    clim_df = pd.read_csv(fclimmonth, sep=';\s+', engine='python')
    # print(clim_df)
    months = clim_df['Month'].values
    files = clim_df['File'].values
    #
    # process data
    new_files = list()
    for i in range(len(months)):
        src = files[i]
        lcl_month = months[i]
        lcl_filenm = alias + 'pat_' + str(lcl_month) + '.asc'
        dst = rasterfolder + '/' + lcl_filenm
        copyfile(src=src, dst=dst)
        # print(lcl_expf)
        new_files.append(dst)
    #
    # export data
    exp_df = pd.DataFrame({'Month': months, 'File': new_files})
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


# deprecated
def run_topmodel_deprec(fseries, fparam, faoi, ftwi, fcn, folder='C:/bin',tui=False, mapback=False, mapvar='R-ET-S1-S2-Qv', qobs=False):
    """

    Run the PLANS3 TOPMODEL

    :param fseries: string file path to input series dataframe. File format: .txt
    Field separator = ';'
    Required fields: 'Date', 'Prec', 'Temp'
    Optional: 'Q'

    Date = daily data in YYYY-MM-DD
    Prec = precipitation in mm
    Temp = temperature in Celsius
    Q = observed flow in mm

    Example of dataframe formatting:

    Date;		Prec;	Temp;
    2005-01-01;	0.0;	25.96;
    2005-01-02;	0.0;	26.96;
    2005-01-03;	7.35;	25.2;
    2005-01-04;	3.95;	27.76;
    2005-01-05;	0.0;	27.24;
       ...		...	    ...
    2015-01-06;	0.0;	27.52;
    2015-01-07;	0.0;	28.84;
    2015-01-08;	12.5;	28.44;

    :param fparam: string file path to parameters dataframe.

    Required fields: 'Parameter' and 'Set' , where:
    Parameter =  string of parameter name
    Set = float of parameter value

    Order and names: m, ksat, qo, a, c, k, n

    Example of dataframe formatting:

    Parameter;  Set;
    m;          5.0;
    ksat;       2.0;
    qo;         1.0;
    a;          1.5;
    c;          100.0;
    lat;        -30.0;
    k;          1.1;
    n;          2.1;

    :param faoi: string file path to AOI (Area Of Interest) raster in .asc format.
    The AOI raster should be a pseudo-boolean (1 and 0) image of the watershed area

    :param ftwi: string file path to TWI raster in .asc format.

    :param fcn: string file path to CN raster in .asc format.

    Note: all rasters must have the same shape (rows x columns)

    :param folder: string file path to destination folder
    :param tui: boolean control to allow terminal messages. Default: False
    :param mapback: boolean control to map simulated variables. Default: False
    :param mapvar: string code to set mapped variables (see topmodel routine for code formatting)
    :return:
    when mapback=False:
    tuple of 3 string file paths:
    1) simulated parameters dataframe
    2) simulated histograms dataframes (count matrix used)
    3) simulated of global variables series dataframe
    4) visual pannel string file path
    when mapback=True:
    tuple of 4 elements:
    1) string file path of simulated parameters dataframe
    2) string file path of simulated histograms dataframes (count matrix used)
    3) string file path of simulated of global variables series dataframe
    5) visual pannel string file path
    4) tuple of string file paths of map lists files
    """
    #
    from hydrology import avg_2d, topmodel_hist, simulation, map_back
    from visuals import pannel_global
    import time
    #
    if tui:
        print('loading series...')
    lcl_df = pd.read_csv(fseries, sep=';', engine='python', parse_dates=['Date'])
    #lcl_df = dataframe_prepro(dataframe=lcl_df, strfields=('Date',))
    #lcl_df.query('Date > "2005-03-15" and Date <= "2005-07-15"', inplace=True)
    #lcl_df.query('Date > "2005-03-15" and Date <= "2010-03-15"', inplace=True)
    #
    # import aoi raster
    if tui:
        print('loading aoi...')
    meta, aoi = input.asc_raster(faoi)
    cell = meta['cellsize']
    #
    # import twi raster
    if tui:
        print('loading twi...')
    meta, twi = input.asc_raster(ftwi)
    #
    # import CN raster
    if tui:
        print('loading cn...')
    meta, cn = input.asc_raster(fcn)
    #
    # import parameters
    if tui:
        print('loading parameters...')
    df_param = pd.read_csv(fparam, sep=';', engine='python', index_col='Parameter')
    m = df_param.loc['m'].values[0]
    ksat = df_param.loc['ksat'].values[0]
    qo = df_param.loc['qo'].values[0]
    a = df_param.loc['a'].values[0]
    c = df_param.loc['c'].values[0]
    lat = df_param.loc['lat'].values[0]
    k = df_param.loc['k'].values[0]
    n = df_param.loc['n'].values[0]
    qt0 = 0.2  # mm/d
    lamb = avg_2d(var2d=twi, weight=aoi)
    #
    # compute histograms
    if tui:
        print('computing histograms...', end='\t\t')
    init = time.time()
    countmatrix, twihist, cnhist = topmodel_hist(twi=twi, cn=cn, aoi=aoi)
    end = time.time()
    if tui:
        print('Enlapsed time: {:.3f} seconds'.format(end - init))
    #
    #
    # run topmodel simulation
    if tui:
        print('running simulation...', end='\t\t')
    init = time.time()
    # mapback conditionals:
    if mapback:
        sim_df, mapped = simulation(lcl_df, twihist, cnhist, countmatrix, lamb=lamb, ksat=ksat, m=m, qo=qo, a=a, c=c,
                                    lat=lat, qt0=qt0, k=k, n=n, tui=False, mapback=mapback, mapvar=mapvar, qobs=qobs)
    else:
        sim_df = simulation(lcl_df, twihist, cnhist, countmatrix, lamb=lamb, ksat=ksat, m=m, qo=qo, a=a, c=c,
                            lat=lat, qt0=qt0, k=k, n=n, tui=False, mapback=mapback, mapvar=mapvar, qobs=qobs)
    end = time.time()
    if tui:
        print('Enlapsed time: {:.3f} seconds'.format(end - init))
    #
    #
    #
    # export files
    if tui:
        print('exporting run parameters...')
    exp_df = pd.DataFrame({'Parameter':('m', 'ksat', 'qo', 'a', 'c', 'lat', 'k', 'n'),
                           'Set':(m, ksat, qo, a, c, lat, k, n)})
    exp_file1 = folder + '/' + 'parameters.txt'
    exp_df.to_csv(exp_file1, sep=';', index=False)
    #
    # export histograms
    if tui:
        print('exporting histograms...')
    exp_df = pd.DataFrame(countmatrix, index=twihist[0], columns=cnhist[0])
    exp_file2 = folder + '/' + 'histograms.txt'
    exp_df.to_csv(exp_file2, sep=';', index_label='TWI\CN')
    #
    # export simulation
    if tui:
        print('exporting simulation results...')
    exp_file3 = folder + '/' + 'sim_series.txt'
    sim_df.to_csv(exp_file3, sep=';', index=False)
    #
    # export visual pannel
    if tui:
        print('exporting visual results...')
    sim_df['Qobs'] = lcl_df['Q']
    exp_file4 = pannel_global(sim_df, grid=False, show=False, qobs=True, folder=folder)
    #
    if mapback:
        if tui:
            print('exporting variable maps...', end='\t\t')
        init = time.time()
        #
        from os import mkdir
        mapvar_lst = mapvar.split('-')  # load string variables alias to list
        mapfiles_lst = list()
        stamp = pd.to_datetime(sim_df['Date'], format='%y-%m-%d')
        for var in mapvar_lst:  # loop across all variables
            lcl_folder = folder + '/' + var
            mkdir(lcl_folder)  # make diretory
            lcl_files = list()
            for t in range(len(stamp)):  # loop across all timesteps
                lcl_filename = var + '_' + str(stamp[t]).split(sep=' ')[0] + '.txt'
                lcl_file = lcl_folder + '/' + lcl_filename
                lcl_files.append(lcl_file)
                # export local dataframe to text file in local folder
                lcl_exp_df = pd.DataFrame(mapped[var][t], index=twihist[0], columns=cnhist[0])
                lcl_exp_df.to_csv(lcl_file, sep=';', index_label='TWI\CN')
                # map = map_back(zmatrix=mapped[var][t], a1=twi, a2=cn, bins1=twihist[0], bins2=cnhist[0])
                # plt.imshow(map[550:1020, 600:950], cmap='jet_r')
                # plt.show()
            # export map list file to main folder:
            lcl_exp_df = pd.DataFrame({'Date': sim_df['Date'], 'File': lcl_files})
            lcl_file = folder + '/' + var + '_maps' + '.txt'
            lcl_exp_df.to_csv(lcl_file, sep=';', index=False)
            mapfiles_lst.append(lcl_file)
        #
        mapfiles_lst = tuple(mapfiles_lst)
        end = time.time()
        if tui:
            print('Enlapsed time: {:.3f} seconds'.format(end - init))
    #
    if mapback:
        return (exp_file1, exp_file2, exp_file3, exp_file4, mapfiles_lst)
    else:
        return (exp_file1, exp_file2, exp_file3, exp_file4)


# deprecated:
def cn_series(flulcseries, flulcparam, fsoils, fsoilsparam, rasterfolder='C:/bin', folder='C:/bin',
              filename='cn_series'):
    """
    derive the CN series raster and txt file
    :param flulcseries: string filepath to lulc series txt
    :param flulcparam: string filepath to lulc param txt
    :param fsoils: string filepath to soils .asc rasterfiles
    :param fsoilsparam: string filepath to soils parameters
    :param rasterfolder: string filepath to raster folder
    :param folder: string filepath to return file
    :param filename: string of filename
    :return: string filepath for txt file
    """
    # import data
    lulc_series_df = pd.read_csv(flulcseries, sep=';', engine='python')
    # print(lulc_series_df)
    dates = lulc_series_df['Date'].values
    files = lulc_series_df['File'].values
    metasoils, soils = input.asc_raster(fsoils)
    lulc_param_df = pd.read_csv(flulcparam, sep=';\s+', engine='python')
    soils_param_df = pd.read_csv(fsoilsparam, sep=';\s+', engine='python')
    lulc_classes = lulc_param_df['Value'].values
    soils_classes = soils_param_df['Value'].values
    cn_a = lulc_param_df['CN-A'].values
    cn_b = lulc_param_df['CN-B'].values
    cn_c = lulc_param_df['CN-C'].values
    cn_d = lulc_param_df['CN-D'].values
    cn_values = (cn_a, cn_b, cn_c, cn_d)
    #
    # process data
    new_files = list()
    for i in range(len(dates)):
        metalulc, lulc = input.asc_raster(files[i])
        lcl_date = dates[i]
        # process data
        cn_map = geo.cn(lulc=lulc, soils=soils, cnvalues=cn_values, lulcclasses=lulc_classes, soilclasses=soils_classes)
        lcl_filenm = 'cn_' + str(lcl_date)
        dst = output.asc_raster(cn_map, metalulc, rasterfolder, lcl_filenm)
        # print(lcl_expf)
        new_files.append(dst)
    #
    # export data
    exp_df = pd.DataFrame({'Date': dates, 'File': new_files})
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


# deprecated:
def map_cn_avg(fcnseries, fseries, folder='C:/bin', filename='cn_calib'):
    """
    Derive the average CN given a time series.
    :param fcnseries: string file path to CN series dataframe. Required fields: 'Date' and 'File'
    :param fseries: string file path to series dataframe. Required field: 'Date'
    :param folder: string file path to destination directory
    :param filename: string file name (without extension)
    :return: string file path to derived file
    """
    cnseries_df = pd.read_csv(fcnseries, sep=';\s+', engine='python', parse_dates=['Date'])
    series_df = pd.read_csv(fseries, sep=';\s+', engine='python', parse_dates=['Date'])
    datemin = series_df['Date'].min()
    datemax = series_df['Date'].max()
    expression = 'Date >= "{}" and Date <= "{}"'.format(datemin, datemax)
    cnseries_df.query(expr=expression, inplace=True)
    files = cnseries_df['File'].values
    size = len(files)
    for i in range(size):
        if i == 0:
            meta, lcl_cn1 = input.asc_raster(files[i])
        else:
            meta, lcl_cn2 = input.asc_raster(files[i])
            lcl_cn1 = lcl_cn1 + lcl_cn2
    cn_avg = lcl_cn1 / size
    #
    exp_file = output.asc_raster(array=cn_avg, meta=meta, folder=folder, filename=filename)
    return exp_file


def pannel_1image_3series(image, imax, t, x1, x2, x3, x1max, x2max, x3max, titles, vline=20,
                          cmap='jet', folder='C:/bin', filename='pannel_topmodel', suff='', show=False):
    """
    Plot a pannel with 1 image (left) and  3 signals (right).
    :param image: 2d numpy array of image
    :param imax: float max value of image
    :param t: 1d array of x-axis shared variable
    :param x1: 1d array of first signal
    :param x2: 1d array of third signal
    :param x3: 1d array of second signal
    :param x1max: float max value of x1
    :param x2max: float max value of x2
    :param x3max: float max value of x3
    :param vline: int of index position of vertical line
    :param folder: string folder path to export image
    :param filename: string of file name
    :param suff: string of suffix
    :return: string file path of plot
    """
    #
    fig = plt.figure(figsize=(16, 8))  # Width, Height
    gs = mpl.gridspec.GridSpec(3, 6, wspace=0.8, hspace=0.6)
    #
    # plot image
    plt.subplot(gs[0:, 0:2])
    im = plt.imshow(image, cmap=cmap, vmin=0, vmax=imax)
    plt.axis('off')
    plt.title(titles[0])
    plt.colorbar(im, shrink=0.4)
    #
    # plot x1
    y = x1
    ymax = x1max
    plt.subplot(gs[0, 2:])
    var = y[vline]
    plt.title('{}: {:.1f}'.format(titles[1], var), loc='left')
    plt.ylabel('mm')
    plt.plot(t, y)
    plt.ylim(0, 1.1 * ymax)
    plt.vlines(t[vline], ymin=0, ymax=1.2 * ymax, colors='r')
    plt.plot(t[vline], y[vline], 'ro')
    #
    # plot x2
    y = x2
    ymax = x2max
    plt.subplot(gs[1, 2:])
    var = y[vline]
    plt.title('{}: {:.2f}'.format(titles[2], var), loc='left')
    plt.ylabel('mm')
    plt.plot(t, y, 'navy')
    plt.ylim(0, 1.1 * ymax)
    plt.vlines(t[vline], ymin=0, ymax=1.2 * ymax, colors='r')
    plt.plot(t[vline], y[vline], 'ro')
    #
    # plot x3
    y = x3
    ymax = x3max
    plt.subplot(gs[2, 2:])
    var = y[vline]
    plt.title('{}: {:.1f}'.format(titles[3], var), loc='left')
    plt.ylabel('mm')
    plt.plot(t, y, 'tab:orange')
    plt.ylim(0, 1.1 * ymax)
    plt.vlines(t[vline], ymin=0, ymax=1.2 * ymax, colors='r')
    plt.plot(t[vline], y[vline], 'ro')
    #
    # Fix date formatting
    # https://matplotlib.org/stable/gallery/recipes/common_date_problems.html#sphx-glr-gallery-recipes-common-date-problems-py
    fig.autofmt_xdate()
    #
    if show:
        plt.show()
        plt.close(fig)
    else:
        # export file
        filepath = folder + '/' + filename + '_' + suff + '.png'
        plt.savefig(filepath)
        plt.close(fig)
        return filepath


def pannel_4image_4series(im4, imax, t, y4, y4max, y4min, cmaps, imtitles, ytitles, ylabels, vline=20,
                          folder='C:/bin', filename='pannel_topmodel', suff='', show=False):
    """
    Plot a pannel with 4 images (left) and  4 signals (right).
    :param im4: tuple of 4 2d arrays (images)
    :param imax: float of images vmax
    :param t: 1d array of x-axis shared variable
    :param y4: tuple of 4 signal series arrays
    :param y4max: tuple of 4 vmax of series arrays
    :param y4min: tuple of 4 vmin of series arrays
    :param cmaps: tuple of 4 string codes to color maps
    :param imtitles: tuple of 4 string titles for images
    :param ytitles: tuple of 4 string titles of series
    :param ylabels: tuple of 4 string y axis labels
    :param vline: int of index position of vertical line
    :param folder: string folder path to export image
    :param show: boolean to show image instead of exporting
    :param filename: string of file name (without extension)
    :param suff: string of suffix
    :return: string file path of plot
    """
    #
    fig = plt.figure(figsize=(16, 9))  # Width, Height
    gs = mpl.gridspec.GridSpec(4, 10, wspace=0.8, hspace=0.6)
    #
    # images
    #
    # Left Upper
    ax = fig.add_subplot(gs[0:2, 0:2])
    im = plt.imshow(im4[0], cmap=cmaps[0], vmin=0, vmax=imax)
    plt.axis('off')
    plt.title(imtitles[0])
    plt.colorbar(im, shrink=0.4)
    #
    # Left bottom
    ax = fig.add_subplot(gs[2:4, 0:2])
    im = plt.imshow(im4[1], cmap=cmaps[1], vmin=0, vmax=0.5 * imax)
    plt.axis('off')
    plt.title(imtitles[1])
    plt.colorbar(im, shrink=0.4)
    #
    # Right Upper
    ax = fig.add_subplot(gs[0:2, 2:4])
    im = plt.imshow(im4[2], cmap=cmaps[2], vmin=0, vmax=0.5 * imax)
    plt.axis('off')
    plt.title(imtitles[2])
    plt.colorbar(im, shrink=0.4)
    #
    # Right Bottom
    ax = fig.add_subplot(gs[2:4, 2:4])
    im = plt.imshow(im4[3], cmap=cmaps[3], vmin=0, vmax=imax)
    plt.axis('off')
    plt.title(imtitles[3])
    plt.colorbar(im, shrink=0.4)
    #
    # series
    #
    # set x ticks
    size = len(t)
    spaces = int(size / 5)
    locs = np.arange(0, size, spaces)
    labels = t[locs]
    #
    ax = fig.add_subplot(gs[0, 5:])
    lcl = 0
    y = y4[lcl]
    plt.plot(t, y)
    plt.vlines(t[vline], ymin=y4min[lcl], ymax=y4max[lcl], colors='r')
    plt.plot(t[vline], y[vline], 'ro')
    var = y[vline]
    plt.title('{}: {:.2f}'.format(ytitles[lcl], var), loc='left')
    plt.ylabel(ylabels[lcl])
    #
    ax2 = fig.add_subplot(gs[1, 5:])#, sharex=ax)
    lcl = lcl + 1
    y = y4[lcl]
    plt.plot(t, y)
    plt.vlines(t[vline], ymin=y4min[lcl], ymax=y4max[lcl], colors='r')
    plt.plot(t[vline], y[vline], 'ro')
    var = y[vline]
    plt.title('{}: {:.2f}'.format(ytitles[lcl], var), loc='left')
    plt.ylabel(ylabels[lcl])
    #
    ax2 = fig.add_subplot(gs[2, 5:])#, sharex=ax)
    lcl = lcl + 1
    y = y4[lcl]
    plt.plot(t, y)
    plt.vlines(t[vline], ymin=y4min[lcl], ymax=y4max[lcl], colors='r')
    plt.plot(t[vline], y[vline], 'ro')
    var = y[vline]
    plt.title('{}: {:.2f}'.format(ytitles[lcl], var), loc='left')
    plt.ylabel(ylabels[lcl])
    #
    ax2 = fig.add_subplot(gs[3, 5:])#, sharex=ax)
    lcl = lcl + 1
    y = y4[lcl]
    plt.plot(t, y, 'k')
    plt.vlines(t[vline], ymin=y4min[lcl], ymax=y4max[lcl], colors='r')
    plt.plot(t[vline], y[vline], 'ro')
    var = y[vline]
    plt.title('{}: {:.2f}'.format(ytitles[lcl], var), loc='left')
    plt.ylabel(ylabels[lcl])
    #
    # Fix date formatting
    # https://matplotlib.org/stable/gallery/recipes/common_date_problems.html#sphx-glr-gallery-recipes-common-date-problems-py
    fig.autofmt_xdate()
    #
    if show:
        plt.show()
        plt.close(fig)
    else:
        # export file
        filepath = folder + '/' + filename + '_' + suff + '.png'
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

