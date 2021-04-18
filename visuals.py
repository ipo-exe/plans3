import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def pannel_obs_sim_analyst(series, freq, params, fld_obs='Obs', fld_sim='Sim', fld_date='Date', filename='analyst', suff='',
                   folder='C:/bin', show=False):
    """
    Pannel of Obs vs. Sim Analyst
    :param series: Series Analyst dataframe (from obs_sim_analyst() function in tools.py)
    :param freq: Frequency Analyst dataframe (from obs_sim_analyst() function in tools.py)
    :param params: Parameters Analyst dataframe
    :param fld_obs: string of field of observed series data
    :param fld_sim: string of field of simulated series data
    :param fld_date: string of date field
    :param filename: string of file name
    :param suff: optional string for suffix
    :param folder: string of export directory
    :param show: boolean to control showing figure
    :return: string of file
    """
    #
    fig = plt.figure(figsize=(18, 9))  # Width, Height
    gs = mpl.gridspec.GridSpec(5, 13, wspace=0.9, hspace=0.9)
    #
    # min max setup
    vmax = np.max((np.max(series[fld_obs]), np.max(series[fld_sim])))
    vmin = np.min((np.min(series[fld_obs]), np.min(series[fld_sim])))
    #
    # plot of CFCs
    plt.subplot(gs[0:2, 0:2])
    plt.title('CFCs', loc='left')
    plt.plot(freq['Exeedance'], freq['ValuesObs'], 'tab:grey', label='Obs')
    plt.plot(freq['Exeedance'], freq['ValuesSim'], 'tab:blue', label='Sim')
    plt.ylim((vmin, 1.2 * vmax))
    plt.yscale('log')
    plt.ylabel('mm')
    plt.xlabel('Exeed. %')
    plt.grid(True)
    plt.legend(loc='upper right')
    #
    # plot of series
    plt.subplot(gs[0:2, 3:10])
    plt.title('Series', loc='left')
    plt.plot(series[fld_date], series[fld_obs], 'tab:grey', linewidth=2, label='Observed')
    plt.plot(series[fld_date], series[fld_sim], 'tab:blue', label='Simulated')
    plt.ylim((vmin, 1.2 * vmax))
    plt.yscale('log')
    plt.ylabel('mm')
    plt.grid(True)
    plt.legend(loc='upper right', ncol=2)
    #
    # plot of Scatter
    plt.subplot(gs[0:2, 11:])
    plt.title('Obs vs. Sim  (R={:.2f})'.format(float(params[params['Parameter'] == 'R']['Value'])), loc='left')
    plt.scatter(series[fld_obs], series[fld_sim], c='tab:grey', s=15, alpha=0.3, edgecolors='none')
    plt.xlabel('Q-obs  (mm)')
    plt.ylabel('Q-sim  (mm)')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([0, vmax], [0, vmax], 'tab:grey', linestyle='--', label='1:1')
    plt.ylim((vmin, 1.2 * vmax))
    plt.xlim((vmin, 1.2 * vmax))
    plt.grid(True)
    plt.legend(loc='upper left')
    #
    # plot of CFC Erros
    plt.subplot(gs[2, 0:2])
    plt.title('CFC Error', loc='left')
    plt.plot(freq['Exeedance'], freq['E'], 'tab:red')
    plt.ylabel('mm')
    plt.xlabel('Exeed. %')
    plt.grid(True)
    #
    # plot Error
    plt.subplot(gs[2, 3:10])
    plt.title('Series - Error', loc='left')
    plt.plot(series[fld_date], series['E'], 'tab:red')
    plt.ylabel('mm')
    plt.grid(True)
    #
    # plot
    plt.subplot(gs[3, 0:2])
    plt.title('CFC - Squared Error', loc='left')
    plt.plot(freq['Exeedance'], freq['SE'], 'tab:red')
    plt.ylabel('mm ^ 2')
    plt.xlabel('Exeed. %')
    plt.grid(True)
    #
    # plot
    plt.subplot(gs[3, 3:10])
    plt.title('Series - Sq. Error', loc='left')
    plt.plot(series[fld_date], series['SE'], 'tab:red')
    plt.ylabel('mm ^ 2')
    plt.grid(True)
    #
    plt.subplot(gs[3, 11:])
    plt.title('Analyst parameters', loc='left')
    plt.text(x=0, y=0.8,  s='Pbias : {:.2f}%'.format(float(params[params['Parameter'] == 'PBias']['Value'])))
    plt.text(x=0, y=0.6,  s='R : {:.2f}'.format(float(params[params['Parameter'] == 'R']['Value'])))
    plt.text(x=0, y=0.4,  s='RMSE : {:.2f} mm'.format(float(params[params['Parameter'] == 'RMSE']['Value'])))
    plt.text(x=0, y=0.2,  s='NSE : {:.2f}'.format(float(params[params['Parameter'] == 'NSE']['Value'])))
    plt.text(x=0, y=0.0,  s='KGE : {:.2f}'.format(float(params[params['Parameter'] == 'KGE']['Value'])))
    plt.text(x=0, y=-0.2, s='RMSElog : {:.2f}'.format(float(params[params['Parameter'] == 'RMSElog']['Value'])))
    plt.text(x=0, y=-0.4, s='NSElog : {:.2f}'.format(float(params[params['Parameter'] == 'NSElog']['Value'])))
    plt.text(x=0, y=-0.8, s='CFC-R : {:.2f}'.format(float(params[params['Parameter'] == 'R-CFC']['Value'])))
    plt.text(x=0, y=-1.0, s='CFC-RMSE : {:.2f}'.format(float(params[params['Parameter'] == 'RMSE-CFC']['Value'])))
    plt.text(x=0, y=-1.2, s='CFC-RMSElog : {:.2f}'.format(float(params[params['Parameter'] == 'RMSElog-CFC']['Value'])))
    plt.axis('off')
    #
    # plot
    plt.subplot(gs[4, 0:2])
    plt.title('CFC - Sq. Error of Log', loc='left')
    plt.plot(freq['Exeedance'], freq['SElog'], 'tab:red')
    plt.xlabel('Exeed. %')
    plt.grid(True)
    # plot
    plt.subplot(gs[4, 3:10])
    plt.title('Series - Sq. Error of Log', loc='left')
    plt.plot(series[fld_date], series['SElog'], 'tab:red')
    plt.grid(True)
    #
    if show:
        plt.show()
        plt.close(fig)
    else:
        # export file
        if suff != '':
            filepath = folder + '/' + filename + '_' + suff + '.png'
        else:
            filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath)
        plt.close(fig)
        return filepath


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


def pannel_prec_q(t, prec, q, grid=True, folder='C:/bin', filename='pannel_prec_q', suff='', show=False):
    #
    fig = plt.figure(figsize=(16, 8))  # Width, Height
    gs = mpl.gridspec.GridSpec(2, 1, wspace=0.8, hspace=0.6)
    # plot prec
    y = prec
    ymax = np.max(y)
    ax1 = fig.add_subplot(gs[0, 0])
    plt.title('Precipitation', loc='left')
    plt.ylabel('mm')
    plt.plot(t, y)
    plt.ylim(0, 1.1 * ymax)
    plt.grid(grid)
    #plt.xticks(locs, labels)
    # plot q
    y = q
    ymax = np.max(y)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    plt.title('Discharge', loc='left')
    plt.ylabel('mm')
    plt.plot(t, y)
    plt.ylim(0, 1.1 * ymax)
    plt.grid(grid)
    #plt.xticks(locs, labels)
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


def pannel_prec_q_logq(t, prec, q, grid=True, folder='C:/bin', filename='pannel_prec_q_logq', suff='', show=False):
    #
    fig = plt.figure(figsize=(16, 8))  # Width, Height
    gs = mpl.gridspec.GridSpec(3, 1, wspace=0.8, hspace=0.6)
    # plot prec
    y = prec
    ymax = np.max(y)
    ax1 = fig.add_subplot(gs[0, 0])
    plt.title('Precipitation', loc='left')
    plt.ylabel('mm')
    plt.plot(t, y)
    plt.ylim(0, 1.1 * ymax)
    plt.grid(grid)
    # plot q
    y = q
    ymax = np.max(y)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    plt.title('Discharge', loc='left')
    plt.ylabel('mm')
    plt.plot(t, y)
    plt.ylim(0, 1.1 * ymax)
    plt.grid(grid)
    # plot log q
    y = q
    ymax = np.max(y)
    ymin = np.min(y)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    plt.title('Discharge', loc='left')
    plt.ylabel('mm')
    plt.plot(t, y)
    plt.ylim(0.9 * ymin, 1.1 * ymax)
    plt.yscale('log')
    plt.grid(grid)
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


def pannel_sim_prec_q_logq(t, prec, qobs, qsim, grid=True,
                           folder='C:/bin', filename='pannel_sim_prec_q_logq', suff='', show=False):
    #
    fig = plt.figure(figsize=(16, 8))  # Width, Height
    gs = mpl.gridspec.GridSpec(3, 1, wspace=0.8, hspace=0.6)
    # plot prec
    y = prec
    ymax = np.max(y)
    ax1 = fig.add_subplot(gs[0, 0])
    plt.title('Precipitation', loc='left')
    plt.ylabel('mm')
    plt.plot(t, y)
    plt.ylim(0, 1.1 * ymax)
    plt.grid(grid)
    # plot q
    y1 = qobs
    y2 = qsim
    ymax = np.max((y1, y2))
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    plt.title('Discharge', loc='left')
    plt.ylabel('mm')
    plt.plot(t, y1)
    plt.plot(t, y2)
    plt.ylim(0, 1.1 * ymax)
    plt.grid(grid)
    # plot log q
    ymin = np.min((y1, y2))
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    plt.title('Discharge', loc='left')
    plt.ylabel('mm')
    plt.plot(t, y1)
    plt.plot(t, y2)
    plt.ylim(0.9 * ymin, 1.1 * ymax)
    plt.yscale('log')
    plt.grid(grid)
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


def pannel_topmodel(dataframe, qobs=False, grid=True, show=False, folder='C:/bin', filename='pannel_topmodel', suff=''):
    """
    visualize the topmodel global variables in a single pannel
    :param dataframe: pandas dataframe from hydrology.topmodel_sim()
    :param grid: boolean for grid
    :param folder: string to destination directory
    :param filename: string file name
    :param suff: string suffix in file name
    :param show: boolean control to show figure instead of saving it
    :return: string file path
    """
    #
    fig = plt.figure(figsize=(16, 25))  # Width, Height
    gs = mpl.gridspec.GridSpec(10, 8, wspace=0.8, hspace=0.6)  # nrows, ncols
    # 1
    ind = 0
    ax = fig.add_subplot(gs[ind, 0:])
    plt.plot(dataframe['Date'], dataframe['Prec'])
    plt.ylabel('Prec mm')
    ax2 = ax.twinx()
    plt.plot(dataframe['Date'], dataframe['Temp'], 'tab:orange')
    plt.ylabel('Temp °C')
    plt.grid(grid)
    # 2
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['PET'], 'k', label='PET')
    plt.plot(dataframe['Date'], dataframe['ET'], 'tab:red', label='ET')
    plt.plot(dataframe['Date'], dataframe['Tpun'] + dataframe['Tpgw'] , 'r', label='Tp')
    plt.ylabel('PET, ET\nmm')
    plt.grid(grid)
    plt.legend(loc='upper right', ncol=3)
    # 3
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['PET'], 'k', label='PET')
    plt.plot(dataframe['Date'], dataframe['Evc'], 'tab:red', label='Evc')
    plt.plot(dataframe['Date'], dataframe['Evs'], 'r', label='Evs')
    plt.ylabel('PET & Ev\nmm')
    plt.grid(grid)
    plt.legend(loc='upper right', ncol=3)
    # 4
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['PET'], 'k', label='PET')
    plt.plot(dataframe['Date'], dataframe['Tpun'], 'tab:red', label='Tpun')
    plt.plot(dataframe['Date'], dataframe['Tpgw'], 'r', label='Tpgw')
    plt.ylabel('PET & Tp\nmm')
    plt.grid(grid)
    plt.legend(loc='upper right', ncol=3)
    # 5
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    #plt.plot(dataframe['Date'], dataframe['Cpy'])
    plt.plot(dataframe['Date'], dataframe['TF'])
    plt.ylabel('Cpy&TF\nmm')
    plt.grid(grid)
    # 6
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['R'], 'tab:orange')
    plt.plot(dataframe['Date'], dataframe['Inf'], 'k')
    plt.ylabel('R & Inf\nmm')
    plt.grid(grid)
    # 7
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['Sfs'])
    plt.ylabel('Sfs\nmm')
    plt.grid(grid)
    # 8
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['D'], 'k')
    plt.plot(dataframe['Date'], dataframe['Unz'])
    plt.ylabel('Unz & D\nmm')
    plt.grid(grid)
    # 9
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['Qv'])
    plt.ylabel('Qv\nmm')
    plt.grid(grid)
    # 10
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    if qobs:
        plt.plot(dataframe['Date'], dataframe['Qobs'], 'tab:orange')
    plt.plot(dataframe['Date'], dataframe['Q'])
    plt.plot(dataframe['Date'], dataframe['Qb'], 'navy')
    if qobs:
        plt.ylim(0.9 * np.min((dataframe['Q'].values, dataframe['Qobs'].values)),
                 1.1 * np.max((dataframe['Q'].values, dataframe['Qobs'].values)))
    plt.ylabel('Q mm')
    plt.yscale('log')
    plt.grid(grid)
    #
    if show:
        plt.show()
        plt.close(fig)
    else:
        # export file
        if suff != '':
            filepath = folder + '/' + filename + '_' + suff + '.png'
        else:
            filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath)
        plt.close(fig)
        return filepath

# deprecated
def pannel_topmodel_maps(t, prec, precmax, qb, qbmax, pet, et, etmax, maps, mapsmax, vline=20,
                         folder='C:/bin', filename='pannel_topmodel', suff='', show=False):
    """
    Plot a pannel of all 13 processes maps of Topmodel and precipitation, baseflow and ET series.
    :param t: Series or Array of dates
    :param prec: array of precipiation data
    :param precmax: float of max value of precipitation
    :param qb: array of baseflow data
    :param qbmax: float of max value of baseflow
    :param pet: array of PET data
    :param et: array of ET data
    :param etmax: float of max value of ET
    :param maps: iterable object (tuple, list, array) of 2d-arrays of 13 processes maps in the following order:
    Prec, VSA, D, S1, S2, TF, Inf, R, Qv, ET, Ev, Tp, Tpgw .
    :param mapsmax: float of max value of map processes
    :param vline: int of index position of vertical line
    :param folder: string path to destination folder
    :param filename: string of file name (without extension)
    :param suff: string suffix to file name
    :param show: boolean control to show or save frame - Default: False
    :return: string file path of plot (only when show=False)
    """
    #
    fig = plt.figure(figsize=(15, 7))  # Width, Height
    nrows = 3
    ncols = 8
    gs = mpl.gridspec.GridSpec(nrows, ncols, wspace=0.8, hspace=0.6)
    #
    # titles setup
    titles = ('Precip. (mm).', 'VSA', 'Deficit', 'Canopy water ', 'Soil water',
              'Throughfall', 'Infiltration', 'Runoff', 'Recharge',
              'ET', 'Evap. (Canopy)', 'Transp. (Soil)', 'Transp. (GW)')
    #
    # plotting maps loop
    count = 0
    for i in range(0, nrows):
        for j in range(0, 5):
            if i > 0 and j == 0:
                pass
            else:
                ax = fig.add_subplot(gs[i, j])
                if i == 0 and j == 1:
                    lcl_max = 1
                    lcl_cmap = 'Blues'  # color map
                elif i == 0 and j == 2:
                    lcl_max = mapsmax
                    lcl_cmap = 'jet'  # color map
                else:
                    lcl_max = mapsmax
                    lcl_cmap = 'viridis_r'  # color map
                im = plt.imshow(maps[count], cmap=lcl_cmap, vmin=0, vmax=lcl_max)
                plt.axis('off')
                plt.title(titles[count])
                if i == 0 and j == 1:
                    pass
                else:
                    plt.colorbar(im, shrink=0.5)
                count = count + 1
    #
    ax = fig.add_subplot(gs[0, 5:])
    plt.plot(t, prec)
    plt.vlines(t[vline], ymin=0, ymax=precmax, colors='r')
    plt.plot(t[vline], prec[vline], 'ro')
    plt.title('Precipitation: {:.2f} mm'.format(prec[vline]), loc='left')
    plt.ylabel('mm')
    #
    ax = fig.add_subplot(gs[1, 5:])
    plt.plot(t, qb, 'navy')
    plt.vlines(t[vline], ymin=0, ymax=qbmax, colors='r')
    plt.plot(t[vline], qb[vline], 'ro')
    plt.title('Baseflow: {:.2f} mm'.format(qb[vline]), loc='left')
    plt.ylabel('mm')
    #
    ax = fig.add_subplot(gs[2, 5:])
    plt.plot(t, et, 'tab:red', label='ET')
    plt.plot(t, pet , 'grey', label='PET')
    plt.vlines(t[vline], ymin=0, ymax=etmax, colors='r')
    plt.plot(t[vline], et[vline], 'ro')
    plt.title('Actual ET: {:.2f} mm'.format(et[vline]), loc='left')
    plt.ylabel('mm')
    plt.legend( ncol=2, loc='upper right')
    #
    # Fix date formatring
    # https://matplotlib.org/stable/gallery/recipes/common_date_problems.html#sphx-glr-gallery-recipes-common-date-problems-py
    fig.autofmt_xdate()
    #
    if show:
        plt.show()
        plt.close(fig)
    else:
        # export file
        if suff != '':
            filepath = folder + '/' + filename + '_' + suff + '.png'
        else:
            filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath)
        plt.close(fig)
        return filepath


def plot_zmap3d(zmap, x, y):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(0, len(x), 1)
    Y = np.arange(0, len(y), 1)
    X, Y = np.meshgrid(Y, X)
    Z = np.sin(R)
    ax.plot_surface(X, Y, np.log10(count + 1), cmap='viridis', edgecolor='none')
    ax.set_title('Surface plot')
    plt.show()


def plot_calib_series(dataframe, grid=True, filename='calib_series', folder='C:/bin', show=True):
    # todo docstring
    fig = plt.figure(figsize=(16, 8))  # Width, Height
    gs = mpl.gridspec.GridSpec(3, 8, wspace=0.8, hspace=0.2)  # nrows, ncols
    # 1
    ind = 0
    ax = fig.add_subplot(gs[ind, 0:])
    plt.plot(dataframe['Date'], dataframe['Prec'])
    plt.ylabel('Prec mm')
    ax2 = ax.twinx()
    plt.plot(dataframe['Date'], dataframe['Temp'], 'tab:orange')
    plt.ylabel('Temp °C')
    plt.grid(grid)
    # 2
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['Q'])
    plt.ylabel('Q mm')
    plt.yscale('log')
    plt.grid(grid)
    #
    # 3
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['IRI'], label='IRI')
    plt.plot(dataframe['Date'], dataframe['IRA'], label='IRA')
    plt.legend(loc='upper right', ncol=2)
    plt.ylabel('Irrigation mm')
    plt.grid(grid)
    #
    if show:
        plt.show()
        plt.close(fig)
    else:
        # export file
        if suff != '':
            filepath = folder + '/' + filename + '_' + suff + '.png'
        else:
            filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath)
        plt.close(fig)
        return filepath


def plot_lulc_view(lulc, lulc_df, basin, meta, mapttl='lulc', filename='mapview', folder='C:/bin', metadata=False, show=False):
    from geo import reclassify, mask
    from matplotlib.colors import ListedColormap
    colors = lulc_df['ColorLULC'].values
    names = lulc_df['LULCName'].values
    ids = lulc_df['IdLULC'].values
    ranges = [np.min(ids), np.max(ids)]
    cmap = ListedColormap(colors)
    #
    fig = plt.figure(figsize=(14, 9))  # Width, Height
    gs = mpl.gridspec.GridSpec(6, 10, wspace=0.5, hspace=0.5)
    #
    ax = fig.add_subplot(gs[:4, :4])
    fmap = mask(lulc, basin)
    im = plt.imshow(fmap, cmap=cmap, vmin=ranges[0], vmax=ranges[1])
    plt.title(mapttl)
    plt.axis('off')
    #
    #
    # canopy
    fmap = reclassify(lulc, upvalues=ids, classes=lulc_df['f_Canopy'].values)
    fmap = mask(fmap, basin)
    ax = fig.add_subplot(gs[:2, 4:6])
    im = plt.imshow(fmap, cmap='viridis_r')
    plt.title('Canopy factor')
    plt.colorbar(im, shrink=0.4)
    plt.axis('off')
    #
    # Surface
    fmap = reclassify(lulc, upvalues=ids, classes=lulc_df['f_Surface'].values)
    fmap = mask(fmap, basin)
    ax = fig.add_subplot(gs[:2, 6:8])
    im = plt.imshow(fmap, cmap='viridis_r', vmin=0)
    plt.title('Surface factor')
    plt.colorbar(im, shrink=0.4)
    plt.axis('off')
    #
    # RD
    fmap = reclassify(lulc, upvalues=ids, classes=lulc_df['f_RootDepth'].values)
    fmap = mask(fmap, basin)
    ax = fig.add_subplot(gs[:2, 8:10])
    im = plt.imshow(fmap, cmap='viridis_r', vmin=0)
    plt.title('RootDepth factor')
    plt.colorbar(im, shrink=0.4)
    plt.axis('off')
    #
    # RD
    fmap = reclassify(lulc, upvalues=ids, classes=lulc_df['f_IRA'].values)
    fmap = mask(fmap, basin)
    ax = fig.add_subplot(gs[2:4, 6:8])
    im = plt.imshow(fmap, cmap='YlGnBu', vmin=0)
    plt.title('IRA factor')
    plt.colorbar(im, shrink=0.4)
    plt.axis('off')
    #
    # RD
    fmap = reclassify(lulc, upvalues=ids, classes=lulc_df['f_IRI'].values)
    fmap = mask(fmap, basin)
    ax = fig.add_subplot(gs[2:4, 8:10])
    im = plt.imshow(fmap, cmap='YlGnBu', vmin=0)
    plt.title('IRI factor')
    plt.colorbar(im, shrink=0.4)
    plt.axis('off')
    #
    # RD
    fmap = reclassify(lulc, upvalues=ids, classes=lulc_df['C_USLE'].values)
    fmap = mask(fmap, basin)
    ax = fig.add_subplot(gs[4:, 6:8])
    im = plt.imshow(fmap, cmap='BrBG_r', vmin=0)
    plt.title('USLE C factor')
    plt.colorbar(im, shrink=0.4)
    plt.axis('off')
    #
    # RD
    fmap = reclassify(lulc, upvalues=ids, classes=lulc_df['P_USLE'].values)
    fmap = mask(fmap, basin)
    ax = fig.add_subplot(gs[4:, 8:10])
    im = plt.imshow(fmap, cmap='BrBG_r', vmin=0)
    plt.title('USLE P factor')
    plt.colorbar(im, shrink=0.4)
    plt.axis('off')
    #
    #
    ax = fig.add_subplot(gs[4:, 1:3])
    labels = lulc_df['LULCName']
    y_pos = np.arange(len(labels))
    areas = lulc_df['f_Canopy']
    ax.barh(y_pos, areas, align='center', color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('% area')
    ax.set_title('lulc distribution')


    if show:
        plt.show()
        plt.close(fig)
    else:
        expfile = folder + '/' + filename + '.png'
        plt.savefig(expfile)
        plt.close(fig)
        return expfile


def plot_shrumap_view(lulc, soils, meta, shruparam, filename='mapview', folder='C:/bin', metadata=True, show=False):
    from matplotlib.colors import ListedColormap
    cmap_lulc = ListedColormap(shruparam['ColorLULC'].values)
    cmap_soil = ListedColormap(shruparam['ColorSoil'].values)
    ranges_lulc = (np.min(shruparam['IdLULC'].values), np.max(shruparam['IdLULC'].values))
    ranges_soil = (np.min(shruparam['IdSoil'].values), np.max(shruparam['IdSoil'].values))
    #
    fig = plt.figure(figsize=(6, 4.5))  # Width, Height
    gs = mpl.gridspec.GridSpec(3, 4, wspace=0.0, hspace=0.0)
    #
    ax = fig.add_subplot(gs[:, :3])
    im = plt.imshow(lulc, cmap=cmap_lulc, vmin=ranges_lulc[0], vmax=ranges_lulc[1])
    im = plt.imshow(soils, cmap=cmap_soil, vmin=ranges_soil[0], vmax=ranges_soil[1], alpha=0.5)
    plt.title('shru')
    plt.axis('off')
    #
    # lengend
    ax = fig.add_subplot(gs[:, 3:])
    plt.text(x=0.0, y=1.0, s='LULC x Soils')
    plt.plot([0, 1], [0, 1], '.', alpha=0.0)
    hei = 0.95
    step = 0.2
    wid = len(shruparam['IdSoil'].unique())
    count = 0
    while count < len(shruparam['IdLULC'].values):
        xp = 0.1
        for i in range(wid):
            plt.plot(xp, hei, 's', color=shruparam['ColorLULC'].values[count])
            plt.plot(xp, hei, 's', color=shruparam['ColorSoil'].values[count], alpha=0.5)
            #plt.text(x=xp + 0.05, y=hei - 0.015, s=shruparam['IdSHRU'].values[count])
            xp = xp + step
            count = count + 1
        hei = hei - 0.05
    plt.ylim((0, 1))
    plt.xlim((0, 1.5))
    #
    if metadata:
        plt.text(x=0.0, y=0.3, s='Metadata:')
        plt.text(x=0.0, y=0.25, s='Rows: {}'.format(meta['nrows']))
        plt.text(x=0.0, y=0.2, s='Columns: {}'.format(meta['ncols']))
        plt.text(x=0.0, y=0.15, s='Cell size: {:.1f} m'.format(meta['cellsize']))
        plt.text(x=0.0, y=0.1, s='xll: {:.2f} m'.format(meta['xllcorner']))
        plt.text(x=0.0, y=0.05, s='yll: {:.2f} m'.format(meta['xllcorner']))
    plt.axis('off')
    #
    if show:
        plt.show()
        plt.close(fig)
    else:
        expfile = folder + '/' + filename + '.png'
        plt.savefig(expfile)
        plt.close(fig)
        return expfile


def plot_qmap_view(map, meta, colors, names, ranges, mapid='dem', filename='mapview', folder='C:/bin',
                   metadata=True, show=False):
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    #
    fig = plt.figure(figsize=(6, 4.5))  # Width, Height
    gs = mpl.gridspec.GridSpec(3, 4, wspace=0.0, hspace=0.0)
    #
    ax = fig.add_subplot(gs[:, :3])
    im = plt.imshow(map, cmap=cmap, vmin=ranges[0], vmax=ranges[1])
    plt.title(mapid)
    plt.axis('off')
    #
    # legend
    ax = fig.add_subplot(gs[:, 3:])
    plt.plot([0, 1], [0, 1], '.', alpha=0.0)
    plt.text(x=0.0, y=1.0, s='Classes')
    hei = 0.95
    for i in range(len(names)):
        plt.plot(0.0, hei, 's', color=colors[i])
        plt.text(x=0.15, y=hei-0.015, s=names[i])
        hei = hei - 0.05
    if metadata:
        plt.text(x=0.0, y=0.3, s='Metadata:')
        plt.text(x=0.0, y=0.25, s='Rows: {}'.format(meta['nrows']))
        plt.text(x=0.0, y=0.2, s='Columns: {}'.format(meta['ncols']))
        plt.text(x=0.0, y=0.15, s='Cell size: {:.1f} m'.format(meta['cellsize']))
        plt.text(x=0.0, y=0.1, s='xll: {:.2f} m'.format(meta['xllcorner']))
        plt.text(x=0.0, y=0.05, s='yll: {:.2f} m'.format(meta['xllcorner']))
    plt.axis('off')
    #
    if show:
        plt.show()
        plt.close(fig)
    else:
        expfile = folder + '/' + filename + '.png'
        plt.savefig(expfile)
        plt.close(fig)
        return expfile


def plot_map_view(map, meta, ranges, mapid='dem', filename='mapview', folder='C:/bin', metadata=True, show=False):
    map_dct = {'dem': ['BrBG_r', 'Elevation'],
               'slope': ['OrRd', 'Degrees'],
               'twi': ['YlGnBu', 'Index units'],
               'fto': ['Blues', 'Index units'],
               'etpat': ['Blues', 'Index units'],
               'catcha': ['Blues', 'Sq. Meters (log10)'],
               'basin': ['Greys', 'Boolean']}
    #
    fig = plt.figure(figsize=(6, 4.5))  # Width, Height
    gs = mpl.gridspec.GridSpec(3, 4, wspace=0.0, hspace=0.0)
    #
    ax = fig.add_subplot(gs[:, :3])
    if mapid == 'catcha':
        map = np.log10(map)
        ranges = np.log10(ranges)
    im = plt.imshow(map, cmap=map_dct[mapid][0], vmin=ranges[0], vmax=ranges[1])
    plt.title(mapid)
    plt.axis('off')
    plt.colorbar(im, shrink=0.4)
    #
    #
    ax = fig.add_subplot(gs[:, 3:])
    plt.text(x=-0.45, y=0.75, s=map_dct[mapid][1])
    if metadata:
        plt.text(x=0.0, y=0.3, s='Metadata:')
        plt.text(x=0.0, y=0.25, s='Rows: {}'.format(meta['nrows']))
        plt.text(x=0.0, y=0.2, s='Columns: {}'.format(meta['ncols']))
        plt.text(x=0.0, y=0.15, s='Cell size: {:.1f} m'.format(meta['cellsize']))
        plt.text(x=0.0, y=0.1, s='xll: {:.2f} m'.format(meta['xllcorner']))
        plt.text(x=0.0, y=0.05, s='yll: {:.2f} m'.format(meta['xllcorner']))
    plt.axis('off')
    #
    if show:
        plt.show()
        plt.close(fig)
    else:
        expfile = folder + '/' + filename + '.png'
        plt.savefig(expfile)
        plt.close(fig)
        return expfile


def plot_histograms(countmatrix, xs, ys, xt, yt, show=False, folder='C:/bin', filename='histograms', suff=''):
    #
    s1 = len(countmatrix)
    s3 = len(countmatrix[0])
    s2 = int(s1/2)
    fig = plt.figure(figsize=(16, 8))  # Width, Height
    gs = mpl.gridspec.GridSpec(s1 + s2, s1 + s3, wspace=0.0, hspace=0.0)
    #
    ax = fig.add_subplot(gs[:s2+2, :s3])
    x = np.arange(0, len(ys))
    plt.plot(x, ys)
    plt.xticks([])
    plt.xlim((0, len(ys)-1))
    plt.ylabel('% Frequency')
    plt.title('SHRU Histogram')
    #plt.axis('off')
    ax = fig.add_subplot(gs[s2:, :s3])
    im = plt.imshow(np.log10(countmatrix + 0.01), cmap='viridis', vmin=0)
    plt.title('')
    plt.ylabel('TWI values')
    plt.xlabel('SHRU Ids')
    plt.xticks(x, xs)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax = fig.add_subplot(gs[s2 + 2: s1 + s2 - 2, s3:])
    plt.yticks([])
    plt.plot(yt, xt)
    plt.xlabel('% Frequency')
    plt.title('TWI Histogram')
    plt.gca().invert_yaxis()
    #
    if show:
        plt.show()
        plt.close(fig)
    else:
        expfile = folder + '/' + filename + '.png'
        plt.savefig(expfile)
        plt.close(fig)
        return expfile
