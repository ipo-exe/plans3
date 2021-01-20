import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


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
    :param folder: string folder path to expor image
    :param filename: string of file name
    :param suff: string of suffix
    :return: string file path of plot
    """
    #
    fig = plt.figure(figsize=(16, 8))  # Width, Height
    gs = mpl.gridspec.GridSpec(3, 6, wspace=0.8, hspace=0.6)
    # plot image
    plt.subplot(gs[0:, 0:2])
    im = plt.imshow(image, cmap=cmap, vmin=0, vmax=imax)
    plt.axis('off')
    plt.title(titles[0])
    plt.colorbar(im, shrink=0.4)
    #
    # set x ticks
    size = len(t)
    spaces = int(size / 5)
    locs = np.arange(0, size, spaces)
    labels = t[locs]
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
    plt.xticks(locs, labels)
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
    plt.xticks(locs, labels)
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
    plt.xticks(locs, labels)
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
    #
    fig = plt.figure(figsize=(16, 9))  # Width, Height
    gs = mpl.gridspec.GridSpec(4, 10, wspace=0.8, hspace=0.6)
    #
    # images
    # Left Upper
    ax = fig.add_subplot(gs[0:2, 0:2])
    im = plt.imshow(im4[0], cmap=cmaps[0], vmin=0, vmax=imax)
    plt.axis('off')
    plt.title(imtitles[0])
    plt.colorbar(im, shrink=0.4)
    # Left bottom
    ax = fig.add_subplot(gs[2:4, 0:2])
    im = plt.imshow(im4[1], cmap=cmaps[1], vmin=0, vmax=0.5 * imax)
    plt.axis('off')
    plt.title(imtitles[1])
    plt.colorbar(im, shrink=0.4)
    # Right Upper
    ax = fig.add_subplot(gs[0:2, 2:4])
    im = plt.imshow(im4[2], cmap=cmaps[2], vmin=0, vmax=2.33)
    plt.axis('off')
    plt.title(imtitles[2])
    plt.colorbar(im, shrink=0.4)
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
    #print(locs)
    #print(labels)
    #
    ax = fig.add_subplot(gs[0, 5:])
    lcl = 0
    y = y4[lcl]
    plt.plot(t, y)
    plt.vlines(t[vline], ymin=y4min[lcl], ymax=y4max[lcl], colors='r')
    plt.plot(t[vline], y[vline], 'ro')
    #plt.xticks(locs, labels)
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
    #plt.xticks(locs, labels)
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
    #plt.xticks(locs, labels)
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
    #plt.xticks(locs, labels)
    var = y[vline]
    plt.title('{}: {:.2f}'.format(ytitles[lcl], var), loc='left')
    plt.ylabel(ylabels[lcl])
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
    visualize the topmodel variables in a single pannel
    :param dataframe: pandas dataframe from hydrology.topmodel_sim()
    :param grid: boolean for grid
    :param folder: string to destination directory
    :param filename: string file name
    :param suff: string suffix in file name
    :param show: boolean control to show figure instead of saving it
    :return: string file path
    """
    #
    fig = plt.figure(figsize=(16, 12))  # Width, Height
    gs = mpl.gridspec.GridSpec(7, 8, wspace=0.8, hspace=0.6)
    #
    ind = 0
    ax = fig.add_subplot(gs[ind, 0:])
    plt.plot(dataframe['Date'], dataframe['Prec'])
    plt.ylabel('Prec mm')
    ax2 = ax.twinx()
    plt.plot(dataframe['Date'], dataframe['Temp'], 'tab:orange')
    plt.ylabel('Temp °C')
    plt.grid(grid)
    #
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['PET'], 'k', label='PET')
    plt.plot(dataframe['Date'], dataframe['ET'], 'tab:red', label='ET')
    plt.plot(dataframe['Date'], dataframe['Tp'], 'r', label='Tp')
    plt.ylabel('PET & ET\nmm')
    plt.grid(grid)
    #plt.legend()
    #
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['S1'])
    plt.ylabel('S1\nmm')
    plt.grid(grid)
    #
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['R'], 'tab:orange')
    plt.plot(dataframe['Date'], dataframe['Inf'], 'k')
    plt.ylabel('R & Inf\nmm')
    plt.grid(grid)
    #
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['D'], 'k')
    plt.plot(dataframe['Date'], dataframe['S2'])
    plt.ylabel('S2 & D\nmm')
    plt.grid(grid)
    #
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    plt.plot(dataframe['Date'], dataframe['Qv'])
    plt.ylabel('Qv\nmm')
    plt.grid(grid)
    #
    ind = ind + 1
    ax2 = fig.add_subplot(gs[ind, 0:], sharex=ax)
    if qobs:
        plt.plot(dataframe['Date'], dataframe['Qobs'], 'tab:orange')
    plt.plot(dataframe['Date'], dataframe['Q'])
    plt.plot(dataframe['Date'], dataframe['Qb'], 'navy')
    plt.ylim(0.9 * np.min((dataframe['Q'].values, dataframe['Qobs'].values)), 1.1 * np.max((dataframe['Q'].values, dataframe['Qobs'].values)))
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
