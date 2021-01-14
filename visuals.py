import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def pannel_topmodel(image, imax, t, x1, x2, x3, vline=20, folder='C:/bin', filename='pannel_topmodel', suff=''):
    fig = plt.figure(figsize=(16, 8))  # Width, Height
    gs = mpl.gridspec.GridSpec(3, 6, wspace=0.8, hspace=0.6)
    #
    plt.subplot(gs[0:, 0:2])
    plt.imshow(image, cmap='jet', vmin=0, vmax=imax)
    plt.axis('off')
    plt.title('Local deficit (mm)')
    plt.colorbar()
    #
    # set x ticks
    locs = np.arange(0, len(t), 12)
    labels = t[locs]
    #
    y = x1
    plt.subplot(gs[0, 2:])
    var = y[vline]
    plt.title('Precipitation: {:.1f} mm'.format(var), loc='left')
    plt.ylabel('mm')
    plt.plot(t, y)
    plt.vlines(t[vline], ymin=0, ymax=np.max(y), colors='r')
    plt.xticks(locs, labels)
    #
    y = x2
    plt.subplot(gs[1, 2:])
    var = y[vline]
    plt.title('Baseflow: {:.4f} mm'.format(var), loc='left')
    plt.ylabel('mm')
    plt.plot(t, y, 'navy')
    plt.vlines(t[vline], ymin=0, ymax=np.max(y), colors='r')
    plt.xticks(locs, labels)
    #
    y = x3
    plt.subplot(gs[2, 2:])
    var = y[vline]
    plt.title('Global deficit: {:.1f} mm'.format(var), loc='left')
    plt.ylabel('mm')
    plt.plot(t, x3, 'tab:orange')
    plt.vlines(t[vline], ymin=0, ymax=np.max(y), colors='r')
    plt.xticks(locs, labels)
    #plt.show()
    #
    filepath = folder + '/' + filename + '_' + suff + '.png'
    plt.savefig(filepath)
    plt.close(fig)
    return filepath
