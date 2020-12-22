import numpy as np
import pandas as pd
import input
import output
import matplotlib.pyplot as plt


def map_cn(flulc, flulcparam, fsoils, fsoilsparam, folder='C:/', filename='cn'):
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
    lulc_param_df = pd.read_csv(flulcparam, sep=';')
    soils_param_df = pd.read_csv(fsoilsparam, sep=';')
    lulc_classes = lulc_param_df['Value'].values
    soils_classes = soils_param_df['Value'].values * 100
    cn_a = lulc_param_df['CN-A'].values
    cn_b = lulc_param_df['CN-B'].values
    cn_c = lulc_param_df['CN-C'].values
    cn_d = lulc_param_df['CN-D'].values
    cn_values = (cn_a, cn_b, cn_c, cn_d)
    #
    # process data
    cn_class = (100 * soils) + lulc
    cn_map = lulc * 0.0
    for i in range(len(soils_classes)):
        for j in range(len(lulc_classes)):
            lcl_class = soils_classes[i] + lulc_classes[j]
            lcl_cn = cn_values[i][j]
            cn_map = cn_map + (cn_class == lcl_class) * lcl_cn
    #
    # export data
    export_file = output.asc_raster(cn_map, metalulc, folder, filename)
    return export_file


def map_grad(fslope, folder='C:/', filename='grad'):
    """
    derive the topographical gradient tan(B) from the slope in degrees
    :param fslope: string path to slope in degrees raster .asc file
    :param folder: string path to destination folder
    :param filename: string of file name
    :return: string path to file
    """
    # import data
    meta, slope = input.asc_raster(fslope)
    # process data
    slope_rad = np.pi * 2 * slope / 360
    grad = np.tan(slope_rad)
    plt.imshow(grad)
    plt.show()
    # export data
    export_file = output.asc_raster(grad, meta, folder, filename)
    return export_file