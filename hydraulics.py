import pandas as pd
import numpy as np


def area_trapezoidal(y, b, m):
    return (b + m * y) * y


def area_circular(y, r):
    alpha = np.arcsin((y - r) / r)
    print(alpha, end='\t')
    theta = ((np.pi / 2) + alpha)
    print(theta)
    return (1 / 8) * ((2 * theta) - np.sin((2 * theta))) * np.square(2 * r)


def hydraulic_radius(a, p):
    """
    computes the hydraulic radius
    :param a: wetted cross section in m2
    :param p: wetted perimeter in m
    :return: hydraulic radius in m
    """
    return a / p


def q_manning(n, a, rh, s):
    """
    Computes the discharge Q of the Manning Equation
    :param n: Manning roughness parameter
    :param a: wetted cross section in m2
    :param rh: hydraulic radius in m
    :param s: slope in m/m
    :return: dicharge in m3/s
    """
    return a * np.power(rh, 2/3) * np.power(s, 0.5) / n


def n_manning(q, a, rh, s):
    """
    Computes the Manning roughness parameter
    :param q: discharge Q of the Manning Equation in m3/s
    :param a: wetted cross section in m2
    :param rh: hydraulic radius in m
    :param s: slope in m/m
    :return: Manning roughness parameter
    """
    return (a * np.power(x1=rh, x2=2 / 3) * np.power(x1=s, x2=0.5)) / q


def get_wetted_geometry(y, df_segments):
    """
    Get the wetted geometry paramteres Wet area and Wet perimeter for a given section segments and Y (water vertical stage)
    :param y: float - water vertical stage in the same coordinate system of segments
    :param df_segments: pandas dataframe of section segments provided in the section_segments() function
    :return: float of wetted area and float of wetted perimeter
    """
    segs_df = df_segments.copy()
    # filter dataframe:
    segs_inner_df = segs_df.query('Ymax <= {}'.format(y))
    segs_border_df = segs_df.query('Ymax > {} and Ymin <= {}'.format(y, y))
    # loop to get inner area:
    inner_a_lst = list()
    for i in range(len(segs_inner_df)):
        lcl_a = segs_inner_df['AN'].values[i] + (segs_inner_df['DX'].values[i] * (y - segs_inner_df['Ymax'].values[i]))
        inner_a_lst.append(lcl_a)
    inner_a = np.sum(inner_a_lst)
    inner_p = segs_inner_df['P'].sum()
    # loop to get border area and perimeter:
    border_a = 0.0
    border_p = 0.0
    for i in range(len(segs_border_df)):
        lcl_dy = y - segs_border_df['Ymin'].values[i]
        lcl_x = (y - segs_border_df['B'].values[i]) / segs_border_df['A'].values[i]
        lcl_dx = np.abs(lcl_x - segs_border_df['X_Ymin'].values[i])
        lcl_a = (np.abs(lcl_dx) * np.abs(lcl_dy)) / 2
        lcl_p = np.sqrt(np.square(lcl_dx) + np.square(lcl_dy))
        border_a = border_a + lcl_a
        border_p = border_p + lcl_p
    wet_a = inner_a + border_a
    wet_p = inner_p + border_p
    return wet_a, wet_p


def section_segments(df_section, stage_field='StageId', x_field='X', y_field='Y'):
    """
    Computes the segments of a channel section
    :param df_section: pandas dataframe object describing the section geometry with at least 3 fields:
    stage index (int), X coordinates (float) and Y coordinates (float)
    :param stage_field: string name of stage id
    :param x_field: string name of X coordinates
    :param y_field: string name of Y coordinates
    :return: pandas dataframe object of section segments. Fields:
    SegmentId,
    X1, X2, Y1, Y2, Ymax, Ymin, X_Ymax, X_Ymin, DX, DY, A (y=ax+b parameter), B (y=ax+b parameter),
    AN (native area of the segment), P (segment perimeter)
    """
    section_df = df_section.copy()
    segments_list = list()
    x1_lst = list()
    x2_lst = list()
    y1_lst = list()
    y2_lst = list()
    ymax_lst = list()
    ymin_lst = list()
    x_ymax_lst = list()
    x_ymin_lst = list()
    dx_lst = list()
    dy_lst = list()
    a_lst = list()
    b_lst = list()
    an_lst = list()
    p_list = list()
    # loop to get segments metadata
    for i in range(len(section_df) - 1):
        # create id
        lcl_id = int(str(section_df[stage_field].values[i]) + str(section_df[stage_field].values[i + 1]))
        # get x and ys
        lcl_x1 = section_df[x_field].values[i]
        lcl_x2 = section_df[x_field].values[i + 1]
        lcl_y1 = section_df[y_field].values[i]
        lcl_y2 = section_df[y_field].values[i + 1]
        # deltas
        lcl_dx = lcl_x2 - lcl_x1
        lcl_dy = lcl_y2 - lcl_y1
        # curve parameters
        if lcl_dx == 0:  # vertical bank
            lcl_a = 10000000
        else:
            lcl_a = lcl_dy / lcl_dx
        lcl_b = lcl_y1 - (lcl_a * lcl_x1)
        # segment native area
        lcl_an = (np.abs(lcl_dx) * np.abs(lcl_dy)) / 2
        # segment native perimeter
        lcl_p = np.sqrt(np.square(lcl_dx) + np.square(lcl_dy))
        # max and mins
        lcl_ymax = np.max((lcl_y1, lcl_y2))
        lcl_ymin = np.min((lcl_y1, lcl_y2))
        if lcl_ymax == lcl_y1:
            lcl_x_ymax = lcl_x1
            lcl_x_ymin = lcl_x2
        else:
            lcl_x_ymax = lcl_x2
            lcl_x_ymin = lcl_x1
        segments_list.append(lcl_id)
        x1_lst.append(lcl_x1)
        x2_lst.append(lcl_x2)
        y1_lst.append(lcl_y1)
        y2_lst.append(lcl_y2)
        ymax_lst.append(lcl_ymax)
        ymin_lst.append(lcl_ymin)
        x_ymax_lst.append(lcl_x_ymax)
        x_ymin_lst.append(lcl_x_ymin)
        dx_lst.append(lcl_dx)
        dy_lst.append(lcl_dy)
        a_lst.append(lcl_a)
        b_lst.append(lcl_b)
        an_lst.append(lcl_an)
        p_list.append(lcl_p)
    # built dataframe
    segments_df = pd.DataFrame({'SegmentId': segments_list,
                                'X1': x1_lst,
                                'X2': x2_lst,
                                'Y1': y1_lst,
                                'Y2': y2_lst,
                                'Ymax': ymax_lst,
                                'Ymin': ymin_lst,
                                'X_Ymax': x_ymax_lst,
                                'X_Ymin': x_ymin_lst,
                                'DX': dx_lst,
                                'DY': dy_lst,
                                'A': a_lst,
                                'B': b_lst,
                                'AN': an_lst,
                                'P': p_list})
    return segments_df





