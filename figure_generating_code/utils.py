# buitin
import colorsys
import functools


# external
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pandas as pd

    
def margined(interval, margin=0.05):
    if not hasattr(margin, '__len__'):
        margin = [margin, margin]
    
    assert hasattr(margin, '__len__') and len(margin) == 2
    
    interval = [min(interval), max(interval)]
    ptp = np.ptp(interval)
    return [interval[0] - margin[0]*ptp, interval[1] + margin[1]*ptp]


def set_limits_equal(ax=None, identity_line='k--', **kwargs):
    if not ax:
        ax = plt.gca()
        
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_ylim( [min(xlim[0],ylim[0]),max(xlim[1],ylim[1])])
    ax.set_xticks(ax.get_yticks())
    ax.set_xlim( [min(xlim[0],ylim[0]),max(xlim[1],ylim[1])])
    if identity_line:
        ax.plot([0, 1], [0, 1], identity_line, transform=ax.transAxes, **kwargs)


def set_axes_equal(ax=None, identity_line='k--', **kwargs):
    if not ax:
        ax = plt.gca()
        
    set_limits_equal(ax=ax, identity_line = identity_line, **kwargs)
    ax.set_aspect('equal', adjustable='box')


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def darken_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], amount * c[1], c[2])


def saturate_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], c[1], 1 - amount * (1-c[2]))


def where_nearest(array,value,is_sorted=False):
    if len(array) == 0:
        return None

    keys = np.arange(len(array))

    if isinstance(array, pd.Series):
        keys = array.index
        array = array.values
    
    if not is_sorted:  
        value = np.array(value)
        if value.ndim == 0:
            idx = np.argmin(np.abs(array - value))
            return keys[idx]
        else:
            assert array.ndim == 1
            idx = np.argmin(np.abs(array - np.expand_dims(value, -1)), -1)
            return keys[idx]
    
    idx = np.searchsorted(array, value, side="left")
    if idx.ndim == 0:
        if idx > 0 and (idx == len(array) or value - array[idx-1] < array[idx] - value):
            return keys[idx - 1]
        else:
            return keys[idx]
    else:
        idx = np.clip(idx, 1, len(array) - 1)
        return keys[idx - (value - array[idx-1] < array[idx] - value)]
        
    
def intersection(*args):
    args = map(set, args)
    return list(functools.reduce(lambda x,y : x & y, args))


def compute_yticklabel_annotation_points(num_newlines=0, fontsize=None, pad=None, size=None, linespacing = None, yticklabel_fontsize_to_pt=None):
    if fontsize is None:
        fontsize = matplotlib.rcParams['ytick.labelsize']
        
    if isinstance(fontsize, str):
        fontsize = matplotlib.font_manager.font_scalings[fontsize] * matplotlib.rcParams['font.size'] # Default 10
        
    if pad is None:
        pad = matplotlib.rcParams['ytick.major.pad'] # Default 3.5
        
    if size is None:
        size = matplotlib.rcParams['ytick.major.size'] # Default 3.5
    
    if linespacing is None:
        linespacing = 1.2
    fontsize_to_pt = 0.715821613307
    linespacing_base = -0.27765476834294284

    if yticklabel_fontsize_to_pt is None:
        yticklabel_fontsize_to_pt = fontsize_to_pt

    return -pad - size - fontsize_to_pt*fontsize - (linespacing-linespacing_base)*yticklabel_fontsize_to_pt*fontsize * num_newlines