# builtin
import os

# external
import numpy as np
from scipy import stats
import pandas as pd
try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
except ImportError:
    pass

BASE_PATH = os.path.dirname(__file__)
LIB_PATH = os.path.join(BASE_PATH, 'lib')


def n_tag(n, n_max=None, sep=' '):
    if n_max is None:
        return sep + f'(n={n})'    
    return sep + f'(n={n}/{n_max})'


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
        
    
def nearest(array, value, is_sorted=False):
    idx = where_nearest(array,value, is_sorted=is_sorted)
    if idx is None:
        return np.nan
    return array[idx]


def snip(a, a_min=-np.inf, a_max=np.inf):
    if a_min is None:
        a_min = -np.inf
    if a_max is None:
        a_max = np.inf
        
    a = np.array(a)
    return a[ (a >= a_min) & (a <= a_max) ]


def qsnip(a, q_min, q_max, retlims=False):
    if len(a) == 0:
        return np.array([])
    
    a_min = np.quantile(a, q_min)
    a_max = np.quantile(a, q_max)
    
    if not retlims:
        return snip(a, a_min, a_max)
    else:
        return snip(a, a_min, a_max), [a_min, a_max]


def qclip(a, q_min, q_max, retlims=False):
    if len(a) == 0:
        return np.array([])
    
    a_min = np.quantile(a, q_min) if q_min else -np.inf
    a_max = np.quantile(a, q_max) if q_max else np.inf
    
    if not retlims:
        return np.clip(a, a_min, a_max)
    else:
        return np.clip(a, a_min, a_max), [a_min, a_max]


def margined(interval, margin):
    if not hasattr(margin, '__len__'):
        margin = [margin, margin]
    
    assert hasattr(margin, '__len__') and len(margin) == 2
    
    interval = [min(interval), max(interval)]
    ptp = np.ptp(interval)
    return [interval[0] - margin[0]*ptp, interval[1] + margin[1]*ptp]


def int_hist(x, density=False, **kwargs):
    lb = min(x)     
    ub = max(x)
    plt.hist(x, bins = 0.5 + np.arange(lb-1,ub+1), edgecolor='black', density=density, **kwargs)
    plt.xticks(np.arange(lb,ub+1))
    return None


def empirical_cdf(arr, density=False, reverse=False, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    arr = np.array(arr)
    
    y = np.arange(len(arr))
    if reverse:
        y = len(arr) - 1 - y
    
    if density:
        y = y / ( len(arr) - 1 )
    
    ax.plot( sorted(arr), y, **kwargs )
    
    return sorted(arr), y


def scatter_with_kde(x, y, s=1, subsampling_size=None, subsampling_replace=True, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
        
    x,y = np.array(x).astype(float), np.array(y).astype(float)
    
    if subsampling_size is not None:
        idx = np.random.choice(np.arange(len(x)), size=subsampling_size, replace=subsampling_replace)
        kde = stats.gaussian_kde([x[idx],y[idx]])
    else:
        kde = stats.gaussian_kde([x,y])
        
    zz = kde([x,y])
    cc = cm.jet((zz-zz.min())/(zz.max()-zz.min()))
    ax.scatter(x,y, s=s, facecolors=cc, **kwargs)


def plot_qcut_boxplot(x, y, q=10, qlim=None, vert=None, whis=None, manage_ticks=False, ax=None):
    x = np.array(x)
    y = np.array(y)
    if vert is None:
        vert = True

    if not ax:
        ax = plt.gca()
    
    if not vert:
        y, x = x, y
    
    if qlim is not None:
        x_min, x_max = np.percentile(x, 100*qlim[0]), np.percentile(x, 100*qlim[1])
        idx = (x >= x_min) & (x <= x_max)
        x = x[idx]
        y = y[idx]
    
    
    labels, bins = pd.qcut(x, q, labels=range(q), retbins=True)
    data = [ y[labels == label] for label in range(10) ]
    ax.boxplot(data, vert=vert, whis=whis, positions=(bins[:-1] + bins[1:])/2, widths=(bins[:-1] - bins[1:]), showfliers=False, manage_ticks=manage_ticks)


def find(condition):
    if isinstance(condition, list):
        condition = np.array(condition)
    if isinstance(condition, np.ndarray):
        res, = np.nonzero(np.ravel(condition))
    if isinstance(condition, pd.Series):
        res = condition.index[condition]
    return res


def scalar_proj(v_from, v_to, axis=None):
    assert (axis is not None) or v_from.ndim == v_to.ndim == 1

    if axis is not None:
        return np.sum(v_from * v_to, axis=axis) / np.sum(v_to * v_to, axis=axis)

    return np.dot(v_from, v_to) / np.dot(v_to, v_to)


def cosine_similarity(x, y, axis=None):
    assert (axis is not None) or x.ndim == y.ndim == 1

    if axis is not None:
        return np.sum(x * y, axis=axis) / np.linalg.norm(x, 2, axis=axis) / np.linalg.norm(y, 2, axis=axis)

    return np.dot(x,y) / np.linalg.norm(x,2) / np.linalg.norm(y,2)





