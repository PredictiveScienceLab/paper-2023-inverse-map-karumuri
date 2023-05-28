#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import warnings
import numpy as np

import torch

import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt

from sklearn.cluster import MeanShift
from sklearn.neighbors.kde import KernelDensity
from scipy.ndimage.filters import gaussian_filter

import random

np.random.seed(37)
torch.manual_seed(37)
random.seed(37)


rangex = (-0.35, 2.25)
rangey = (-1.3, 1.3) #(-1.0, 3.0) #(-3.0, 3.0)

def segment_points_np(p_, length, angle):
    p = np.array(p_)
    angle = np.array(angle)
    p[:,0] += length * np.cos(angle)
    p[:,1] += length * np.sin(angle)
    return p_, p

def forward_process_np(xi, lens):
    xi = xi.reshape(-1,4)
    starting_pos = np.zeros((xi.shape[0], 2))
    starting_pos[:,1] = xi[:, 0]
    A = starting_pos
    _, B = segment_points_np(A, lens[0], xi[:,1])
    _, C = segment_points_np(B, lens[1], xi[:,1] + xi[:,2])
    _, D = segment_points_np(C, lens[2], xi[:,1] + xi[:,2] + xi[:,3])
    return A, B, C, D;

def draw_isolines(samples, lens, color, filter_width):
    if not filter_width > 0:
        return

    xi = np.array(samples)
    
    xi0, xi1, xi2, y = forward_process_np(xi, lens)

    hist, xbins, ybins = np.histogram2d(y[:, 0], y[:, 1], bins=600, range=[rangex, rangey], density=True)
    hist = gaussian_filter(hist, filter_width)

    percentile = 0.03 * np.sum(hist)
    for q in np.logspace(-99, np.log10(np.max(hist)), 8000, endpoint=True):
        if np.sum(hist[hist < q]) > percentile: break
    else:
        q = 1.

    X, Y = np.meshgrid(0.5 * (xbins[:-1] + xbins[1:]),
                       0.5 * (ybins[:-1] + ybins[1:]))

    plt.contour(X, Y, hist.T, [q], colors=color, linewidths=0.7, zorder=3)
    
    
def update_plot(xi_samples, xi_data, y_data, lens, color_code=0, filter_width=4., arrows=False, target_label=False):

    fig = plt.figure(figsize=(7.48/2, 7.48/2/1.618)) # fig = plt.figure()
    axarr = fig.add_subplot(1,1,1) # here is where you add the subplot to f

    # if prior:
    #     color_code = 4
    # else:
    #     color_code = random.randint(0, 3)
        
    cmap = cm.tab20c
    colors = [[cmap(4*c_index), cmap(4*c_index+1), cmap(4*c_index+2)] for c_index in range(5)][color_code]
    
    lens = np.array(lens)
    xi = np.array(xi_samples)
    y_target = y_data[0]
    
    xi0, xi1, xi2, xi3 = forward_process_np(xi, lens)

    plt.axvline(x=0, ls=':', c='gray', linewidth=.75)
    if not arrows:
        print('---------------')
        # plt.axvline(x=y_target[0], ls='-', c='gray', linewidth=.5, alpha=.5, zorder=-1)
        # plt.axhline(y=y_target[1], ls='-', c='gray', linewidth=.5, alpha=.5, zorder=-1)
        l_cross = 0.6
        plt.plot([y_target[0] - l_cross, y_target[0] + l_cross], [y_target[1], y_target[1]], ls='-', c='gray', linewidth=.75, alpha=.5, zorder=-1)
        plt.plot([y_target[0], y_target[0]], [y_target[1] - l_cross, y_target[1] + l_cross], ls='-', c='gray', linewidth=.75, alpha=.5, zorder=-1)
        if target_label:
            print('****************')
            plt.text(y_target[0] + 0.05, y_target[1] + 0.05, r'$\mathbf{y}$', ha='left', va='bottom', color='gray', fontsize=10) #r'$\mathbf{y}^*$'


    opts = {'alpha':0.05, 'scale':1, 'angles':'xy', 'scale_units':'xy', 'headlength':0, 'headaxislength':0, 'linewidth':1.0, 'rasterized':True}
    plt.quiver(xi0[:,0], xi0[:,1], (xi1-xi0)[:,0], (xi1-xi0)[:,1], **{'color': colors[0], **opts})
    plt.quiver(xi1[:,0], xi1[:,1], (xi2-xi1)[:,0], (xi2-xi1)[:,1], **{'color': colors[1], **opts})
    plt.quiver(xi2[:,0], xi2[:,1], (xi3-xi2)[:,0], (xi3-xi2)[:,1], **{'color': colors[2], **opts})
    
    exemplar_color = colors[0] * np.array([.5, .5, .5, 1])
    A, B, C, D = forward_process_np(xi_data, lens)
    plt.plot([A[0,0], B[0,0], C[0,0]],
             [A[0,1], B[0,1], C[0,1]],
                     '-', color=exemplar_color, linewidth=1, zorder=4)
    ground_truth = plt.arrow(C[0,0], C[0,1],
          D[0,0] - C[0,0], D[0,1] - C[0,1],
         color=exemplar_color, linewidth=1, head_width=0.05, head_length=0.04, overhang=0.1, length_includes_head=True, zorder=4, label="ground truth")
    plt.scatter([A[0,0],], [A[0,1],],
            s=100, marker='s', linewidth=1, edgecolors=exemplar_color, facecolors='white', zorder=3)
    plt.scatter([A[0,0], B[0,0], C[0,0]],
                [A[0,1], B[0,1], C[0,1]],
                    s=10, linewidth=1, edgecolors=exemplar_color, facecolors='white', zorder=5)

    # plt.xlim(*rangex); plt.ylim(*rangey)
    
    draw_isolines(xi, lens, colors, filter_width)
    # plt.gca().set_xticks([]); plt.gca().set_yticks([])

    return(fig)


# In[ ]:




