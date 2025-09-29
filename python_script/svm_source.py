# -*- coding: utf-8 -*-
"""
@author: J. Salmon, A. Gramfort, C. Vernade
"""

###############################################################################
#               Import part
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import pylab as pl

###############################################################################
#              Colorblind Purple-Lilac Palette
###############################################################################

PURPLE_COLORBLIND_PALETTE = [
    '#4A148C',
    '#7B1FA2', 
    '#9C27B0', 
    '#BA68C8', 
    '#E1BEE7', 
    '#CE93D8',  
    '#AB47BC',  
    '#8E24AA',  
]

PURPLE_BINARY_PALETTE = [
    '#4A148C', 
    '#E1BEE7', 
    '#7B1FA2',  
]

###############################################################################
#               Data Generation
###############################################################################


def rand_gauss(n=100, mu=[1, 1], sigmas=[0.1, 0.1]):
    """ Sample n points from a Gaussian variable with center mu,
    and std deviation sigma
    """
    d = len(mu)
    res = np.random.randn(n, d)
    return np.array(res * sigmas + mu)


def rand_bi_gauss(n1=100, n2=100, mu1=[1, 1], mu2=[-1, -1], sigmas1=[0.1, 0.1],
                  sigmas2=[0.1, 0.1]):
    """ Sample n1 and n2 points from two Gaussian variables centered in mu1,
    mu2, with respective std deviations sigma1 and sigma2
    """
    ex1 = rand_gauss(n1, mu1, sigmas1)
    ex2 = rand_gauss(n2, mu2, sigmas2)
    y = np.hstack([np.ones(n1), -1 * np.ones(n2)])
    X = np.vstack([ex1, ex2])
    ind = np.random.permutation(n1 + n2)
    return X[ind, :], y[ind]

###############################################################################
#           Displaying labeled data
###############################################################################

symlist = ['o', 's', 'D', 'x', '+', '*', 'p', 'v', '-', '^']


def plot_2d(data, y=None, w=None, alpha_choice=1):
    """ Plot in 2D the dataset data, colors and symbols according to the
    class given by the vector y (if given); the separating hyperplan w can
    also be displayed if asked"""

    k = np.unique(y).shape[0] if y is not None else 1
    
    if k <= 3:
        color_list = PURPLE_BINARY_PALETTE[:k]
    else:
        color_list = PURPLE_COLORBLIND_PALETTE[:k]
    
    # Style theme_minimal() de ggplot2
    ax = plt.gca()
    ax.set_facecolor('white')
    
    if y is None:
        labs = [""]
        idxbyclass = [range(data.shape[0])]
    else:
        labs = np.unique(y)
        idxbyclass = [np.where(y == labs[i])[0] for i in range(len(labs))]

    for i in range(len(labs)):
        plt.scatter(data[idxbyclass[i], 0], data[idxbyclass[i], 1],
                    color=color_list[i], s=100, marker=symlist[i], 
                    alpha=0.8)
    
    plt.ylim([np.min(data[:, 1]), np.max(data[:, 1])])
    plt.xlim([np.min(data[:, 0]), np.max(data[:, 0])])
    
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='black')
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    
    mx = np.min(data[:, 0])
    maxx = np.max(data[:, 0])
    if w is not None:
        plt.plot([mx, maxx], [mx * -w[1] / w[2] - w[0] / w[2],
                              maxx * -w[1] / w[2] - w[0] / w[2]],
                 color='#4A148C', linewidth=3, alpha=alpha_choice)

###############################################################################
#           Displaying tools for the Frontiere
###############################################################################


def frontiere(f, X, y, w=None, step=50, alpha_choice=1, colorbar=True,
              samples=True):
    """ trace la frontiere pour la fonction de decision f"""
    
    min_tot0 = np.min(X[:, 0])
    min_tot1 = np.min(X[:, 1])
    max_tot0 = np.max(X[:, 0])
    max_tot1 = np.max(X[:, 1])
    delta0 = (max_tot0 - min_tot0)
    delta1 = (max_tot1 - min_tot1)
    
    xx, yy = np.meshgrid(np.arange(min_tot0, max_tot0, delta0 / step),
                         np.arange(min_tot1, max_tot1, delta1 / step))
    z = np.array([f(vec) for vec in np.c_[xx.ravel(), yy.ravel()]])
    z = z.reshape(xx.shape)
    labels = np.unique(z)
    
    if len(labels) <= 3:
        color_list = PURPLE_BINARY_PALETTE[:len(labels)]
    else:
        color_list = PURPLE_COLORBLIND_PALETTE[:len(labels)]
    
    background_colors = []
    for color in color_list:
        rgb = plt.matplotlib.colors.to_rgb(color)
        light_rgb = tuple(0.3 * c + 0.7 for c in rgb)
        background_colors.append(light_rgb)
    
    my_cmap = ListedColormap(background_colors)
    
    ax = plt.gca()
    ax.set_facecolor('white')
    
    plt.imshow(z, origin='lower', interpolation="mitchell", alpha=0.60,
               cmap=my_cmap, extent=[min_tot0, max_tot0, min_tot1, max_tot1])
    
    if colorbar is True:
        cbar = plt.colorbar(ticks=labels)
        cbar.ax.set_yticklabels(labels)
        cbar.ax.tick_params(colors='black')
        cbar.set_label('Classes', color='black', fontweight='normal')

    labels_y = np.unique(y)
    k = np.unique(y).shape[0]
    
    if k <= 3:
        point_colors = PURPLE_BINARY_PALETTE[:k]
    else:
        point_colors = PURPLE_COLORBLIND_PALETTE[:k]
    
    if samples is True:
        for i, label in enumerate(y):
            label_num = np.where(labels_y == label)[0][0]
            plt.scatter(X[i, 0], X[i, 1], color=point_colors[label_num],
                        s=120, marker=symlist[label_num], alpha=0.9)
    
    plt.xlim([min_tot0, max_tot0])
    plt.ylim([min_tot1, max_tot1])
    
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    
    if w is not None:
        plt.plot([min_tot0, max_tot0],
                 [min_tot0 * -w[1] / w[2] - w[0] / w[2],
                  max_tot0 * -w[1] / w[2] - w[0] / w[2]],
                 color='#4A148C', linewidth=4, alpha=alpha_choice)


def plot_gallery(images, titles, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    fig = pl.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    fig.patch.set_facecolor('white')
    
    pl.subplots_adjust(bottom=0, left=.01, right=.99, top=.90,
                       hspace=.35)
    for i in range(n_row * n_col):
        ax = pl.subplot(n_row, n_col, i + 1)
        ax.set_facecolor('white')
        pl.imshow(images[i])
        pl.title(titles[i], size=12, color='#4A148C', fontweight='normal')
        pl.xticks(())
        pl.yticks(())
        
        for spine in ax.spines.values():
            spine.set_visible(False)


def title(y_pred, y_test, names):
    pred_name = names[int(y_pred)].rsplit(' ', 1)[-1]
    true_name = names[int(y_test)].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
