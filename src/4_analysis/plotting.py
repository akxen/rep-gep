"""Creating plots"""

import os

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator


def plot_base():
    """Plot base image"""
    fig, ax = plt.subplots()

    rectangles_1 = []
    rectangles_2 = []
    rectangles_3 = []
    rect_1 = Rectangle((0, 0), 150, 20, color='red')
    rect_2 = Rectangle((150, 0), 125, 45, color='blue')
    rect_3 = Rectangle((275, 0), 100, 60, color='grey')

    rectangles_1.append(rect_1)
    rectangles_2.append(rect_2)
    rectangles_3.append(rect_3)

    pc_1 = PatchCollection(rectangles_1, facecolors='#8a6d3d', alpha=0.8, edgecolor='k', linewidth=0.5)
    pc_2 = PatchCollection(rectangles_2, facecolors='#3d728a', alpha=0.8, edgecolor='k', linewidth=0.5)
    pc_3 = PatchCollection(rectangles_3, facecolors='#9c001f', alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.add_collection(pc_1)
    ax.add_collection(pc_2)
    ax.add_collection(pc_3)
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 75])
    ax.tick_params(labelsize=6)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.xaxis.set_minor_locator(MultipleLocator(20))
    ax.set_xlabel('Energy offers (MWh)', fontsize=7)
    ax.set_ylabel('Price ($/MWh)', fontsize=7, labelpad=-0.1)

    fig.set_size_inches(1.9685, 2.756)
    fig.subplots_adjust(left=0.17, bottom=0.13, top=0.99, right=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig('output/price_setters_1.png', dpi=800)
    fig.savefig('output/price_setters_1.pdf', transparent=True)

    return fig, ax


def demand_plot_1(fig, ax):
    """Add demand"""

    ax.plot([200, 200], [0, 68], color='k', linestyle='--', alpha=0.9, linewidth=0.7)
    ax.plot([0, 150], [45, 45], color='k', linestyle='--', alpha=0.9, linewidth=0.6)
    ax.text(150, 70, 'Demand', fontsize=6)
    ax.text(-40, 44, '$\lambda$', fontsize=7, color='red')

    fig.savefig('output/price_setters_2.png', dpi=800)
    fig.savefig('output/price_setters_2.pdf', transparent=True)

    return fig, ax


def demand_plot_2(fig, ax):
    """Add demand"""

    ax.plot([0, 150], [45, 45], color='k', linestyle='--', alpha=0.9, linewidth=0.6)
    ax.text(-40, 44, '$\lambda$', fontsize=7, color='red')
    ax.text(80, 46.5, r'$C_{g^{\star}}+(E_{g^{\star}} - \phi)\tau$', fontsize=7)

    ax.text(80, 66, r'$\lambda = C_{g^{\star}}+(E_{g^{\star}} - \phi)\tau$', fontsize=7)

    ax.collections[0].set_alpha(0.2)
    ax.collections[2].set_alpha(0.2)

    fig.savefig('output/price_setters_3.png', dpi=800)
    fig.savefig('output/price_setters_3.pdf', transparent=True)

    return fig, ax


if __name__ == '__main__':
    # rc('text', usetex=True)

    fig, ax = plot_base()
    # demand_plot_1(fig, ax)
    demand_plot_2(fig, ax)



    plt.show()