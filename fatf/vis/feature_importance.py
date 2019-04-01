"""
Functions for calculating feature importance and 
Individual Conditional Expectation (ICE)
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# License: new BSD

import numpy as np 

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_individual_conditional_expectiation(
                                            ice: np.ndarray,
                                            feature_name: str,
                                            values: np.ndarray,
                                            category: int,
                                            category_name: str = ''
                                            ) -> plt.Figure:
    """
    Plots individual conditional expectations for class

    Parameters
    ----------
    ice: np.ndarray 
        Shape [n_samples, steps, n_classes] containing probabilities
        outputted from ICE
    feature_name: str 
        Specificy which feature was used to calculate the ICE
    values: np.array 
        Containing values of feature tested for ICE
    category: int 
        Which class to plot probabilities for
    category_name: str 
        Name of class chosen. If None then the category_name will be 
        the category integer converted to str. Default: None
    
    Returns
    -------
    ax: plt.Figure
        Figure with individual conditional expectation and partial dependence line
        plotted
    """
    if plt is None:
        raise ImportError('plot_ICE function requires matplotlib package. This can' 
                          'be installed with "pip install matplotlib"')
    if not category_name:
        category_name = str(category)
    ax = plt.subplot(111)
    lines = np.zeros((ice.shape[0], ice.shape[1], 2))
    lines[:, :, 1] = ice[:, :, category]
    lines[:, :, 0] = np.tile(values, (ice.shape[0], 1))
    collect = LineCollection(lines, label='Individual Points', color='black')
    ax.add_collection(collect)
    mean = np.mean(ice[:, :, category], axis=0)
    ax.plot(values, mean, color='yellow', linewidth=10, alpha=0.6, label='Mean')
    ax.legend()
    ax.set_ylabel('Probability of belonging to class %s'%category_name)
    ax.set_xlabel(feature_name)
    ax.set_title('Individual Conditional Expecatation')
    return ax

def plot_partial_dependence()