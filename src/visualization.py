import random
import pickle
import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.neural_network import MLPRegressor
from copy import deepcopy

import sys
sys.path.append('../')

def sample_new_point(X_test, y_test, i=None):

    """
    Sample a random point from the test data.

    Parameters
    ----------

    X_test : numpy object
        Test data.

    y_test : numpy object
        Test labels.

    i : int
        Index of point to sample (optional).

    Returns
    -------

    X_sample : numpy object
        Sampled input variables.

    y_test : numpy object
        Sampled label.
    """

    # select a random point or given
    if i is not None:
        sample = i
    else:
        sample = random.randrange(X_test.shape[0])

    # take point from dataset
    X_sample = X_test[sample, :].reshape(1,-1)
    y_sample = y_test[sample].reshape(-1)

    return X_sample, y_sample


def get_range(scaler, test_unscaled):

    """
    Obtain minimum and maximum value of data to use on axis.

    Parameters
    ----------

    scaler : sklearn scaler
        From the scaler, we obtain the min/max on the train set.

    test_unscaled : numpy array
        For the min/max on the test set.

    Returns
    -------

    minimum : float
        Minimum value across the train and test set.

    maximum : float
        Maximum value across the train and test set.

    """

    # get minimum and maximum

    min_test = np.amin(test_unscaled, axis=0)
    max_test = np.amax(test_unscaled, axis=0)

    minimum = np.minimum(scaler.data_min_, min_test)
    maximum = np.maximum(scaler.data_max_, max_test)

    return minimum, maximum


def make_dict_df(X1, y1, X2, y2, X_vars, y_var, concat_data=True):

    """
    Turn data into dataframe, then dictionary, to be passed to D3.
    Either concatenating points and neighboring points, or minimum
    and maximum of range.

    Parameters
    ----------

    X1 : numpy array
        First array. Could be input point or minimum of x range.

    y1 : numpy array
        First array. Could be input prediction or minimum of y range.

    X2 : numpy array
        Second array. Could be neighboring points or maximum of x range.

    y2 : numpy array
        Second array. Could be neighboring labels or maximum of y range.

    X_vars : list
        Names of input variables.

    y_var : string
        Name of output variable.

    type : string
        Whether to concatenate data points or range extremas.

    Returns
    -------

    final_data : dict
        Dictionary of all datapoints.

    """

    # place all data together
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    all_data = np.concatenate((X, y), axis=1)

    # all variable names together
    varnames = X_vars + [y_var]
    varnames = [str(v) for v in varnames]

    # place in dataframe
    df = pd.DataFrame(all_data, columns=varnames)

    # indicate which point is the new one and which are neighbors
    if concat_data:
        point_type = ["neighbor"] * X.shape[0]
        point_type[0] = "new_point"
        df["point_type"] = point_type

    final_data = df.to_dict('records')

    return final_data


def sort_axes(data, axes):

    """
    Sort axes according to similarity. Method from Massie (2004).

    Parameters
    ----------

    data : numpy array
        Set of points used for sorting.

    axes : list
        Axis names which are to be sorted.

    Returns
    -------

    sorted_axes : list
        Sorted version of axes.
        
    """

    axes_new = deepcopy(axes)

    sorted_axes = []

    # place all data in a dataframe
    df = pd.DataFrame(data, columns=axes_new)

    # get all combinations variables (all possible pairs)
    axis_pairs = combinations(axes_new, 2)

    # set up dictionary to keep similarities
    similarities = {}

    # for all pairs, get similarity: inverse Euclidean distance
    for pair in axis_pairs:
        similarities[pair] = -np.linalg.norm(df[pair[0]].values - df[pair[1]].values)

    # most similar pair has largest similarity score
    most_similar_pair = max(similarities.keys(), key=(lambda k: similarities[k]))
    sorted_axes.append(most_similar_pair[0])
    sorted_axes.append(most_similar_pair[1])

    # sorted items may be removed from axes
    axes_new.remove(most_similar_pair[0])
    axes_new.remove(most_similar_pair[1])

    # latest item in list
    latest_in_list = most_similar_pair[1]

    # now, for the latest entry, we want to find the most similar axis iteratively
    while len(axes_new) > 0:

        # create similarities anew
        similarities = {}
        for ax in axes_new:
            similarities[ax] = -np.linalg.norm(df[latest_in_list].values - df[ax].values)

        # most similar pair is added to sorted list and removed from unsorted
        most_similar_ax = max(similarities.keys(), key=(lambda k: similarities[k]))
        sorted_axes.append(most_similar_ax)
        axes_new.remove(most_similar_ax)
        latest_in_list = most_similar_ax

    return sorted_axes
