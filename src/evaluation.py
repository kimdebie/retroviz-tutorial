import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def rs_at_threshold_plot(rs, y_test, y_pred, normalized=True):

    """
    Creating a plot showing the average RETRO score at different
    thresholds of error. Also calculate the Pearson correlation between
    the RETRO score and the error.

    Parameters
    ----------

    rs : numpy array
        RETRO-scores.

    y_test : numpy array
        Target variable from the test set.

    y_pred : numpy array
        Predicted target variable for the test set.

    normalized : bool
        Whether the RETRO-scores provided are normalized (optional).


    Returns
    -------

    plt : matplotlib object
        The generated plot.

    p_corr : float
        The Pearson correlation coefficient between the RETRO-scores and the
        absolute errors.

    """

    # calculate absolute error for each point
    error = abs(y_test.reshape(-1, 1)-y_pred.reshape(-1,1))

    # calculate pearson correlation for error and RETRO score
    p_corr, pvalue = pearsonr(list(error.flatten()), list(rs))

    # set error thresholds: binning
    bins = np.histogram(error, bins=10)[1].astype('float')
    binned = np.digitize(error, bins)

    # we want the maximum error in a bin as label of the bin
    binlabel = np.zeros((binned.shape[0])).astype('float')
    for ix in range(binlabel.shape[0]):
        binlabel[ix] = round(bins[binned[ix]-1][0], 3)

    # find average error in each bin
    df = pd.DataFrame({"rs": rs.astype('float').flatten(),"bin": binlabel.flatten(), "error": error.flatten()})
    mean = df.groupby("bin").mean()["rs"].reset_index()
    std = df.groupby("bin").std()["rs"].reset_index().fillna(1e-6)

    # plotting
    plt.plot(mean.bin, mean.ts, '-o')
    plt.fill_between(mean.bin, mean.ts-std.ts, mean.ts+std.ts, alpha=.1)
    plt.xlabel("Maximum error")
    plt.ylabel("Average trust score")
    plt.title(f"Maximum error vs. average RETRO-score (corr: {round(p_corr,3)})")

    # if points are normalized, show appropriate y-axis
    if normalized:
        plt.ylim(-0.1,1.1)

    return plt, p_corr

def overlapping_points(rs, errors, frac=50):

    """
    Find the share of points that overlaps between the lowest RETRO-scores and
    the largest errors.

    Parameters
    ----------

    rs : numpy array
        RETRO-scores.

    errors : numpy array
        Errors.

    frac : float
        For which fraction of points we want to calculate the overlap (i.e.
        bottom fraction of RETRO scores and top fraction of errors).

    Returns
    -------

    overlap_frac : float
        Fraction of points that overlaps.

    """

    # index indicates order of array when sorted
    error_ix = errors.argsort()
    rs_ix = rs.argsort()[::-1]

    # index up to/from which the points should be selected
    mid = int(error_ix.shape[0]-error_ix.shape[0]/100*frac)

    # how many of the points overlap?
    overlap = np.intersect1d(error_ix[mid:], rs_ix[mid:]).shape

    # calculating the fraction of points that overlaps
    overlap_frac = overlap[0] / (error_ix.shape[0]-mid)

    return overlap_frac
