import numpy as np

from sklearn.neighbors import KDTree
from sklearn.neighbors import DistanceMetric

class RetroScore:

    """
    RETRO-scores: a measure of regression uncertainty based on nearest neighbors.

    """

    def __init__(self, k=5, metric='euclidean'):

        """
        Initialize the RETRO-score method.

        Parameters
        ----------

        k : int
            Number of neighbors to retrieve.

        metric: string
            Distance metric to use. E.g. Euclidean, Manhattan.

        """

        self.k = k
        self.metric = metric
        self.dist = DistanceMetric.get_metric(self.metric)


    def fit(self, X_train, y_train, X_train_unred):

        """
        Saving the data to use for RS method.

        Parameters
        ----------

        X_train : numpy array
            Numpy array containing the training data points.
            Shape (n_points, n_dimensions)

        y_train : numpy array
            Numpy array containing the training labels.
            Shape (n_points, 1)

        """

        # saving the data
        self.X_train = X_train
        self.y_train = y_train

        # for dimensionality-reduced data, we store both an unreduced and
        # reduced version of the data
        self.X_train_unred = X_train_unred

        # setting up KDTree to do neighbor selection
        self.kdtree_X = KDTree(self.X_train, metric=self.metric)
        self.kdtree_y = KDTree(self.y_train, metric=self.metric)


    def filter_errors(self, y_train_pred, discarded_errors=10):

        """
        Filtering points from erroneous regions from training data.

        Parameters
        ----------

        y_train_pred : numpy array
            Predictions the model generated for the training points. To be
            compared with ground-truth training labels.
            Shape identical to self.y_train.

        discarded_errors : int
            Percentage of points that is to be discarded
        """

        # how much error is there in predictions?
        train_error = abs(self.y_train - y_train_pred)

        # define maximum error at percentile
        max_error = np.percentile(train_error, 100-discarded_errors)

        # filter out points with training errors larger than max
        self.X_train = self.X_train[np.where(train_error <= max_error)[0], :]
        self.y_train = self.y_train[np.where(train_error <= max_error)[0], :]

        # filter unreduced data in the same way (if present)
        if self.X_train_unred is not None:
            self.X_train_unred = self.X_train_unred[np.where(train_error <= max_error)[0], :]

        # check if any points remain, to prevent we remove all by accident
        if self.X_train.shape[0] > 0:

            # update KDTrees
            self.kdtree_X = KDTree(self.X_train, metric=self.metric)
            self.kdtree_y = KDTree(self.y_train, metric=self.metric)

        else:
            print("No points remain! Keeping the original data set.")


    def get_score(self, X, y_pred, dimred):

        """
        Calculating the RETRO-score.

        Parameters
        ----------

        X : numpy array
            Points for which we want to calculate the retro score.
            Shape (n_points, n_dimensions)

        y_pred : numpy array
            Predicted value for the points.
            Shape (n_points, 1)

        Returns
        -------

        retro_score : numpy array
            retro scores. Shape (n_points, 1)

        neighbors_x : numpy array
            Feature values of neighbors on which calculation was based.
            Shape (n_points, n_neighbors, n_features)

        neighbors_y : numpy array
            Labels of neighbors on which calculation was based.
            Shape (n_points, n_neighbors)

        """

        # get distance and index of closest kNN neighbors
        dist_closest_X, ind_closest_X = self.kdtree_X.query(X, k=self.k)

        # average distance to the neighbors (in feature values)
        avg_dist_X = np.mean(np.absolute(dist_closest_X),axis=1)

        # initiate arrays to store the feature values of neighbors
        # we need these in case we want to visualize them
        if dimred:
            # keep the not-dimensionality-reduced instances to display
            neighbors_x = np.zeros((X.shape[0], self.k, self.X_train_unred.shape[1]))
        else:
            neighbors_x = np.zeros((X.shape[0], self.k, X.shape[1]))

        # initialize array for target values of neighbors
        neighbors_y = np.zeros((X.shape[0], self.k))

        # for each point and neighbor, find feature values/labels
        for point in range(ind_closest_X.shape[0]):
            for nb in range(ind_closest_X.shape[1]):

                # for dimensionality-reduced instances, we want to store the
                # original unreduced instances
                if dimred:
                    neighbors_y[point,nb] = self.y_train[ind_closest_X[point, nb]]
                    neighbors_x[point,nb] = self.X_train_unred[ind_closest_X[point, nb]]

                # else store instances as they are
                else:
                    neighbors_y[point,nb] = self.y_train[ind_closest_X[point, nb]]
                    neighbors_x[point,nb] = self.X_train[ind_closest_X[point, nb]]

        # find the mean target value of neighbors
        ymean_nbs = np.mean(neighbors_y, axis=1).reshape(X.shape[0],-1)

        # initialize empty array for retro scores
        retro_scores = np.tile(None, (X.shape[0],1))

        # calculate the retro score
        for ix in range(X.shape[0]):
            retro_scores[ix] = self.calculate_score(y_pred[ix], ymean_nbs[ix], avg_dist_X[ix])

        return retro_scores, neighbors_x, neighbors_y


    def calculate_score(self, y_pred, y_nbs_mean, dist_X):

        """
        Calculating the distance of an instance to its neighbors in the y-dimension
        and calculating the RETRO-score.

        Parameters
        ----------

        y_pred : int
            Predicted value for point.

        y_nbs_mean: int
            Mean ground-truth label for neighbors.

        dist_X : int
            Mean distance to neighbors.

        Returns
        -------

        components: numpy array
            RETRO score and its components: weighted distance to neighbors in
            features and in labels. Shape (3, 1)

        """

        # distance between ground-truth label for neighbors and prediction
        dist_y = self.dist.pairwise(y_pred.reshape(-1,1), y_nbs_mean.reshape(-1,1)).flatten()[0]

        # weighting terms for retro score
        dist_y_weight = 1
        dist_X_weight = 1

        # weighting the terms
        weighted_y_dist = dist_y_weight*dist_y
        weighted_x_dist = dist_X_weight*abs(dist_X)

        # calculating the retro score
        retro = -(weighted_y_dist + weighted_x_dist)

        return retro


    def get_score_train(self, X, y_pred):

        """
        Calculating the RETRO-score for the training points (for normalization).

        Parameters
        ----------

        X : numpy array
            Training points.
            Shape (n_points, n_dimensions)

        y_pred : numpy array
            Predicted value for training points.
            Shape (n_points, 1)

        Returns
        -------

        retro_score : numpy array
            retro scores. Shape (n_points, 1)

        """

        # get distance and index of closest kNN neighbors - plus one extra point;
        # the first will be discarded as it is the point itself
        dist_closest_X, ind_closest_X = self.kdtree_X.query(X, k=self.k+1)

        # average distance to the neighbors (in feature values) - skip the first
        # point, as this is the point itself
        avg_dist_X = np.mean(np.absolute(dist_closest_X[:,1:]),axis=1)

        # store the feature values of neighbors
        neighbors_val = np.zeros((X.shape[0], self.k))

        # for each point and neighbor, find feature values
        for point in range(ind_closest_X.shape[0]):

            # skipping the closest neighbor, i.e. the point itself
            for nb in range(0, ind_closest_X.shape[1]-1):
                neighbors_val[point,nb] = self.y_train[ind_closest_X[point, nb+1]]

        # mean label of neighbors
        ymean_nbs = np.mean(neighbors_val, axis=1).reshape(X.shape[0],-1)

        # initialize empty array for retro scores
        retro_scores = np.tile(None, (X.shape[0], 1))

        # calculate the retro score
        for ix in range(X.shape[0]):
            retro_scores[ix] = self.calculate_score(y_pred[ix], ymean_nbs[ix], avg_dist_X[ix])

        return retro_scores


    def normalize(self, rs_train, rs_test):

        """
        Normalize RETRO-score for test set based on scores for train set.

        Parameters
        ----------

        rs_train : numpy array
            RETRO-scores as calculated on train set.

        rs_test : numpy array
            RETRO-scores as calculated on test set.

        Returns
        -------

        retro_score_normalized : numpy array
            Normalized RETRO-score.
        """

        # set up an empty numpy array to fill out with retro score
        retro_score_normalized = np.tile(None, (rs_test.shape[0]))

        # getting the minimum and maximum retro scores on train set
        max_rs = np.max(rs_train)
        min_rs = np.min(rs_train)

        # scale all retro score points individually
        for ix in range(rs_test.shape[0]):
            retro_score_normalized[ix] = self.scale(rs_test[ix], min_rs, max_rs)

        return retro_score_normalized


    def scale(self, p, min_rs, max_rs):

        """
        Scale points to 0-1 range.

        Parameters
        ----------
        p : int
            Number to be scaled.

        min_rs : int
            Minimum of range that must be scaled.

        max_rs : int
            Maximum of range that must be scaled.

        Returns
        -------
        rs : int
            Scaled version of p.
        """

        # do initial scaling
        rs = (p - min_rs) / (max_rs - min_rs)

        # if outside bounds, scale back to bounds
        if rs > 1:
            return 1
        elif rs < 0:
            return 0
        else:
            return rs


def run_retro_score(rs, X_train, y_train, X_test, y_pred, y_train_pred, discarded_errors=10,
                    dimred=False, X_train_unred=None):

    """
    Run the RETRO-score.

    Parameters
    ----------

    rs : RetroScore object
        Initialized RetroScore object.

    X_train : numpy array
        Training points to base the RETRO-score calculation on.

    y_train : numpy array
        Training labels to base the RETRO-score calculation on.

    X_test : numpy array
        Points for which to calculate the retro score.

    y_pred : numpy array
        Predicted labels for X_test points.

    y_train_pred : numpy array
        Predicted labels for X_train points.

    discarded_errors : int
        Proportion of points to be discarded from train set in RETRO-score
        calculation.

    Returns
    -------

    retro_score : numpy array
        The normalized RETRO-scores.

    retro_score_unn : numpy array
        The unnormalized RETRO-scores.

    x_nbs : numpy array
        The x-values of the neighboring instances.

    y_nbs : numpy array
        The y-values of the neighboring instances.

    """

    # reshaping y arrays (in case they're not in right format)
    y_train = y_train.reshape(-1,1)
    y_pred = y_pred.reshape(-1,1)
    y_train_pred = y_train_pred.reshape(-1,1)

    # fitting the method, filtering the training points
    rs.fit(X_train, y_train, X_train_unred)
    rs.filter_errors(y_train_pred, discarded_errors=10)

    # calculating the retro score
    retro_score_unn, x_nbs, y_nbs = rs.get_score(X_test, y_pred, dimred)

    # normalizing the retro score
    retro_score_train = rs.get_score_train(X_train, y_train_pred)
    retro_score = rs.normalize(retro_score_train, retro_score_unn)

    return retro_score, retro_score_unn, x_nbs, y_nbs
