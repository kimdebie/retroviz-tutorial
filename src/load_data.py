import pandas as pd
import numpy as np

from shap import KernelExplainer

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

class DataLoader:

    """
    Class for loading data. Used in the experiments to evaluate RETRO-VIZ.

    Parameters
    ----------

    rand_state : int
        Initialize the numpy random state with an integer (optional).

    """

    def __init__(self, rand_state=42):

        """
        Initializing the DataLoader with most elements empty.

        """

        # data
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # dataset and variable names
        self.name = None
        self.vars_X = None
        self.var_y = None

        # variable scaler
        self.scalerX = None
        self.scalerY = None

        # random state
        self.rand_state = rand_state
        np.random.seed(self.rand_state)

    def scale_min_max(self, min=-1, max=1):

        """
        Scale data so that each train feature sits within min-max range.

        Parameters
        ----------

        min : float
            Minimum value to use for scaling each feature (optional).

        max : float
            Maximum value to use for scaling each feature (optional).

        """

        # creating a scaler, and scaling X variables
        self.scalerX = MinMaxScaler(feature_range=(min, max)).fit(self.X_train)
        self.X_train = self.scalerX.transform(self.X_train)
        self.X_test = self.scalerX.transform(self.X_test)

        try:
            # when y is already a Pandas Series object
            self.scalerY = MinMaxScaler(feature_range=(min, max)).fit(self.y_train.reshape(-1,1))
            self.y_train = self.scalerY.transform(self.y_train.reshape(-1,1)).reshape(-1)
            self.y_test = self.scalerY.transform(self.y_test.reshape(-1,1)).reshape(-1)

        except:
            # when y must be converted to a Pandas Series object
            self.scalerY = MinMaxScaler(feature_range=(min, max)).fit(self.y_train.values.reshape(-1,1))
            self.y_train = pd.Series(self.scalerY.transform(self.y_train.values.reshape(-1,1)).reshape(-1))
            self.y_test = pd.Series(self.scalerY.transform(self.y_test.values.reshape(-1,1)).reshape(-1))

    def unscale(self, X, y):

        """
        Revert scaling of features to the orginal feature values, after they
        have been scaled.

        Parameters
        ----------

        X : numpy array
            Array to scale back using the X scaler.

        y : numpy array
            Array to scale back using the y scaler.

        Returns
        -------

        X_unscaled : numpy array
            Unscaled version of input X.

        y_unscaled : numpy array
            Unscaled version of input y.

        """

        # perform an inverse transform on the input
        X_unscaled = self.scalerX.inverse_transform(X)
        y_unscaled = self.scalerY.inverse_transform(y.reshape(-1,1))

        return X_unscaled, y_unscaled

    def data_shift_automatic(self, fraction_shift=0.3, fraction_points=0.1, sample=True):

        """
        Performing a dataset shift. For a subset of the instances, a subset of
        the features is shifted upwards. The features to shift are selected
        according to their Shapley value, so that the dataset shift will likely
        impact the predictions of the model applied to the shifted data.

        Parameters
        ----------

        fraction_shift : float
            The fraction of variables to shift (optional).

        fraction_points : float
            The fraction of points from the total dataset (so train and test
            combined) to be shifted (optional).

        sample : bool
            Whether a subsample of the points should be used to select the most
            important features in the dataset, to speed up calculations
            (optional).

        """

        # fit a linear regression to the data
        reg = LinearRegression().fit(self.X, self.y)

        # optionally take a sample of the points to speed up calculations
        if sample:
            X = self.X[np.random.randint(self.X.shape[0], size=100), :]
        else:
            X = self.X

        # build a Shapley explainer on the regression and get SHAP values
        explainer = KernelExplainer(reg.predict, X)
        shap_values = explainer.shap_values(X, nsamples=20, l1_reg="aic")

        # determine most important variables by SHAP value on average
        avg_shap = np.average(np.absolute(shap_values),axis=0).flatten()

        # get number of features to shift
        shift_count = int(fraction_shift * self.X.shape[1])

        # get indices of most important features
        shift_fts = avg_shap.argsort()[::-1][:shift_count]

        # new array for new data with shifted features
        shifted_X = np.zeros_like(self.X)

        # number of points to shift
        ix = int((1-fraction_points) * self.X.shape[0])

        # shift feature by feature
        for f_ix in range(self.X.shape[1]):

            # place original feature in matrix
            shifted_X[:,f_ix] = self.X[:,f_ix]

            # check if feature has to be shifted
            if f_ix in shift_fts:

                # get feature from data
                ft = self.X[ix:,f_ix]

                # determine the maximum of this feature
                max_f = np.max(ft)

                # shift feature upward according to a Gaussian distribution
                shifted_X[ix:,f_ix] = ft + np.random.normal(max_f, abs(.1*max_f),
                    shifted_X[ix:,f_ix].shape[0])

        # store the (partially) shifted features back
        self.X = shifted_X

    def randomize_order(self):

        """
        Randomize the order of the data, so that train and test data are from
        the same distribution when test is taken from the bottom of the data.

        """

        # join X and y variables
        joined = np.concatenate((self.X, self.y.reshape(-1,1)), axis=1)

        # shuffle the joint data
        np.random.shuffle(joined)

        # separate the data again
        self.X = joined[:,:-1]
        self.y = joined[:,-1].reshape(-1)

    def split_train_test(self, test_size=0.2, random=True):

        """
        Split data into a train and test set.

        Parameters
        ----------

        test_size : float
            Fraction of points to use as test set (optional).

        random : bool
            Whether the train and test data should be split randomly. If False,
            test data is taken from the last rows of the data.
        """

        # data order is randomized
        if random:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                    self.y, test_size=test_size, random_state=self.rand_state)

        # data order is not randomized (e.g. when last set of points is shifted)
        else:

            # number of points selected into test
            n_test = int((1-test_size) * self.X.shape[0])

            # split train and test
            self.X_train = self.X[:n_test]
            self.X_test = self.X[n_test:]
            self.y_train = self.y[:n_test]
            self.y_test = self.y[n_test:]

    def get_split_data(self):

        """
        Function that returns the split data.

        """

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_vars(self):

        """
        Function that returns variable names.

        """

        try:
            vars_X = self.vars_X.tolist()

        except:
            vars_X = self.vars_X

        var_y = self.var_y

        return vars_X, var_y

    def airfoil(self):

        """
        Loading the Airfoil dataset.

        """

        # loading the data
        airfoil = pd.read_csv('../data/airfoil.dat', sep='\t',header=None,
                     names=['freq', 'angle_attack', 'chord', 'freestream', 'suction', 'pressure'])

        # store data
        self.X = np.array(airfoil.drop(['pressure'], axis=1))
        self.y = np.array(airfoil['pressure'])

        # store names
        self.vars_X = airfoil.drop(['pressure'], axis=1).columns
        self.var_y = 'pressure'
        self.name = 'airfoil'

    def cyclepower(self):

        """
        Loading the Cyclepower dataset.

        """

        # loading the data
        cyclepower = pd.read_csv('../data/cyclepower.csv',sep=';')
        for col in cyclepower.columns:
            cyclepower[col] = cyclepower[col].str.replace(',','.').astype('float')

        # store data
        self.X = np.array(cyclepower.drop(['PE'], axis=1))
        self.y = np.array(cyclepower['PE'])

        # store names
        self.vars_X = cyclepower.drop(['PE'], axis=1).columns
        self.var_y = 'PE'
        self.name = 'cyclepower'

    def diabetes(self):

        """
        Loading the Diabetes dataset.

        """

        # loading the data
        diabetes = datasets.load_diabetes()

        # store data
        self.X = diabetes.data
        self.y = diabetes.target

        # store names
        self.vars_X = diabetes.feature_names
        self.var_y = 'disease_progression'
        self.name = 'diabetes'

    def boston(self):

        """
        Loading the Boston dataset.

        """

        # loading the data
        boston = datasets.load_boston()

        # store data
        self.X = boston.data
        self.y = boston.target

        # store names
        self.vars_X = boston.feature_names
        self.var_y = 'house_price'
        self.name = 'boston'

    def california_housing(self):

        """
        Loading the California dataset.

        """

        # loading the data
        california = datasets.fetch_california_housing()

        # store data
        self.X = california.data
        self.y = california.target

        # store names
        self.vars_X = california.feature_names
        self.var_y = 'house_value'
        self.name = 'california_housing'

    def winequality2(self):

        """
        Loading the Winequality dataset. This version is used to test model
        overfit and underfit.
        """

        # loading the data
        winequality = pd.read_csv('../data/winequality/winequality-white.csv', sep=';')

        # store data
        self.X = np.array(winequality.drop(['quality'], axis=1))
        self.y = np.array(winequality['quality'])

        # store names
        self.vars_X = winequality.drop(['quality'], axis=1).columns
        self.var_y = 'quality'
        self.name = 'winequality_white'

    def winequality(self):

        """
        Loading the Winequality dataset. This version of the dataset is used to
        test distributional shift.
        """

        # loading the data
        white = pd.read_csv('../data/winequality-white.csv', sep=';')
        red = pd.read_csv('../data/winequality-red.csv', sep=';')

        # storing red and white wine separately
        X_white = white.drop(['quality'], axis=1)
        X_red = red.drop(['quality'], axis=1)

        y_white = white['quality']
        y_red = red['quality']

        # joining red and white data
        self.X = np.array(pd.concat([X_white, X_red]))
        self.y = np.array(pd.concat([y_white, y_red]))

        # create an array with the color of each instance
        self.color = np.concatenate( ( np.ones((y_white.shape[0])), np.zeros((y_red.shape[0])) ) )

        # store names
        self.vars_X = white.drop(['quality'], axis=1).columns
        self.var_y = 'quality'
        self.name = 'wine'

    def winequality_distshift(self, color=1):

        """
        Splitting Winequality dataset into train and test with distributional
        shift.
        """

        self.X_train = np.array(self.X[self.color==color])
        self.X_test = np.array(self.X[self.color!=color])
        self.y_train = np.array(self.y[self.color==color])
        self.y_test = np.array(self.y[self.color!=color])

    def autompg(self):

        """
        Loading the Autompg dataset. This version of the dataset is used to
        test distributional shift.
        """

        # loading the data
        autompg_full = pd.read_csv('../data/auto-mpg.data', sep='\t', names=['a', 'carnames'])
        autompg = autompg_full.a.str.split(expand=True)
        autompg.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                            'acceleration', 'model year', 'origin']

        # converting data to float
        for col in autompg.columns:
            autompg[col] = autompg[col].astype('float')

        # store data
        self.X = autompg.drop(['mpg', 'origin'], axis=1)
        self.y = autompg['mpg']
        self.origin = autompg['origin']

        # store names
        self.name = 'autompg'
        self.vars_X = autompg.drop(['mpg', 'origin'], axis=1).columns
        self.var_y = 'mpg'

    def autompg_distshift(self, shift=1):

        """
        Splitting Winequality dataset into train and test with distributional
        shift.

        Parameters
        ----------

        shift : int
            Which city to use as test set city.

        """

        self.X_train = np.array(self.X[self.origin != shift])
        self.X_test = np.array(self.X[self.origin == shift])
        self.y_train = np.array(self.y[self.origin != shift])
        self.y_test = np.array(self.y[self.origin == shift])

    def autompg2(self):

        """
        Loading the Autompg dataset. This version of the dataset is used to
        test model over- and underfit.

        """

        # loading the data
        autompg_full = pd.read_csv('../data/auto-mpg.data', sep='\t', names=['a', 'carnames'])
        autompg = autompg_full.a.str.split(expand=True)
        autompg.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                            'acceleration', 'model year', 'origin']

        # converting data to float
        for col in autompg.columns:
            autompg[col] = autompg[col].astype('float')

        # store data
        self.X = np.array(autompg.drop(['mpg'], axis=1))
        self.y = np.array(autompg['mpg'])

        # store names
        self.name = 'autompg'
        self.vars_X = autompg.drop(['mpg'], axis=1).columns
        self.var_y = 'mpg'

    def abalone(self):

        """
        Loading the Abalone dataset.

        """

        # loading the data
        abalone = pd.read_csv('../data/abalone.data', names=['sex',
        'length', 'diameter', 'height', 'wholeweight', 'shuckedweight',
        'visceraweight', 'shellweight', 'rings'])

        # store data
        self.X = np.array(abalone.drop(['rings', 'sex'], axis=1))
        self.y = np.array(abalone['rings'])

        # store names
        self.name = 'abalone'
        self.vars_X = abalone.drop(['rings', 'sex'], axis=1).columns
        self.var_y = 'rings'

    def toxicfish(self):

        """
        Loading the Fish toxicity dataset.

        """

        # loading the data
        fish = pd.read_csv('../data/toxicfish.csv', names=['CIC0',
        'SMI1_Dz', 'GATS1i', 'NdsCH', 'NdssC', 'MLOGP', 'LC50'], sep=';')

        # store data
        self.X = np.array(fish.drop(['LC50'], axis=1))
        self.y = np.array(fish['LC50'])

        # store names
        self.name = 'toxicfish'
        self.vars_X = fish.drop(['LC50'], axis=1).columns
        self.var_y = 'LC50'

    def energyefficiency(self):

        """
        Loading the Energy efficiency dataset.

        """

        # loading the data
        energy = pd.read_csv('../data/energyefficiency.csv')

        # store data
        self.X = np.array(energy.drop(['Y1', 'Y2'], axis=1))
        self.y = np.array(energy['Y2'])

        # store names
        self.name = 'energyefficiency'
        self.vars_X = energy.drop(['Y1', 'Y2'], axis=1).columns
        self.var_y = 'Y2'

    def communities(self):

        """
        Loading the Communities dataset.

        """

        # loading the data
        community = pd.read_csv('../data/communities.data', header=None, na_values="?")

        # store data
        self.X = np.array(community.drop([0,1,2,3,4,127], axis=1).dropna(axis=1))
        self.y = np.array(community[127])

        # store names
        self.name = 'communities'
        self.vars_X = list(range(self.X.shape[1]))
        self.var_y = 'predicted_crime'

    def superconductor(self):

        """
        Loading the Superconductor dataset.

        """

        # loading the data
        super = pd.read_csv('../data/superconductor.csv')

        # store data
        self.X = np.array(super.drop(['critical_temp'], axis=1))
        self.y = np.array(super['critical_temp'])

        # store names
        self.name = 'superconductor'
        self.vars_X = super.drop(['critical_temp'], axis=1).columns
        self.var_y = 'critical_temp'
