#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Python software for IEMS 351 at Northwestern University

Copyright 2022 by Andreas Waechter & Gokce Kahvecioglu

This module provides a number of function and classes to the students
that they can use to write their own code.  For example, it includes code
for some of the test objective functions, plotting functions, etc
'''

# Import NumPy for linear algebra computations
import numpy as np

# Import modules for plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# The following suppresses an annoying warning
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


# The following provides the MNIST data set
import tensorflow as tf


class OptProblem:
    '''
    Base class for an unconstrained optimzation problem.

    An instance of this class defines an unconstrained optimization problem.
    The class must have methods for evaluating the value of the objective
    function, its gradient, and a starting point.

    An optimization method also needs a starting point.  It is usually part
    of the optimization problem, so an instance of this class should also
    provide a default starting point.  In this way, all information needed to
    solve an optimization problem is available from this class.  The starting
    point can also be used to find out the dimension of the optimization
    variable.

    In the following, n denotes the number of optimization variables
    '''

    def __init__(self):
        # nothing to be done in base blass constructor
        pass

    def value(self, x):
        '''
        Compute value of the objective function at point x

        Input arguments:
            x: n-dim array with vector of input variables

        Return values:
            val: scalar value of objective function at x
        '''
        raise Exception('Method "value" is not implemented')

    def gradient(self, x):
        '''
        Compute gradient of objective function at point x

        Input arguments:
            x: n-dim array with vector of input variables

        Return values:
            grad: n-dim array with gradient of objective function at x
        '''
        raise Exception('Method "gradient" is not implemented')

    def hessian(self, x):
        '''
        Compute Hessian of objective function at point x

        Input arguments:
            x: n-dim array with vector of input variables

        Return values:
            Hess: n-by-n-dim array with Hessian of objective function at x
        '''
        raise Exception('Method "hessian" not is implemented')

    def starting_point(self):
        '''
        Provide a default starting point

        Return value:
            x_start: n-dim array with default starting point
        '''
        raise Exception('method "starting_point" not implemented')


class DataSet:
    '''
    Base class for data sets.

    An object in this class stores the training and test data, which can
    be accessed with get_train and get_test methods.
    '''

    def __init__(self, Z_train, y_train, Z_test, y_test):
        '''
        Store the provided training and test data

        Input arguments:
            Z_train: features of training data
            y_train: labels of training data
            Z_test: features of test data
            y_test: labels of test data
        '''

        self._Z_train = Z_train
        self._y_train = y_train

        self._Z_test = Z_test
        self._y_test = y_test

    def get_train(self):
        '''
        provide the training data.

        Return values:
            Z_train: features of training data
            y_train: lables of training data
        '''
        return (self._Z_train, self._y_train)

    def get_test(self):
        '''
        provide the test data.

        Return values:
            Z_test: features of test data
            y_test: lables of test data
        '''
        return (self._Z_test, self._y_test)


class MnistBinaryDataSet(DataSet):
    '''
    Data set for binary classification constructed from two digits of the
    MNIST data set
    '''

    def __init__(self, digit_1, digit_2):
        '''
        Create the data set from MNIST by filtering out two digits

        Input arguments:
            digit_1: Digit for class 1
            digit_2: Digit for class 2
        '''

        # Load the data from Keras
        mnist = tf.keras.datasets.mnist
        (Z_train, y_train), (Z_test, y_test) = mnist.load_data()
        Z_train = Z_train.reshape((Z_train.shape[0], 784)) / 255.
        Z_test = Z_test.reshape((Z_test.shape[0], 784)) / 255.

        # Add bias feature
        N = Z_train.shape[0]
        ones = np.ones((N, 1))
        Z_train = np.hstack((Z_train, ones))
        N = Z_test.shape[0]
        ones = np.ones((N, 1))
        Z_test = np.hstack((Z_test, ones))
        # Filter out the two digits
        Z = []
        y = []

        for digit in [digit_1, digit_2]:
            indices = np.where(y_train == digit)[0]
            Z.append(Z_train[indices])
            y.append(y_train[indices]*0 + (1 if digit == digit_1 else -1))
        self._Z_train = np.concatenate(Z)
        self._y_train = np.concatenate(y)

        Z = []
        y = []
        for digit in [digit_1, digit_2]:
            indices = np.where(y_test == digit)[0]
            Z.append(Z_test[indices])
            y.append(y_test[indices]*0 + (1 if digit == digit_1 else -1))
        self._Z_test = np.concatenate(Z)
        self._y_test = np.concatenate(y)

        print('Creating dataset for two digits from MNIST:')
        print('Label  1 is for digit %d' % digit_1)
        print('Label -1 is for digit %d\n' % digit_2)

    def show_digit(self, z):
        '''
        Display digit given by the pixel in feature vector z
        '''
        pixels = 255.*(1. - z[:-1].reshape((28, 28)))
        pixels = np.array(pixels, dtype='uint8')
        plt.imshow(pixels, cmap='gray')
        plt.show()


class CsvDataSet(DataSet):
    '''
    Data set that is read from a csv file.

    When created, this object reads data from a csv file with the provided
    name.  The data is randomly split into training and test data.
    '''

    def __init__(self, filename, frac_test=0.25, rand_seed=0,
                 add_dummy_feature=True):
        '''
        Read the data from a csv file and split the data randomly into training
        and test data.

        Input arguments:
            filename: name of csv file
            frac_test: Fraction of data that should be used as test data.  The
                data points read in will be randomly split into two parts, the
                training and the test data.  The random number generator for
                the random split is initialized with rand_seed
            rand_seed: Random seed for splitting of data
            add_dummy_feature: If True, a new column will be added that has all
                ones in it.
        '''

        # Read the first two lines to extract which of the columns are features
        # and which are lables
        fh = open(filename, 'r')

        # Read first line with lables and output them
        line = fh.readline()

        # Now read next line which has flags that indicate if the column is a
        # features (F), a label (L), should be ignored (*)
        line = fh.readline()
        # remove newline and double quote characters
        line = line.strip('"\n')
        # close the csv file again
        fh.close()

        # Split the line into the individual flag headers
        header_info = line.split(',')
        # extract the indices of all columns that do not have a * flag
        col_numbers = \
            [i for i in range(len(header_info)) if header_info[i] != '*']
        # remove the ignored columns from header_info
        header_info = [header_info[i] for i in col_numbers]
        # determine column indices of features
        feature_col = \
            [i for i in range(len(header_info)) if header_info[i] == 'F']
        # Throw an error if there are more than one columns marked as labels
        if header_info.count('L') != 1:
            raise('There should be exactly one column with label L ' +
                  'in csv file')
        # determine column indices of labels
        label_col = header_info.index('L')

        # Now read the CVS file in which the entries are separated by commas.
        # Skip the first two line since they were header without data.  Only
        # read the columns that are not marked with '*'
        dataset = \
            np.genfromtxt(filename, delimiter=',',
                          skip_header=2, usecols=tuple(col_numbers))

        if add_dummy_feature is True:
            # add intercept (constant 1) as feature to the dataset
            dataset = np.append(dataset, np.ones((len(dataset), 1)), 1)
            feature_col.append((dataset.shape[1]-1))  # column number

        # store the current state of the random number generator to restore it
        # later.  This way this does not affect other functions that also rely
        # on random numbers.
        randState = np.random.get_state()
        # Randomly reshuffle the data points with the given random number
        # generator seed
        np.random.seed(rand_seed)
        np.random.shuffle(dataset)
        # Restore state of random number generator
        np.random.set_state(randState)

        # first determine the number of data points
        num_data, _ = dataset.shape
        # Fraction of data used as training data
        frac_train = 1.0 - frac_test
        # Number of training points
        num_train = int(num_data*frac_train)

        # extract the training data
        self._Z_train = dataset[:num_train, feature_col]
        self._y_train = dataset[:num_train, label_col]
        # extract the test data
        self._Z_test = dataset[num_train:, feature_col]
        self._y_test = dataset[num_train:, label_col]

        # Remove rows with NaN (not a number) in any of the features.  We
        # assume that we do not have any NaN in the lables
        nan_indicator = ~np.isnan(self._Z_train).any(axis=1)
        self._Z_train = self._Z_train[nan_indicator]
        self._y_train = self._y_train[nan_indicator]
        nan_indicator = ~np.isnan(self._Z_test).any(axis=1)
        self._Z_test = self._Z_test[nan_indicator]
        self._y_test = self._y_test[nan_indicator]


class LogisticRegressionProblem(OptProblem):
    '''
    Optimization problem object for logistic regression with data read from a
    csv file.
    '''

    def __init__(self, dataSet, regularization=0.):
        '''
        Initialize the object with the provided data set.

        Inputs:
            dataSet:
                DataSet objective with training and test data
            regularization:
                regularization parameter.  (is divided by num_feat)
        '''

        # Get the training and test data from the provided data set object
        (Z_train, y_train) = dataSet.get_train()
        (Z_test, y_test) = dataSet.get_test()

        # throw an error if labels are not -1, +1
        y = np.concatenate((y_train, y_test), axis=0)
        if len([i for i in range(len(y))
                if y[i] == -1 or y[i] == 1]) != len(y):
            raise Exception('Labels are not in {-1,1}')

        self._N_train = Z_train.shape[0]
        self._N_test = Z_test.shape[0]
        self._num_feat = Z_train.shape[1]

        self._Z_train = Z_train
        self._y_train = y_train

        self._Z_test = Z_test
        self._y_test = y_test

        self._regu = regularization

        print('LogisticRegressionObj has been created.')
        print('Number of features........: %d' % self._num_feat)
        print('Size of training set......: %d' % self._N_train)
        print('Size of test set..........: %d' % self._N_test)
        print('Regularization parameter..: %f' % self._regu)
        print('')

    def value(self, x):
        # scale each row of Z by corresponding entry in y
        Zx = self._Z_train.dot(x)
        yZx = self._y_train * Zx
        exp_neg_yZx = np.exp(-1.0*yZx)
        value_vector = np.log(1.0 + exp_neg_yZx)
        val = np.sum(value_vector) / self._N_train
        if self._regu > 0.:
            # Do not include the bias term in the regularization
            val += self._regu * np.linalg.norm(x[:-1])**2
        return val

    def gradient(self, x):
        # scale each row of Z by corresponding entry in y
        Zx = self._Z_train.dot(x)
        yZx = self._y_train * Zx
        exp_pos_yZx = np.exp(yZx)
        diag = -1./(1.+exp_pos_yZx)/self._N_train
        grad = self._Z_train.T.dot(diag*self._y_train)
        if self._regu > 0.:
            # Do not include the bias term in the regularization
            grad[:-1] += 2.*self._regu * x[:-1]
        return grad

    def hessian(self, x):
        Zx = self._Z_train.dot(x)
        yZx = self._y_train * Zx
        exp_pos_yZx = np.exp(yZx)
        diag = exp_pos_yZx/((1.+exp_pos_yZx)**2)/self._N_train
        diag = diag.reshape(self._N_train, 1)
        hess = self._Z_train.T.dot(diag*self._Z_train)
        if self._regu > 0.:
            hess += 2.*self._regu * np.identity(self._num_feat)
            hess[-1:-1] -= 2.*self._regu
        return hess

    def starting_point(self):
        # For logistic regression, zero is a good starting point.
        dim = self._Z_train.shape[1]
        x_start = np.zeros(dim)
        return x_start

    def compute_accuracy_train(self, x):
        '''
        returns accuracy of current iterator x using training set
        accuracy: (# of correct classification) / (# of data points)
        '''
        classifier = np.sign(np.dot(self._Z_train, x))  # y assumed in {-1,1}
        accuracy_rate = np.mean(classifier == self._y_train)
        return accuracy_rate

    def compute_accuracy_test(self, x):
        '''
        returns accuracy of current iterator x using test set
        accuracy: (# of correct classification) / (# of data points)
        '''
        classifier = np.sign(np.dot(self._Z_test, x))  # y assumed in {-1,1}
        accuracy_rate = np.mean(classifier == self._y_test)
        return accuracy_rate

    def misclassified_test(self, x):
        '''
        returns the data points with labels that were misclassified in the
        test set
        '''
        classifier = np.sign(np.dot(self._Z_test, x))
        indices = (classifier != self._y_test)
        return (self._Z_test[indices], self._y_test[indices])


class NonlinearRegressionModel:
    '''
    Base class for nonlinear regression models.

    A nonlinear regression model m(z;x) is a mathematical formula that tries to
    predict the response of a process based on values for predictor variables
    z.  The model is parameterized by some parameters x that need to be
    adjusted by regression to minimize the error in the prediction.

    To find the optimal model parameters in the regression, the optimization
    method needs the values of the model as well as the partial derivatives
    with respect to the model parameters x.

    The methods in this class compute the quantities for all data points (Z)
    at once.  In the notation below, N denotes the number of data points,
    d denotes the number of predictor variables, and n is the number of model
    parameters.
    '''

    def __init__(self):
        # nothing to be done in the base class constructor
        pass

    def values(self, Z, x):
        '''
        Compute the model values with a given set of parameters x for all
        data points in Z.

        Input arguments:
            Z: N-by-d-dimensional array with the predictor variable values.
                Each row of Z corresponds to one data point
            x: n-dimensional array with the model parameters
        Return value:
            vals: N-dimensional array with the values of the model (evaluations
                of the formula) for each data point.
        '''
        raise Exception('Method "values" not implemented')

    def derivatives(self, Z, x):
        '''
        Compute the partial derivatives of the model with respect to the
        parameters for all data points in Z.

        Input arguments:
            Z: N-by-d-dimensional array with the predictor variable values.
                Each row of Z corresponds to one data point
            x: n-dimensional array with the model parameters
        Return value:
            M: N-by-n-dimensional array with the partial derivatives of the
                model with respect to all data points.  Each row corresponds
                to a data point, and in each row is the gradient of the model
                with respect to x.
        '''
        raise Exception('Method "derivatives" not implemented')

    def starting_point(self):
        '''
        Return (default) starting point; i.e., typical values for the model
        parameters that might be a good guess of the optimal values.
        '''
        raise Exception('Method "starting_point" not implemented')


class BellCurveRegressionModel(NonlinearRegressionModel):
    '''
    Subclass of the NonlinearRegressionModel for fitting bell curve to
    data.  We use this as an example also to get a nonconvex problem.

    The model has the form

    m(z;x1,x2) = exp(-(z-x1)^2/x2)
    '''

    def __init__(self, Z):
        '''
        Do a sanity check for the data: is the number of predictor
        variables correct?
        '''
        N, d = Z.shape
        if d != 1:
            raise Exception('Number of predictor variables is not 1')

    def values(self, Z, x):
        '''
        Compute the model values with a given set of parameters x for all
        data points in Z.
        '''

        # For convenience, get the values of the parameters out of x
        x1 = x[0]
        x2 = x[1]

        # Need to get Z into 1-dim array
        N, _ = Z.shape
        Zcol = Z.reshape(N)

        # Evaluate the formula (for all data points at once)
        vals = np.exp(-np.square(Zcol-x1)/x2)
        return vals

    def derivatives(self, Z, x):
        '''
        Compute the partial derivatives of the model with respect to the
        parameters for all data points in Z.
        '''

        # For convenience, get the values of the parameters out of x
        x1 = x[0]
        x2 = x[1]

        # Compute the individual partial derivatives (for all data points at
        # once).
        exp_term = np.exp(-np.square(Z-x1)/x2)
        d1 = 2*(Z-x1)/x2*exp_term
        d2 = np.square(Z-x1)/np.square(x2)*exp_term

        # Concatenate the (column) vectors of the individual partial
        # derivatives to a matrix
        derivs = np.hstack((d1, d2))
        return derivs

    def starting_point(self):
        '''
        typical values for the parameters
        '''
        x_start = np.array([1., 1.])
        return x_start


class NonlinearRegressionProblem(OptProblem):
    '''
    Optimization problem for nonlinear regression.

    This objective function is for the computation of model parameters in a
    nonlinear regression model.  It is computed by summing up the squared
    prediction errors over all data points.

    To define the nonlinear regression objective function, we need a nonlinear
    regression model (NonlinearRegressionModel) and training data.

    In the following, N denotes the size of the training set, d denotes the
    number of predictor variables, and n denotes the parameters in the
    nonlinear model.

    Members in this class:
        Z: N-by-d dimensional array with the values of predictor variables in
            the training set
        y: N-dimensional array with the values of the response variable in the
            training set
        nonlinModel: Object of class NonlinearRegressionModel
    '''

    def __init__(self, nonlinModel, Z, y):
        '''
        Store training data and nonlinear model object
        '''
        self._nonlinModel = nonlinModel
        self._Z = Z
        self._y = y

    def value(self, x):
        '''
        Compute value of the objective function

        sum_{i=1}^N (m(z^i,x)-y^i)^2
        '''

        # Compute the values predicted for the current parameters x for all
        # training points
        model_vals = self._nonlinModel.values(self._Z, x)

        # compute the error (residual) as the difference between the model
        r = model_vals - self._y

        # square the residual and then sum up over all data points
        r = r**2
        val = np.sum(r)
        return val

    def gradient(self, x):
        '''
        Compute gradient of the objective function

        2*M(x)^T*r(x)
        '''
        # Compute the values predicted for the current parameters x for all
        # training points
        model_vals = self._nonlinModel.values(self._Z, x)

        # compute the error (residual) as the difference between the model
        r = model_vals - self._y

        # compute the matrix of partial derivatives
        M = self._nonlinModel.derivatives(self._Z, x)

        # compute the gradient
        grad = 2. * M.T.dot(r)
        return grad

    def hessian(self, x):
        '''
        Compute Hessian (here Gauss-Newton approximation) of the objective
        function.

        M(x)^T M(x)
        '''
        # compute the matrix of partial derivatives
        M = self._nonlinModel.derivatives(self._Z, x)

        # compute the gradient
        hessian = M.T.dot(M)
        return hessian

    def starting_point(self):
        '''
        Return starting point provided by the model
        '''
        return self._nonlinModel.starting_point()


def plot_bellcurve_model(filename, x1=None, x2=None):
    '''
    Plot data points for bell curve regression problem together with
    the actual curve.

    Input arguments:
        filename: csv data file name
        x1, x2: parameters in the model
            If set to None, only the data points are plotted
    '''
    dataset = CsvDataSet(filename, frac_test=0., add_dummy_feature=False)
    (z, y) = dataset.get_train()
    plt.close()
    plt.plot(z, y, 'o')
    plt.xlabel('$z$')
    plt.ylabel('$g(z)$')
    #
    if x1 is not None:
        z_min = np.min(z)
        z_max = np.max(z)
        Z = np.arange(z_min, z_max, (z_max-z_min)/100)
        M = np.exp(-np.square(Z-x1)/x2)
        plt.plot(Z, M)

    plt.show()
