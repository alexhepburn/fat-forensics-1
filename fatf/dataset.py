"""
Dataset class 
"""

# Author: Alex Hepburn <ah13558@bristol.ac.uk>
# Licence: new BSD


import numpy as np
import csv


class Dataset():
    '''
    Dataset class for storing values, target variable and attribute names.

    Dataset class to use in classification or regression algorithms with the FATF package.
    The data, header and target are stored in separate attributes. Currently the only type of 
    data supported are floats and categorical data is assumed to be transformed into a one-hot
    encoding.

    Attributes:
        X: numpy.ndarray of attributes of shape [number of data points, number of features]
        target: list of target variables corresponding to the data points in X
        header: list of strings that are attribute names for features in X. Must be the same
            length as X[0, :]
        class_names: list of strings that correspond to the target names
    '''

    def __init__(self, X, target, header, class_names):
        self._X = X
        self._target = target
        self._header =  header
        self._class_names = class_names
        if len(self._header) != self._X.shape[1]:
            raise ValueError('Length of header must be the same as number of features in X')
        if len(self._target) != self._X.shape[0]:
            raise ValueError('Number of targets given must be equal to the number of data points in X')
        if len(self._class_names) != len(set(self._target)):
            raise ValueError('Number of class names given must be equal to number of classes defined')

    @classmethod
    def from_csv(self, file, sep=",", header='infer', target='infer'):
        '''
        Creates Dataset object from file.

        Args:
            file: file name where the dataset is stored in csv format
            sep: delimiter to be used when reading in file as csv
            names: list of column headers to be used when creating dataset object. If
                value is 'infer' then the first line of the file is assumed to be column
                headers. 
            output: list of target variables corresponding to X specified in the file. 

        Returns:
            Returns a Dataset object created from reading in X, target and header from csv
            For example a csv file containing:

            
            weight,height,length,output
            100,180,5,0
            50,60,2,1
            

            Would return:
            >>>Dataset(X=[[100,180,5], [50,60,2]], target=[0,1], header=['weight', 'height', 'length'])
        '''
        skiprows = 0
        if header == 'infer':
            with open(file, 'r') as f:
                reader = csv.reader(f, delimiter=sep)
                if target == 'infer':
                    header = next(reader)[:-1]
                else:
                    header = next(reader)
            skiprows = 1
        X = np.loadtxt(file, delimiter=sep, skiprows=skiprows)
        if target == 'infer':
            target = X[:, -1].astype(int).tolist() # last column in csv are target variables
            X = np.delete(X, -1, axis=1)
        class_names = [str(i) for i in range(0, len(set(target)))] # class names just as numbers
        return Dataset(X, target, header, class_names)

    def __len__(self):
        return self._X.shape[0]

    def __str__(self):
        '''
        Method for printing out dataset nicely.

        Returns:
            String of dataset containing data, headers and target variables.
        '''
        shape = self._X.shape
        b = np.zeros((shape[0], shape[1]+1))
        b[:, 0:shape[1]] = self._X
        b[:, -1] = self._target
        matrix = [self._header + ['targets']]
        for l in b.tolist():
            matrix.append(l)
        return('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix]))

    def __getitem__(self, key):
        '''
        Method for supporting pandas like indexing e.g. print(Dataset['size'])
        '''
        if type(key) == str: # check if list of keys or just one key
            key = [key]
        i = [self._header.index(item) for item in key]
        return self._X[:, i]

    def __setitem__(self, key, val):
        '''
        Method for supporting pandas like adding e.g. Dataset['size'] = np.array(...)
        '''
        if type(key) == str: # In case user wants to add multiple columns
            self.add_columns(key, val)
        else:
            if len(key) != len(val):
                raise ValueError('Number of keys and values must be equal')
            matrix = np.vstack(val).T
        self.add_columns(key, matrix)

    def __delitem__(self, key):
        '''
        Deletes column in matrix

        e.g. del Dataset['size']
        '''
        self.remove_columns([key])

    @property
    def X(self):
        return self._X

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        if len(target) != self._X.shape[0]:
            raise ValueError('Number of targets given must be equal to the number of data points in X')
        self._target = target

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, header):
        if len(header) != self._X.shape[1]:
            raise ValueError('Length of header must be the same as number of features in X')
        self._header = header

    @property
    def class_names(self):
        return self._class_names

    @class_names.setter
    def class_names(self, class_names):
        if len(class_names) != len(set(self._target)):
            raise ValueError('Number of class names given must be equal to number of classes defined')

    def add_columns(self, name, values):
        '''
        Adds column to matrix X and name to header

        Args:
            name: list of names of columns to add
            values: array for adding one column or ndarray for adding multiple columns
        '''
        shape = self._X.shape
        self._header = self._header + name
        if len(name) == 1:
            b = np.zeros((shape[0], shape[1]+1)) 
            b[:, 0:shape[1]] = self._X
            b[:, -1] = values
        else:
            b = np.zeros((shape[0], shape[1]+values.shape[1]))
            b[:, 0:shape[1]] = self._X
            b[:, -values.shape[1]:] = values
        self._X = b

    def add_data(self, data):
        '''
        Adds datapoint to end of X ndarray
        '''
        if type(data) == np.ndarray: # if only one datapoint is given
            data = [data]
        data = np.hstack(data)

    def remove_data(self, ind):
        '''
        Removes data points with index in ind
        '''
        if type(ind) == int: # if only one index is given
            ind = [ind]
        for i in ind:
            if i>=self._X.shape[0]:
                raise ValueError('{} is out of range for dataset.'.format(i))
        self._X = np.delete(self._X, ind, axis=0)
        self._target = [i for j, i in enumerate(self._target) if j not in ind]

    def remove_columns(self, features):
        '''
        Removes columns whose names or indices are in features
        '''
        if not features:
            raise ValueError('Empty list passed to remove_columns method.')
        column_index = []
        for f in features:
            if f not in self._header:
                raise ValueError(f+" is not a colum name in stored header.")
            column_index.append(self._header.index(f))
        self._header = [x for x in self._header if x not in features]
        self._X = np.delete(self._X, column_index, axis=1)

    def combine_features(self, features, name=None, keep=True, operation=np.mean):
        '''
        Combines features using operation.

        Args:
            features: List of features to combine or list of column indices
            name: String name of new feature to be appended to dataset. If None then name will
                be features that are combined separated by '-' e.g. length-width-combined.
                Default is None.
            keep: Boolean to say whether to keep the columns that are to be combined or delete
                them. Default is True.
            operation: Numpy operation that will operate on a list of features. Must be able to 
            take axis parameter and operate on ndarrays. Default is np.mean. 
        '''
        if not features:
            raise ValueError('Empty list passed to combine_features method.')
        ind_bool = False # false if features is list of strings, true if lists of indicies
        if type(features[0]) == int:
            ind_bool = True
        if not name: # Setup name of new feature
            if ind_bool:
                name = '{}-{}-combined'.format(features)
            else:
                name = '-'.join(features)
        ind_flag = False # Whether features is a list of indices or not
        if type(features[0]) == str:
            ind = [self._header.index(item) for item in features]
            val = self._X[:, ind]
            new_feature = operation(val, axis=1)
            self.add_columns([name], new_feature)
