"""
Functions for testing data augmentation classes.
"""
# Author: Alex Hepburn <ah13558@bristol.ac.uk>
#         Kacper Sokol <k.sokol@bristol.ac.uk>
# License: new BSD

import pytest

import numpy as np

import fatf
from fatf.exceptions import IncorrectShapeError
import fatf.utils.data.augmentation as fuda
from fatf.utils.testing.arrays import (BASE_NP_ARRAY, BASE_STRUCTURED_ARRAY,
                                       NOT_BASE_NP_ARRAY)

ONE_D_ARRAY = np.array([0, 4, 3, 0])

NUMERICAL_NP_ARRAY = np.array([
    [0, 0, 0.08, 0.69],
    [1, 0, 0.03, 0.29],
    [0, 1, 0.99, 0.82],
    [2, 1, 0.73, 0.48],
    [1, 0, 0.36, 0.89],
    [0, 1, 0.07, 0.21]])
NUMERICAL_STRUCT_ARRAY = np.array(
    [(0, 0, 0.08, 0.69),
     (1, 0, 0.03, 0.29),
     (0, 1, 0.99, 0.82),
     (2, 1, 0.73, 0.48),
     (1, 0, 0.36, 0.89),
     (0, 1, 0.07, 0.21)],
    dtype=[('a', 'i'), ('b', 'i'), ('c', 'f'), ('d', 'f')])
CATEGORICAL_NP_ARRAY = np.array([
    ['a', 'b', 'c'],
    ['a', 'f', 'g'],
    ['b', 'c', 'c']])
CATEGORICAL_STRUCT_ARRAY = np.array(
    [('a', 'b', 'c'),
     ('a', 'f', 'g'),
     ('b', 'c', 'c')],
    dtype=[('a', 'U1'), ('b', 'U1'), ('c', 'U1')])
MIXED_ARRAY = np.array(
    [(0, 'a', 0.08, 'a'),
     (0, 'f', 0.03, 'bb'),
     (1, 'c', 0.99, 'aa'),
     (1, 'a', 0.73, 'a'),
     (0, 'c', 0.36, 'b'),
     (1, 'f', 0.07, 'bb')],
    dtype=[('a', 'i'), ('b', 'U1'), ('c', 'f'), ('d', 'U2')])


class BaseAugmentor(fuda.Augmentation):
    """
    Dummy class to test :func`fatf.utils.data.augmentation.Augmentation.
    _validate_input` and :func`fatf.utils.data.augmentation.Augmentation.
    _validate_sample_input`.

    """
    def __init__(self, dataset, categorical_indices=None):
        super(BaseAugmentor, self).__init__(dataset, categorical_indices)
    
    def sample(self, data_row=None, num_samples=10):
        self._validate_sample_input(data_row, num_samples)


class BrokenAugmentor(fuda.Augmentation):
    """
    Class with no `sample` function defined.
    """
    def __init__(self, dataset, categorical_indices=None):
        super(BaseAugmentor, self).__init__(dataset, categorical_indices)


def test_Augmentation():
    """
    tests :class`fatf.utils.data.augmentation.Augmentation`
    """
    msg = ('Can\'t instantiate abstract class BrokenAugmentor with abstract '
           'methods sample')
    with pytest.raises(TypeError) as exin:
        augmentor = BrokenAugmentor(NUMERICAL_NP_ARRAY)
    assert str(exin.value) == msg

    msg = ('dataset must be a numpy.ndarray.')
    with pytest.raises(TypeError) as exin:
        augmentor = BaseAugmentor(0)
    assert str(exin.value) == msg

    msg = ('categorical_indices must be a numpy.ndarray or None')
    with pytest.raises(TypeError) as exin:
        augmentor = BaseAugmentor(NUMERICAL_NP_ARRAY, categorical_indices=0)
    assert str(exin.value) == msg

    msg = ('The input dataset must be a 2-dimensional array.')
    with pytest.raises(IncorrectShapeError) as exin:
        augmentor = BaseAugmentor(ONE_D_ARRAY, np.array([0]))
    assert str(exin.value) == msg

    msg = ('Indices {} are not valid for input dataset.')
    with pytest.raises(IndexError) as exin:
        augmentor = BaseAugmentor(NUMERICAL_NP_ARRAY, np.array([10]))
    assert str(exin.value) == msg.format(np.array([10]))
    with pytest.raises(IndexError) as exin:
        augmentor = BaseAugmentor(NUMERICAL_NP_ARRAY, np.array(['a']))
    assert str(exin.value) == msg.format(np.array(['a']))
    with pytest.raises(IndexError) as exin:
        augmentor = BaseAugmentor(NUMERICAL_STRUCT_ARRAY, np.array(['l']))
    assert str(exin.value) == msg.format(np.array(['l']))

    msg = ('No categorical_indcies were provided. The categorical columns '
           'will be inferred by looking at the type of values in the dataset.')
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(CATEGORICAL_NP_ARRAY)
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, np.array([0, 1, 2]))
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(CATEGORICAL_STRUCT_ARRAY)
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, 
                          np.array(['a', 'b', 'c']))
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(MIXED_ARRAY)
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, np.array(['b', 'd']))

    msg = ('String based indices were found in dataset but not given as '
           'categorical_indices. String based columns will automatically be '
           'treated as categorical columns.')
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(CATEGORICAL_NP_ARRAY, np.array([0]))
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, np.array([0, 1, 2]))
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(CATEGORICAL_STRUCT_ARRAY, np.array(['a']))
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, 
                          np.array(['a', 'b', 'c']))
    with pytest.warns(UserWarning) as warning:
        augmentor = BaseAugmentor(MIXED_ARRAY, np.array(['b']))
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices, np.array(['b', 'd']))

    # Validate sample input rows
    augmentor = BaseAugmentor(NUMERICAL_NP_ARRAY)
    msg = ('data_row must be numpy.ndarray.')
    with pytest.raises(TypeError) as exin:
        sample = augmentor.sample(1)
    assert str(exin.value) == msg

    msg = ('num_samples must be an integer.')
    with pytest.raises(TypeError) as exin:
        sample = augmentor.sample(np.array([]), 'a')
    assert str(exin.value) == msg

    msg = ('num_samples must be an integer greater than 0.')
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(np.array([]), -1)
    assert str(exin.value) == msg

    msg = ('data_row provided is not of the same dtype as the dataset used to '
           'initialise this class. Please ensure that the dataset and data_row '
           'dtypes are identical.')
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(np.array(['a', 'b', 'c', 'd']))
    assert str(exin.value) == msg
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(MIXED_ARRAY[0])
    assert str(exin.value) == msg
    augmentor = BaseAugmentor(CATEGORICAL_STRUCT_ARRAY, 
                              np.array(['a', 'b', 'c']))
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(np.array([0.1]))
    assert str(exin.value) == msg
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(MIXED_ARRAY[0][['a', 'b']])
    assert str(exin.value) == msg

    augmentor = BaseAugmentor(NUMERICAL_NP_ARRAY)
    msg = ('data_row must contain the same number of features as the dataset '
           'used in the class constructor.')
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(np.array([0.1]))
    assert str(exin.value) == msg
    augmentor = BaseAugmentor(CATEGORICAL_NP_ARRAY, np.array([0, 1, 2]))
    with pytest.raises(ValueError) as exin:
        sample = augmentor.sample(np.array(['a']))
    assert str(exin.value) == msg

    msg = ('data_row must be a 1-dimensional array.')
    with pytest.raises(IncorrectShapeError) as exin:
        sample = augmentor.sample(NUMERICAL_NP_ARRAY)
    assert str(exin.value) == msg
    augmentor = BaseAugmentor(NUMERICAL_STRUCT_ARRAY)
    with pytest.raises(IncorrectShapeError) as exin:
        sample = augmentor.sample(NUMERICAL_STRUCT_ARRAY)
    assert str(exin.value) == msg


def test_NormalSampling():
    """
    tests :func`fatf.utils.data.augmentation.NormalSampling`
    """
    fatf.setup_random_seed()

    # Test class inheritence and calcuating non_categorical_indices
    # and categorical_indices
    augmentor = fuda.NormalSampling(NUMERICAL_NP_ARRAY, np.array([0]))
    assert augmentor.__class__.__bases__[0].__name__ == 'Augmentation'
    assert np.array_equal(augmentor.categorical_indices, np.array([0]))
    assert np.array_equal(augmentor.non_categorical_indices,
                          np.array([1, 2, 3]))

    augmentor = fuda.NormalSampling(NUMERICAL_NP_ARRAY)
    assert np.array_equal(augmentor.categorical_indices, np.array([]))
    assert np.array_equal(augmentor.non_categorical_indices,
                          np.array([0, 1, 2, 3]))

    augmentor = fuda.NormalSampling(NUMERICAL_STRUCT_ARRAY, np.array(['a']))
    assert np.array_equal(augmentor.categorical_indices, np.array(['a']))
    assert np.array_equal(augmentor.non_categorical_indices,
                          np.array(['b', 'c', 'd']))

    msg = ('No categorical_indcies were provided. The categorical columns '
           'will be inferred by looking at the type of values in the dataset.')
    with pytest.warns(UserWarning) as warning:
        augmentor = fuda.NormalSampling(CATEGORICAL_NP_ARRAY)
    assert len(warning) == 1
    assert str(warning[0].message) == msg
    assert np.array_equal(augmentor.categorical_indices,
                          np.array([0, 1, 2]))
    assert np.array_equal(augmentor.non_categorical_indices, np.array([]))
