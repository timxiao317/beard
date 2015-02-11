# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Tests of misc utilities.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_equal

import scipy.sparse as sp

from sklearn.preprocessing import StandardScaler

from ..transformers import FuncTransformer
from ..misc import MemoizedTransformer


def test_memoized_transformer():
    """Test for MemoizedTransformer."""
    X = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int)
    memo = MemoizedTransformer(FuncTransformer(lambda v: v+1),
                               lambda r: r[0], 5).fit(X)

    # Misses
    Xt = memo.transform(X)
    assert_array_equal(Xt, X + 1)
    assert_equal(memo.cache_.shape, (5, 3))
    assert_equal(len(memo.index_), 2)

    # Hits
    Xt = memo.transform(X)
    assert_array_equal(Xt, X + 1)
    assert_equal(memo.cache_.shape, (5, 3))
    assert_equal(len(memo.index_), 2)

    # Fill in the cache
    X = np.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8],
                  [9, 10, 11],
                  [12, 13, 14],
                  [15, 16, 17],
                  [15, 16, 17]], dtype=np.int)

    Xt = memo.transform(X)
    assert_array_equal(Xt, X + 1)
    assert_equal(memo.cache_.shape, (5, 3))
    assert_equal(len(memo.index_), 5)


def test_memoized_transformer_sparse():
    """Test for MemoizedTransformer, using sparse data."""
    X = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float)
    X = sp.csr_matrix(X)
    memo = MemoizedTransformer(StandardScaler(with_mean=False),
                               lambda r: r[0, 0], 5).fit(X)

    # Misses
    Xt = memo.transform(X)
    assert_array_almost_equal(Xt.todense(), [[0.0, 2./3, 4./3],
                                             [2.0, 8./3, 10./3]])
    assert_equal(memo.cache_.shape, (5, 3))
    assert_equal(len(memo.index_), 2)

    # Hits
    Xt = memo.transform(X)
    assert_array_almost_equal(Xt.todense(), [[0.0, 2./3, 4./3],
                                             [2.0, 8./3, 10./3]])
    assert_equal(memo.cache_.shape, (5, 3))
    assert_equal(len(memo.index_), 2)

    # Fill in the cache
    X = np.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8],
                  [9, 10, 11],
                  [12, 13, 14],
                  [15, 16, 17]], dtype=np.float)
    X = sp.csr_matrix(X)

    Xt = memo.transform(X)
    assert_equal(memo.cache_.shape, (5, 3))
    assert_equal(len(memo.index_), 5)
