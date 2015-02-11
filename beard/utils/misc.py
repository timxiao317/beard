# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Miscellaneous helpers.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

from __future__ import print_function

from functools import wraps
import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


def memoize(func):
    """Memoization function."""
    cache = {}

    @wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrap


class MemoizedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, key, capacity=1000, verbose=0):
        self.transformer = transformer
        self.key = key
        self.capacity = capacity
        self.verbose = verbose

    def fit(self, X, y=None):
        self.transformer.fit(X, y=y)
        Xt = self.transformer.transform(X[:1])
        self.sparse_ = sp.issparse(Xt)

        if self.sparse_:
            self.cache_ = sp.lil_matrix((self.capacity, Xt.shape[1]),
                                        dtype=Xt.dtype)
        else:
            self.cache_ = np.empty((self.capacity, Xt.shape[1]),
                                   dtype=Xt.dtype)

        self.index_ = {}  # from key to index in cache
        self.reverse_index_ = {}  # from index in cache to key
        self.next_ = 0

        return self

    def transform(self, X, y=None):
        if self.sparse_:
            Xt = sp.lil_matrix((X.shape[0], self.cache_.shape[1]),
                               dtype=self.cache_.dtype)
        else:
            Xt = np.empty((X.shape[0], self.cache_.shape[1]),
                          dtype=self.cache_.dtype)

        # Find hits and misses
        hits = []
        hits_indices = []
        misses = []
        misses_keys = []

        for i in range(X.shape[0]):
            key_i = self.key(X[i])

            if key_i in self.index_:
                hits.append(i)
                hits_indices.append(self.index_[key_i])

            else:
                misses.append(i)
                misses_keys.append(key_i)

        if self.verbose > 0:
            print("MemoizedTransformer: %d hits, %d misses" %
                  (len(hits), len(misses)))

        # Fill the hits
        if len(hits) > 0:
            if self.sparse_:
                self._sparse_assign(Xt, hits, self.cache_, hits_indices)
            else:
                Xt[hits] = self.cache_[hits_indices]

        # Fill the misses
        if len(misses) > 0:
            misses = np.array(misses)
            uniques, indices, inverse = np.unique(misses_keys,
                                                  return_index=True,
                                                  return_inverse=True)

            X_missing = self.transformer.transform(X[misses[indices]])

            if self.sparse_:
                X_missing = X_missing.tolil()
                self._sparse_assign(Xt, misses, X_missing, inverse)
            else:
                Xt[misses] = X_missing[inverse]

            # Fill the cache
            tocopy = {}

            for j, (i, key_i) in enumerate(zip(misses, misses_keys)):
                if key_i not in self.index_:
                    if self.next_ in self.reverse_index_:
                        old_key = self.reverse_index_[self.next_]
                        if old_key in self.index_:
                            del self.index_[old_key]

                    self.index_[key_i] = self.next_
                    self.reverse_index_[self.next_] = key_i
                    tocopy[self.next_] = inverse[j]

                    self.next_ += 1
                    if self.next_ == self.capacity:
                        self.next_ = 0

            if self.sparse_:
                self._sparse_assign(self.cache_, tocopy.keys(),
                                    X_missing, tocopy.values())
            else:
                self.cache_[tocopy.keys()] = X_missing[tocopy.values()]

        if self.sparse_:
            return Xt.tocsr()
        else:
            return Xt

    def _reset(self):
        self.index_ = {}
        self.reverse_index_ = {}
        self.next_ = 0

        if hasattr(self, "cache_"):
            if self.sparse_:
                self.cache_ = sp.lil_matrix(self.cache_.shape,
                                            dtype=self.cache_.dtype)
            else:
                self.cache_ = np.empty(self.cache_.shape,
                                       dtype=self.cache_.dtype)

        import gc
        gc.collect()

    def _sparse_assign(self, A, rowsA, B, rowsB):
        A.rows[rowsA] = B.rows[rowsB]
        A.data[rowsA] = B.data[rowsB]
