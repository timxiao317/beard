# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Advanced author disambiguation example.

This example shows how to build a full author disambiguation pipeline.
The pipeline is made of two steps:

    1) Supervised learning, for inferring a distance or affinity function
       between publications. This estimator is learned from labeled paired data
       and models whether two publications have been authored by the same
       person.

    2) Semi-supervised block clustering, for grouping together publications
       from the same author. Publications are blocked by last name + first
       initial, and then clustered using hierarchical clustering together with
       the affinity function learned at the previous step. For each block,
       the best cut-off threshold is chosen so as to maximize some scoring
       metric on the provided labeled data.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>

"""

from __future__ import print_function

import argparse
import pickle
import numpy as np
import sys

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import v_measure_score
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import squareform

from beard.clustering import BlockClustering
from beard.clustering import ScipyHierarchicalClustering
from beard.metrics import b3_f_score
from beard.metrics import b3_precision_recall_fscore
from beard.similarity import PairTransformer
from beard.similarity import CosineSimilarity
from beard.similarity import AbsoluteDifference
from beard.utils import normalize_name
from beard.utils import name_initials
from beard.utils import FuncTransformer
from beard.utils import Shaper


def resolve_publications(signatures, records):
    """Resolve the 'publication' field in signatures."""
    if isinstance(signatures, list):
        signatures = {s["signature_id"]: s for s in signatures}

    if isinstance(records, list):
        records = {r["publication_id"]: r for r in records}

    for signature_id, signature in signatures.items():
        signature["publication"] = records[signature["publication_id"]]

    return signatures, records


def get_author_full_name(s):
    v = s["author_name"]
    v = normalize_name(v) if v else ""
    return v


def get_author_other_names(s):
    v = s["author_name"]
    v = v.split(",", 1)
    v = normalize_name(v[1]) if len(v) == 2 else ""
    return v


def get_author_initials(s):
    v = s["author_name"]
    v = v if v else ""
    v = "".join(name_initials(v))
    return v


def get_author_affiliation(s):
    v = s["author_affiliation"]
    v = normalize_name(v) if v else ""
    return v


def get_title(s):
    v = s["publication"]["title"]
    v = v if v else ""
    return v


def get_journal(s):
    v = s["publication"]["journal"]
    v = v if v else ""
    return v


def get_abstract(s):
    v = s["publication"]["abstract"]
    v = v if v else ""
    return v


def get_coauthors(s):
    v = s["publication"]["authors"]
    v = " ".join(v)
    return v


def get_keywords(s):
    v = s["publication"]["keywords"]
    v = " ".join(v)
    return v


def get_collaborations(s):
    v = s["publication"]["collaborations"]
    v = " ".join(v)
    return v


def get_references(s):
    v = s["publication"]["references"]
    v = " ".join(str(r) for r in v)
    v = v if v else ""
    return v


def get_year(s):
    v = s["publication"]["year"]
    v = int(v) if v else -1
    return v


def _groupby_signature_id(r):
    return r[0]["signature_id"]


def build_distance_estimator(X, y):
    # Build a vector reprensation of a pair of signatures
    transformer = FeatureUnion([
        ("author_full_name_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("full_name", FuncTransformer(func=get_author_full_name)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=_groupby_signature_id)),
            ("combiner", CosineSimilarity())
        ])),
        ("author_other_names_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("other_names", FuncTransformer(func=get_author_other_names)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=_groupby_signature_id)),
            ("combiner", CosineSimilarity())
        ])),
        ("author_initials_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("initials", FuncTransformer(func=get_author_initials)),
                ("shaper", Shaper(newshape=(-1,))),
                ("count", CountVectorizer(analyzer="char_wb",
                                          ngram_range=(1, 1),
                                          binary=True,
                                          decode_error="replace")),
            ]), groupby=_groupby_signature_id)),
            ("combiner", CosineSimilarity())
        ])),
        ("affiliation_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("affiliation", FuncTransformer(func=get_author_affiliation)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=_groupby_signature_id)),
            ("combiner", CosineSimilarity())
        ])),
        ("coauthors_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("coauthors", FuncTransformer(func=get_coauthors)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=_groupby_signature_id)),
            ("combiner", CosineSimilarity())
        ])),
        ("title_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("title", FuncTransformer(func=get_title)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=_groupby_signature_id)),
            ("combiner", CosineSimilarity())
        ])),
        ("journal_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("journal", FuncTransformer(func=get_journal)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(analyzer="char_wb",
                                           ngram_range=(2, 4),
                                           dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=_groupby_signature_id)),
            ("combiner", CosineSimilarity())
        ])),
        ("abstract_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("abstract", FuncTransformer(func=get_abstract)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=_groupby_signature_id)),
            ("combiner", CosineSimilarity())
        ])),
        ("keywords_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("keywords", FuncTransformer(func=get_keywords)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=_groupby_signature_id)),
            ("combiner", CosineSimilarity())
        ])),
        ("collaborations_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("collaborations", FuncTransformer(func=get_collaborations)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=_groupby_signature_id)),
            ("combiner", CosineSimilarity())
        ])),
        ("references_similarity", Pipeline([
            ("pairs", PairTransformer(element_transformer=Pipeline([
                ("references", FuncTransformer(func=get_references)),
                ("shaper", Shaper(newshape=(-1,))),
                ("tf-idf", TfidfVectorizer(dtype=np.float32,
                                           decode_error="replace")),
            ]), groupby=_groupby_signature_id)),
            ("combiner", CosineSimilarity())
        ])),
        ("year_diff", Pipeline([
            ("pairs", FuncTransformer(func=get_year, dtype=np.int)),
            ("combiner", AbsoluteDifference())  # FIXME: when one is missing
        ]))])

    # Train a classifier on these vectors
    classifier = GradientBoostingClassifier(n_estimators=500,
                                            max_depth=9,
                                            max_features=10,
                                            learning_rate=0.125,
                                            verbose=3)

    # Return the whole pipeline
    estimator = Pipeline([("transformer", transformer),
                          ("classifier", classifier)]).fit(X, y)

    return estimator


def affinity(X, step=10000):
    """Custom affinity function, using a pre-learned distance estimator."""
    # This assumes that 'distance_estimator' lives in global

    all_i, all_j = np.triu_indices(len(X), k=1)
    n_pairs = len(all_i)
    distances = np.zeros(n_pairs)

    for start in range(0, n_pairs, step):
        end = min(n_pairs, start+step)
        Xt = np.empty((end-start, 2), dtype=np.object)

        for k, (i, j) in enumerate(zip(all_i[start:end],
                                       all_j[start:end])):
            Xt[k, 0], Xt[k, 1] = X[i, 0], X[j, 0]

        Xt = distance_estimator.predict_proba(Xt)[:, 1]
        distances[start:end] = Xt[:]

    return squareform(distances)


def affinity_baseline(X):
    """Compute pairwise distances between (author, affiliation) tuples.

    Note that this function is a heuristic. It should ideally be replaced
    by a more robust distance function, e.g. using a model learned over
    pairs of tuples.
    """
    distances = np.zeros((len(X), len(X)), dtype=np.float)

    for i, j in zip(*np.triu_indices(len(X), k=1)):
        name_i = normalize_name(X[i, 0]["author_name"])
        name_j = normalize_name(X[j, 0]["author_name"])

        if name_i == name_j:
            distances[i, j] = 0.0
        else:
            distances[i, j] = 1.0

    distances += distances.T
    return distances


def blocking(X):
    """Blocking function using last name and first initial as key."""
    def last_name_first_initial(name):
        names = name.split(",", 1)

        try:
            name = "%s %s" % (names[0], names[1].strip()[0])
        except IndexError:
            name = names[0]

        name = normalize_name(name)
        return name

    blocks = []

    for signature in X[:, 0]:
        blocks.append(last_name_first_initial(signature["author_name"]))

    return np.array(blocks)


def tune(clusterer, truth, thresholds):
    best_score = -np.inf
    best_t = 1.0

    for t in thresholds:
        for b, c in clusterer.clusterers_.items():
            if hasattr(c, "best_threshold_"):
                del c.best_threshold_

            if hasattr(c, "threshold"):
                c.set_params(threshold=t)

        score = b3_f_score(truth, clusterer.labels_)
        print(t, score)

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


if __name__ == "__main__":
    # Parse command line arugments
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance_model', default=None, type=str)
    parser.add_argument('--distance_data', default=None, type=str)
    parser.add_argument('--clustering_data', default=None, type=str)
    parser.add_argument('--clustering_threshold', default=0.9995, type=float)
    parser.add_argument('--clustering_test_size', default=0.75, type=float)
    parser.add_argument('--clustering_random_state', default=42, type=int)
    parser.add_argument('--baseline', default=0, type=int)
    parser.add_argument('--tuner', default=0, type=int)
    args = parser.parse_args()

    # Load paired data
    if args.distance_data is not None:
        X, y, signatures, records = pickle.load(open(args.distance_data, "r"))
        signatures, records = resolve_publications(signatures, records)

        Xt = np.empty((len(X), 2), dtype=np.object)
        for k, (i, j) in enumerate(X):
            Xt[k, 0] = signatures[i]
            Xt[k, 1] = signatures[j]
        X = Xt

        # Learn a distance estimator on paired signatures
        distance_estimator = build_distance_estimator(X, y)

        if args.distance_model is not None:
            pickle.dump(distance_estimator, open(args.distance_model, "w"))

    if args.distance_model is not None and args.distance_data is None:
        distance_estimator = pickle.load(open(args.distance_model, "r"))

    # Load signatures to cluster
    if args.clustering_data is not None:
        signatures, truth, records = pickle.load(open(args.clustering_data, "r"))
        _, records = resolve_publications(signatures, records)

        X = np.empty((len(signatures), 1), dtype=np.object)
        for i, signature in enumerate(signatures):
            X[i, 0] = signature

        # Semi-supervised block clustering
        train, test = train_test_split(np.arange(len(X)),
                                       test_size=args.clustering_test_size,
                                       random_state=args.clustering_random_state)
        y = -np.ones(len(X), dtype=np.int)

        if args.tuner == 0:
            y[train] = truth[train]

        clusterer = BlockClustering(
            blocking=blocking,
            base_estimator=ScipyHierarchicalClustering(
                threshold=args.clustering_threshold,
                affinity=affinity if args.baseline == 0 else affinity_baseline,
                method="complete",
                scoring=b3_f_score),
            verbose=1,
            n_jobs=-1).fit(X, y)

        if args.tuner > 0:
            best_t, best_score = tune(clusterer, truth, np.arange(0.99, 1.0, 0.0001))
            print(args.clustering_data, best_t, best_score)

        else:
            labels = clusterer.labels_

            # # Print clusters
            # for cluster in np.unique(labels):
            #     entries = set()

            #     for signature in X[labels == cluster, 0]:
            #         entries.add((signature["author_name"],
            #                      signature["author_affiliation"]))

            #     print("Cluster #%d = %s" % (cluster, entries))
            # print()

            # Statistics
            results = []
            results.append(args.clustering_data)
            results.append(args.clustering_test_size)
            results.append(args.distance_model)

            for score in [b3_precision_recall_fscore]:
                results.extend(score(truth, labels))
                results.extend(score(truth[train], labels[train]))
                results.extend(score(truth[test], labels[test]))

            print(results)
