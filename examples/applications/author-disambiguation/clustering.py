# -*- coding: utf-8 -*-
#
# This file is part of Beard.
# Copyright (C) 2015 CERN.
#
# Beard is a free software; you can redistribute it and/or modify it
# under the terms of the Revised BSD License; see LICENSE file for
# more details.

"""Author disambiguation -- Clustering.

See README.rst for further details.

.. codeauthor:: Gilles Louppe <g.louppe@cern.ch>
.. codeauthor:: Mateusz Susik <mateusz.susik@cern.ch>

"""

import argparse
import codecs
import os
import pickle
import json
from os.path import join, dirname, abspath

import numpy as np

from functools import partial
from utils import load_split

# These imports are used during unpickling.
from utils import get_author_full_name
from utils import get_author_other_names
from utils import get_author_initials
from utils import get_surname
from utils import get_first_initial
from utils import get_second_initial
from utils import get_author_affiliation
from utils import get_title
from utils import get_journal
from utils import get_abstract
from utils import get_coauthors_from_range
from utils import get_keywords
from utils import get_collaborations
from utils import get_references
from utils import get_topics
from utils import get_year
from utils import group_by_signature
from utils import load_signatures

from beard.clustering import BlockClustering
from beard.clustering import block_last_name_first_initial
from beard.clustering import block_phonetic
from beard.clustering import ScipyHierarchicalClustering
from beard.metrics import b3_f_score
from beard.metrics import b3_precision_recall_fscore
from beard.metrics import paired_precision_recall_fscore, paired_precision_recall_fscore_new





def _affinity(X, step=10000):
    """Custom affinity function, using a pre-learned distance estimator."""
    # Assumes that 'distance_estimator' lives in global, making things fast
    global distance_estimator

    all_i, all_j = np.triu_indices(len(X), k=1)
    n_pairs = len(all_i)
    distances = np.zeros(n_pairs, dtype=np.float64)

    for start in range(0, n_pairs, step):
        end = min(n_pairs, start+step)
        Xt = np.empty((end-start, 2), dtype=np.object)

        for k, (i, j) in enumerate(zip(all_i[start:end],
                                       all_j[start:end])):
            Xt[k, 0], Xt[k, 1] = X[i, 0], X[j, 0]

        Xt = distance_estimator.predict_proba(Xt)[:, 1]
        distances[start:end] = Xt[:]

    return distances


def pairwise_precision_recall_f1(preds, truths):
    tp = 0
    fp = 0
    fn = 0
    n_samples = len(preds)
    for i in range(n_samples - 1):
        pred_i = preds[i]
        for j in range(i + 1, n_samples):
            pred_j = preds[j]
            if pred_i == pred_j:
                if truths[i] == truths[j]:
                    tp += 1
                else:
                    fp += 1
            elif truths[i] == truths[j]:
                fn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    tp = float(tp)
    if tp_plus_fp == 0:
        precision = 0.
    else:
        precision = tp / tp_plus_fp
    if tp_plus_fn == 0:
        recall = 0.
    else:
        recall = tp / tp_plus_fn
    if not precision or not recall:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return tp, fp, fn, precision, recall, f1


def clustering(input_signatures, input_records, distance_model,
               input_clusters=None, output_clusters=None,
               verbose=1, n_jobs=-1, clustering_method="average",
               train_signatures_file=None, clustering_threshold=None,
               results_file=None, blocking_function="block_phonetic",
               blocking_threshold=1, blocking_phonetic_alg="nysiis"):
    """Cluster signatures using a pretrained distance model.

    Parameters
    ----------
    :param input_signatures: string
        Path to the file with signatures. The content should be a JSON array
        of dictionaries holding metadata about signatures.

        [{"signature_id": 0,
          "author_name": "Doe, John",
          "publication_id": 10, ...}, { ... }, ...]

    :param input_records: string
        Path to the file with records. The content should be a JSON array of
        dictionaries holding metadata about records

        [{"publication_id": 0,
          "title": "Author disambiguation using Beard", ... }, { ... }, ...]

    :param distance_model: string
        Path to the file with the distance model. The file should be a pickle
        created using the ``distance.py`` script.

    :param input_clusters: string
        Path to the file with knownn clusters. The file should be a dictionary,
        where keys are cluster labels and values are the `signature_id` of the
        signatures grouped in the clusters. Signatures assigned to the cluster
        with label "-1" are not clustered.

        {"0": [0, 1, 3], "1": [2, 5], ...}

    :param output_clusters: string
        Path to the file with output cluster. The file will be filled with
        clusters, using the same format as ``input_clusters``.

    :param verbose: int
        If not zero, function will output scores on stdout.

    :param n_jobs: int
        Parameter passed to joblib. Number of threads to be used.

    :param clustering_method: string
        Parameter passed to ``ScipyHierarchicalClustering``. Used only if
        ``clustering_test_size`` is specified.

    :param train_signatures_file: str
        Path to the file with train set signatures. Format the same as in
        ``input_signatures``.

    :param clustering_threshold: float
        Threshold passed to ``ScipyHierarchicalClustering``.

    :param results_file: str
        Path to the file where the results will be output. It will give
        additional information about pairwise variant of scores.

    :param blocking_function: string
        must be a defined blocking function. Defined functions are:
        - "block_last_name_first_initial"
        - "block_phonetic"

    :param blocking_threshold: int or None
        It determines the maximum allowed size of blocking on the last name
        It can only be:
        -   None; if the blocking function is block_last_name_first_initial
        -   int; if the blocking function is block_phonetic
            please check the documentation of phonetic blocking in
            beard.clustering.blocking_funcs.py

    :param blocking_phonetic_alg: string or None
        If not None, determines which phonetic algorithm is used. Options:
        -  "double_metaphone"
        -  "nysiis" (only for Python 2)
        -  "soundex" (only for Python 2)
    """
    # Assumes that 'distance_estimator' lives in global, making things fast
    global distance_estimator
    distance_estimator = pickle.load(open(distance_model, "rb"))

    try:
        distance_estimator.steps[-1][1].set_params(n_jobs=1)
    except:
        pass

    signatures, records = load_signatures(input_signatures,
                                          input_records)

    indices = {}
    X, y = np.empty((len(signatures), 1), dtype=np.object), None
    for i, signature in enumerate(sorted(signatures.values(),
                                         key=lambda s: s["signature_id"])):
        X[i, 0] = signature
        indices[signature["signature_id"]] = i

    if blocking_function == "block_last_name_first_initial":
        block_function = block_last_name_first_initial
    else:
        block_function = partial(block_phonetic,
                                 threshold=blocking_threshold,
                                 phonetic_algorithm=blocking_phonetic_alg)

    true_clusters = json.load(open(input_clusters, "r"))

    '''
    # Semi-supervised block clustering
    if input_clusters:
        true_clusters = json.load(open(input_clusters, "r"))
        y_true = -np.ones(len(X), dtype=np.int)

        labelIndexs = tuple(true_clusters.keys())
        for label, signature_ids in true_clusters.items():
            for signature_id in signature_ids:
                y_true[indices[signature_id]] = labelIndexs.index(label)

        y = -np.ones(len(X), dtype=np.int)

        if train_signatures_file:
            train_signatures = json.load(open(train_signatures_file, "r"))
            train_ids = [x['signature_id'] for x in train_signatures]
            del train_signatures
            y[train_ids] = y_true[train_ids]
            test_ids = list(set([x['signature_id'] for _, x in
                                 signatures.iteritems()]) - set(train_ids))
        else:
            y = None

    else:
        y = None
    '''

    clusterer = BlockClustering(
        # blocking=block_function,
        base_estimator=ScipyHierarchicalClustering(
            affinity=_affinity,
            n_clusters=len(true_clusters),
            threshold=clustering_threshold,
            method=clustering_method,
            supervised_scoring=b3_f_score),
        verbose=verbose,
        n_jobs=n_jobs).fit(X, y)

    pres = clusterer.labels_
    truth = {sig: clu for clu, sigs in true_clusters.items() for sig in sigs}
    result = pairwise_precision_recall_f1(truth, pres)
    print(result)
    if output_clusters:
        with open(output_clusters, 'w') as f:
            f.write(','.join(map(str, result)))
    return result
    '''
    labels = clusterer.labels_

    # Save predicted clusters
    if output_clusters:
        clusters = {}

        for label in np.unique(labels):
            mask = (labels == label)
            clusters[str(label)] = [r[0]["signature_id"] for r in X[mask]]

        json.dump(clusters, open(output_clusters, "w"))

    # Statistics
    if verbose and input_clusters:
        print("Number of blocks =", len(clusterer.clusterers_))
        print("True number of clusters", len(np.unique(y_true)))
        print("Number of computed clusters", len(np.unique(labels)))

        b3_overall = b3_precision_recall_fscore(y_true, labels)
        print("B^3 F-score (overall) =", b3_overall[2])

        if train_signatures_file:
            b3_train = b3_precision_recall_fscore(
                y_true[train_ids],
                labels[train_ids]
            )
            b3_test = b3_precision_recall_fscore(
                y_true[test_ids],
                labels[test_ids]
            )
            print("B^3 F-score (train) =", b3_train[2])
            print("B^3 F-score (test) =", b3_test[2])
            if results_file:
                paired_overall = paired_precision_recall_fscore(y_true, labels)
                paired_train = paired_precision_recall_fscore(
                    y_true[train_ids],
                    labels[train_ids]
                )
                paired_test = paired_precision_recall_fscore(
                    y_true[test_ids],
                    labels[test_ids]
                )

                json.dump({
                    "description": ["precision", "recall", "f_score"],
                    "b3": {"overall": list(b3_overall),
                           "train": list(b3_train),
                           "test": list(b3_test)
                           },
                    "paired": {"overall": list(paired_overall),
                               "train": list(paired_train),
                               "test": list(paired_test)
                               }
                }, open(results_file, 'w'))
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance_model", default="linkage.dat", type=str)
    # parser.add_argument("--input_signatures", required=True, type=str)
    # parser.add_argument("--input_records", required=True, type=str)
    # parser.add_argument("--input_clusters", default=None, type=str)
    # parser.add_argument("--output_clusters", required=True, type=str)
    parser.add_argument("--out_filename", default="result.csv", type=str)
    parser.add_argument("--train_dataset_name", default="whoiswho_new", type=str)
    parser.add_argument("--test_dataset_name", default="whoiswho_new", type=str)
    parser.add_argument("--split_dir", default="../../../../split/", type=str)
    parser.add_argument("--clustering_method", default="average", type=str)
    parser.add_argument("--clustering_threshold", default=None, type=float)
    parser.add_argument("--train_signatures", default=None, type=str)
    parser.add_argument("--results_file", default=None, type=str)
    parser.add_argument("--blocking_function", default="block_phonetic",
                        type=str)
    parser.add_argument("--blocking_threshold", default=1, type=int)
    parser.add_argument("--blocking_phonetic_alg", default="nysiis", type=str)
    parser.add_argument("--verbose", default=1, type=int)
    parser.add_argument("--n_jobs", default=1, type=int)
    args = parser.parse_args()
    exp_name = "{}_{}".format(args.train_dataset_name, args.test_dataset_name)
    _, train_name_list, test_name_list = load_split(args.split_dir, '{}_python2'.format(args.test_dataset_name))
    exp_path = join(dirname(abspath(__file__)), exp_name)
    model_path = join(dirname(abspath(__file__)), args.train_dataset_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    PROJ_DIR = dirname(dirname(dirname(dirname(abspath(__file__)))))
    PARENT_PROJ_DIR = dirname(PROJ_DIR)
    dataset_path = join(PARENT_PROJ_DIR, 'sota_data', 'louppe_data', args.test_dataset_name)
    output_file = join(exp_path, args.out_filename)
    distance_model = join(model_path, args.distance_model)
    wf = codecs.open(output_file, 'w', encoding='utf-8')
    wf.write('name,precision,recall,f1,tp,fp,fn\n')
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    precision_sum = 0
    recall_sum = 0
    for test_name in test_name_list[:400]:
        try:
            input_signatures = os.path.join(dataset_path, test_name, "signatures.json")
            input_records = os.path.join(dataset_path, test_name, "records.json")
            input_clusters = os.path.join(dataset_path, test_name, "clusters.json")
            tp, fp, fn, precision, recall, f1 = clustering(input_signatures, input_records, distance_model,
                       input_clusters, None,
                       args.verbose, args.n_jobs, args.clustering_method,
                       args.train_signatures, args.clustering_threshold,
                       args.results_file)
            wf.write('{0},{1:.5f},{2:.5f},{3:.5f},{4:.5f},{5:.5f},{6:.5f}\n'.format(test_name.encode('utf-8'), precision, recall, f1, tp, fp, fn))
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
            precision_sum += precision
            recall_sum += recall
        except:
            continue
    macro_precision = precision_sum / len(test_name_list)
    macro_recall = recall_sum / len(test_name_list)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    micro_precision = tp_sum / (tp_sum + fp_sum)
    micro_recall = tp_sum / (tp_sum + fn_sum)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    wf.write('macro,{0:.5f},{1:.5f},{2:.5f}\n'.format(
        macro_precision, macro_recall, macro_f1))
    wf.write('micro,{0:.5f},{1:.5f},{2:.5f}\n'.format(
        micro_precision, micro_recall, micro_f1))
