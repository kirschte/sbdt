# -*- coding: utf-8 -*-
# gtheo@ethz.ch
# modified by: Moritz Kirschte
"""Example test file."""

import os
from typing import Any, Optional

import sbdt
import matplotlib.pyplot as plt
import numpy as np
# pylint: disable=import-error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
# pylint: enable=import-error

os.environ["CUDA_VISIBLE_DEVICES"] = ""
plt.rcParams['figure.figsize'] = (11.7, 8.27)
plt.rcParams['figure.dpi'] = 200



def get_abalone(n_rows: Optional[int] = None) -> Any:
    """Parse the abalone dataset.

    Args:
        n_rows (int): Numbers of rows to read.

    Returns:
        Any: X, y, cat_idx, num_idx
    """
    # pylint: disable=redefined-outer-name,invalid-name
    # Re-encode gender information
    data = pd.read_csv(
        './datasets/real/abalone.data',
        names=['sex', 'length', 'diameter', 'height', 'whole weight',
               'shucked weight', 'viscera weight', 'shell weight', 'rings'])
    data['sex'] = pd.factorize(data['sex'])[0]
    if n_rows:
        data = data.head(n_rows)
    y = data.rings.values.astype(float)
    del data['rings']
    X = data.values.astype(float)
    cat_idx = [0]  # Sex
    num_idx = list(range(1, X.shape[1]))  # Other attributes
    cat_values = [[0, 1, 2], [], [], [], [], [], [], []]
    return X, y, cat_idx, num_idx, cat_values


if __name__ == '__main__':
    # pylint: disable=redefined-outer-name,invalid-name
    X, y, cat_idx, num_idx, cat_values = get_abalone()
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    # A simple baseline: mean of the training set
    y_pred = np.mean(y_train).repeat(len(y_test))
    print('Constant Mean - RMSE: {0:f}'.format(
        np.sqrt(np.mean(np.square(y_pred - y_test)))))

    # Train the model using a depth-first approach
    model = sbdt.S_BDT(
        # tree ensemble
        nb_trees = 150,
        subsampling_ratio = 0.1,
        learning_rate = 0.1,
        max_depth = 2,
        newton_boosting = False,
        balance_partition = True,
        lambda_reg_mode = 1,  # ADD
        l2_lambda = 15,
        reg_delta = 2.0,
        # split
        min_samples_split = 0,
        ignore_split_constraints = True,
        max_features = 0,  # RAND
        max_feature_values = 0,  # RAND
        criterion = 0,  # XGD_MSE
        reuse_attr = True,
        # privacy
        use_dp = True,
        privacy_budget = 0.1,
        privacy_budget_init_score_ratio = 0.1,
        privacy_budget_gain_ratio = 0.5,
        leaf_denom_noise_weight = 0.2,
        l2_threshold = 0.1,
        hess_l2_threshold = 1.0,
        init_score_threshold = 1.0,
        numeric_feature_weight = 1.0,
        leaf_noise = 1,  # GAUSS
        # individual privacy filter
        use_privacy_filter = False,
        approximate_privacy_filter = False,
        pf_additional_nb_trees = 0,
        pf_l2_threshold = 0.1,
        pf_hess_l2_threshold = 1.0,
        pf_subsampling_ratio_factor = 1.0,
        # stream baseline
        additional_nb_trees = 0,
        # other config
        refine_splits_rounds = 0,
        num_split_candidates = 32,
        gradient_filtering = False,
        leaf_clipping = False,
        cyclical_feature_interactions = True,
        refine_splits = False,
        random_splits_from_candidates = True,
        refine_splits_subsample = 1.0,
        cut_off_leaf_denom = True,
        # debugging
        custom_noise_scale = -1.0,
        verbosity = 4,  # 1: debug; 2: info; 4: err
        # dataset
        task='Regression',
        feature_val_border = (0., .5),
        continuous_learning = False,
        scale_y = True,
        cat_idx = cat_idx,
        num_idx = num_idx,
        cat_values = cat_values,
  )
    model.train(X_train, y_train, "abalone")
    print(f'Real eps: {model.real_eps:.6f}, alpha: {model.alpha:4.0f}, max rho: {model.max_rho:7.4f}, noise sigma: {model.leaf_sigma:7.4f}')
    y_pred = model.predict(X_test)  # for a python-only prediction, use `model.python_predict(X_test)`
    print('S-GBDT - RMSE: {0:f}'.format(
        model.score(y_test, y_pred, 0)))  # 0: no_classify (rmse), 1: acc, 2: untuned_acc, 3: auc (approx), 4: auc (exact), 5: f1

    trees = model.exportTrees()
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plot_tree(trees[i])
    plt.show()