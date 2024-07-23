# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
""" S-BDT Cython interface definition """

cimport cython
cimport numpy as np
from csbdt cimport (RMSE, VVD, VVI, BinaryClassification, DataSet,
                    DPEnsemble, ModelParams, Regression, Task, classify_metric,
                    criterion_type, inverse_scale_y, lambda_reg_type,
                    max_feature_type, noise_type, pf_accountant_type,
                    setup_logging)
from libc.stdlib cimport srand
from libc.time cimport time
from libcpp cimport bool
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np
import py.io
from sklearn.base import BaseEstimator


cdef class S_BDT:
    cdef DPEnsemble* gbdt
    cdef ModelParams model_params
    cdef DataSet* dataset

    def __cinit__(
        self,
        # tree ensemble
        int nb_trees,
        double subsampling_ratio,
        double learning_rate,
        int max_depth,
        bool newton_boosting,
        bool balance_partition,
        lambda_reg_type lambda_reg_mode,
        double l2_lambda,
        double reg_delta,
        # splits
        int min_samples_split,
        bool ignore_split_constraints,
        max_feature_type max_features,
        max_feature_type max_feature_values,
        criterion_type criterion,
        bool reuse_attr,
        # privacy
        bool use_dp,
        double privacy_budget,
        double privacy_budget_init_score_ratio,
        double privacy_budget_gain_ratio,
        double leaf_denom_noise_weight,
        double l2_threshold,
        double hess_l2_threshold,
        double init_score_threshold,
        double numeric_feature_weight,
        noise_type leaf_noise,
        # individual privacy filter
        bool use_privacy_filter,
        bool approximate_privacy_filter,
        int pf_additional_nb_trees,
        double pf_l2_threshold,
        double pf_hess_l2_threshold,
        double pf_subsampling_ratio_factor,
        # stream baseline
        int additional_nb_trees,
        # other configs
        int refine_splits_rounds,
        int num_split_candidates,
        bool gradient_filtering,
        bool leaf_clipping,
        bool cyclical_feature_interactions,
        bool refine_splits,
        bool random_splits_from_candidates,
        double refine_splits_subsample,
        bool cut_off_leaf_denom,
        # debugging
        double custom_noise_scale,
        int verbosity,
        # dataset
        string task,
        pair[double, double] feature_val_border,
        bool continuous_learning,
        bool scale_y,
        vector[int] cat_idx,
        vector[int] num_idx,
        VVD cat_values,
    ):
        # initialize C++ randomness
        srand(<unsigned> time(NULL))

        # setup logging
        setup_logging(min(7, max(0, verbosity)))

        cdef shared_ptr[Task] task_
        if <unicode> task == u'Regression':
            task_ = make_shared[Regression]()
        elif <unicode> task == u'BinaryClassification':
            task_ = make_shared[BinaryClassification]()

        self.model_params = ModelParams(
            nb_trees=nb_trees,
            subsampling_ratio=subsampling_ratio,
            learning_rate=learning_rate,
            max_depth=max_depth,
            newton_boosting=newton_boosting,
            balance_partition=balance_partition,
            lambda_reg_mode=lambda_reg_mode,
            l2_lambda=l2_lambda,
            reg_delta=reg_delta,
            min_samples_split=min_samples_split,
            ignore_split_constraints=ignore_split_constraints,
            max_features=max_features,
            max_feature_values=max_feature_values,
            criterion=criterion,
            reuse_attr=reuse_attr,
            use_dp=use_dp,
            privacy_budget=privacy_budget,
            privacy_budget_init_score_ratio=privacy_budget_init_score_ratio,
            privacy_budget_gain_ratio=privacy_budget_gain_ratio,
            leaf_denom_noise_weight=leaf_denom_noise_weight,
            l2_threshold=l2_threshold,
            hess_l2_threshold=hess_l2_threshold,
            init_score_threshold=init_score_threshold,
            numeric_feature_weight=numeric_feature_weight,
            leaf_noise=leaf_noise,
            use_privacy_filter=use_privacy_filter,
            approximate_privacy_filter=approximate_privacy_filter,
            pf_additional_nb_trees=pf_additional_nb_trees,
            pf_l2_threshold=pf_l2_threshold,
            pf_hess_l2_threshold=pf_hess_l2_threshold,
            pf_subsampling_ratio_factor=pf_subsampling_ratio_factor,
            additional_nb_trees=additional_nb_trees,
            refine_splits_rounds=refine_splits_rounds,
            num_split_candidates=num_split_candidates,
            gradient_filtering=gradient_filtering,
            leaf_clipping=leaf_clipping,
            cyclical_feature_interactions=cyclical_feature_interactions,
            refine_splits=refine_splits,
            random_splits_from_candidates=random_splits_from_candidates,
            refine_splits_subsample=refine_splits_subsample,
            cut_off_leaf_denom=cut_off_leaf_denom,
            custom_noise_scale=custom_noise_scale,
            verbosity=verbosity,
            task=task_,
            feature_val_border=feature_val_border,
            continuous_learning=continuous_learning,
            scale_y=scale_y,
            cat_idx=cat_idx,
            num_idx=num_idx,
            cat_values=cat_values,
        )
        if self.gbdt != NULL:
            del self.gbdt
        self.gbdt = new DPEnsemble(&self.model_params)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def train(self, np.ndarray[double, ndim=2, mode="c"] X not None, np.ndarray[double, ndim=1, mode="c"] y not None, string name, np.ndarray[long, ndim=1, mode="c"] cluster_ids=None):    
        cdef VVD x_vec = X
        cdef vector[double] y_vec = y
        if self.dataset != NULL:
            del self.dataset

        capture = py.io.StdCaptureFD(out=False, in_=False)
        self.dataset = new DataSet(x_vec, y_vec, name)
        if cluster_ids is not None:
            self.dataset.cluster_ids = cluster_ids

        if self.model_params.scale_y:
            self.dataset.scale_y(self.model_params, -1, 1)
        self.gbdt.train(self.dataset)
        out, err = capture.reset()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def predict(self, np.ndarray[double, ndim=2, mode="c"] X not None, int treestop_early=-1):
        cdef VVD x_vec = X
        cdef vector[double] y_pred

        capture = py.io.StdCaptureFD(out=False, in_=False)
        if treestop_early >= 0:
            y_pred = self.gbdt.predict(x_vec, treestop_early)
        else:
            y_pred = self.gbdt.predict(x_vec)
        if self.model_params.scale_y:
            inverse_scale_y(self.model_params, self.dataset.scaler, y_pred)
        
        out, err = capture.reset()
        return np.asarray(y_pred)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def python_predict(self, np.ndarray[double, ndim=2, mode="c"] X not None):
        split_attr, split_val, next_node, leafs = self.getTrees()
        _cat_idx = np.asarray(self.model_params.cat_idx, dtype=int)

        def _d_predict(split_attr, next_node, leafs, row_val):
            def _d_predict_(index):
                if next_node[index][2] != -1:
                    return leafs[next_node[index][2]]
                lpred = _d_predict_(next_node[index][0])
                rpred = _d_predict_(next_node[index][1])
                if np.isin(split_attr[index], _cat_idx):
                    return np.where( row_val[:, index] == 0, lpred, rpred)
                else:
                    return np.where( row_val[:, index] <= 0, lpred, rpred)
            return _d_predict_

        y_pred = self.gbdt.init_score
        lr = self.model_params.learning_rate
        nb_trees = self.model_params.nb_trees
        for i in range(nb_trees):
            split_attr_, split_val_, next_node_, leafs_ = np.asarray(split_attr[i], dtype=int), np.asarray(split_val[i], dtype=float), np.asarray(next_node[i], dtype=int), np.asarray(leafs[i], dtype=float)
            row_val = X[:,split_attr_] - split_val_
            y_pred_i = _d_predict(split_attr_, next_node_, leafs_, row_val)(len(next_node_) - 1)
            y_pred = y_pred + lr * y_pred_i
        
        if self.model_params.scale_y:
            y_pred -= self.dataset.scaler.min_;
            y_pred /= self.dataset.scaler.scale;
        return y_pred
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def score(self, np.ndarray[double, ndim=1, mode="c"] y not None, np.ndarray[double, ndim=1, mode="c"] y_pred not None, classify_metric metric=RMSE):
        cdef vector[double] y_vec = y
        cdef vector[double] y_pred_vec = y_pred

        capture = py.io.StdCaptureFD(out=False, in_=False)
        score = self.model_params.task.get().compute_score(y_vec, y_pred_vec, metric if metric else RMSE)

        out, err = capture.reset()
        return score


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getTrees(self):
        cdef VVI split_attr
        cdef VVD split_val
        cdef vector[VVI] next_node
        cdef VVD leafs
        self.gbdt.getTrees(split_attr, split_val, next_node, leafs)
        return split_attr, split_val, next_node, leafs


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def exportTrees(self):
        cdef VVD value
        cdef VVI children_left
        cdef VVI children_right
        cdef VVD threshold
        cdef vector[vector[double complex]] impurity
        cdef VVI feature
        cdef VVI n_node_samples
        self.gbdt.exportTrees(value, children_left, children_right, threshold,
            impurity, feature, n_node_samples)
        trees = []
        for i in range(len(value)):
            trees.append(
                MockedTree(
                    np.asarray(value[i]),
                    np.asarray(children_left[i]),
                    np.asarray(children_right[i]),
                    np.asarray(threshold[i]),
                    np.asarray(impurity[i]),
                    np.asarray(feature[i]),
                    np.asarray(n_node_samples[i]),
                    self.model_params.criterion)
            )
        return trees


    @property
    def init_score(self):
        return self.gbdt.init_score

    @property
    def real_eps(self):
        return self.gbdt.real_eps
    
    @property
    def alpha(self):
        return self.gbdt.alpha
    
    @property
    def max_rho(self):
        return self.gbdt.max_rho


    @property
    def leaf_sigma(self):
        return self.gbdt.leaf_sigma


    def __dealloc__(self):
        del self.gbdt, self.dataset


class MockedTree(BaseEstimator):
    def __init__(self, value, children_left, children_right,
                 threshold, impurity, feature, n_node_samples, criterion):
        if criterion == 0:
            self.criterion = 'xgdboost_mse'
        elif criterion == 1:
            self.criterion = 'mse'
        elif criterion == 2:
            self.criterion = 'friedman_mse'
        self.tree_ = TreeExporter(value, children_left, children_right,
                 threshold, impurity, feature, n_node_samples)
    
    @staticmethod
    def fit() -> None:
        """Stub for MockedTree"""
        return

    @staticmethod
    def predict() -> None:
        """Stub for MockedTree"""
        return
    
    @staticmethod
    def __sklearn_is_fitted__() -> bool:
        return True

class TreeExporter:
    def __init__(self, value, children_left, children_right,
                 threshold, impurity, feature, n_node_samples):
        self.n_outputs = 1
        self.value = value.reshape(-1,1,1)
        self.children_left = children_left
        self.children_right = children_right
        self.threshold = threshold
        self.impurity = impurity
        self.feature = feature
        self.n_node_samples = n_node_samples
        self.n_classes = np.asarray([1])
        self.weighted_n_node_samples = np.full(fill_value=1, shape=len(value))
