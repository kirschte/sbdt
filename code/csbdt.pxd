# build_library.pxd
""" S-BDT Cython interface definition """

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

ctypedef vector[vector[double]] VVD
ctypedef vector[vector[int]] VVI

cdef extern from "parameters.h":
    ctypedef enum max_feature_type:
        RAND,
        ONE,
        LOG2,
        SQRT,
        ALL
    ctypedef enum criterion_type:
        XGD_MSE,
        XGBOOST
    ctypedef enum noise_type:
        LAPLACE,
        GAUSS
    ctypedef enum pf_accountant_type:
        GRADIENTS,
        RENYI
    ctypedef enum lambda_reg_type:
        MAX,
        ADD
    cdef struct ModelParams:
        # tree ensemble
        int nb_trees
        double subsampling_ratio
        double learning_rate
        int max_depth
        bool newton_boosting
        bool balance_partition
        lambda_reg_type lambda_reg_mode
        double l2_lambda
        double reg_delta
        # splits
        int min_samples_split
        bool ignore_split_constraints
        max_feature_type max_features
        max_feature_type max_feature_values
        criterion_type criterion
        bool reuse_attr
        # privacy
        bool use_dp
        double privacy_budget
        double privacy_budget_init_score_ratio
        double privacy_budget_gain_ratio
        double leaf_denom_noise_weight
        double l2_threshold
        double hess_l2_threshold
        double init_score_threshold
        double numeric_feature_weight
        noise_type leaf_noise
        # individual privacy filter
        bool use_privacy_filter
        bool approximate_privacy_filter
        int pf_additional_nb_trees
        double pf_l2_threshold
        double pf_hess_l2_threshold
        double pf_subsampling_ratio_factor
        # stream baseline
        int additional_nb_trees
        # other configs
        int refine_splits_rounds
        int num_split_candidates
        bool gradient_filtering
        bool leaf_clipping
        bool cyclical_feature_interactions
        bool refine_splits
        bool random_splits_from_candidates
        double refine_splits_subsample
        bool cut_off_leaf_denom
        # debugging
        double custom_noise_scale
        int verbosity
        # dataset
        shared_ptr[Task] task
        pair[double, double] feature_val_border
        bool continuous_learning
        bool scale_y
        vector[int] cat_idx
        vector[int] num_idx
        VVD cat_values


cdef extern from "loss.h":
    ctypedef enum classify_metric:
        NO_CLASSIFY,
        ACC,
        UNTUNED_ACC,
        AUC_WMW,
        AUC,
        F1
    cdef cppclass Task:
        vector[double] compute_gradients(vector[double] &y, vector[double] &y_pred, bool bce_loss) except +
        vector[double] compute_hessians(vector[double] &y, vector[double] &gradients) except +
        double compute_init_score(vector[double] &y, double threshold, bool use_dp, double privacy_budget) except +
        double compute_score(vector[double] &y, vector[double] &y_pred, classify_metric metric) except +
    cdef cppclass Regression(Task):
        Regression() except +
        vector[double] compute_gradients(vector[double] &y, vector[double] &y_pred, bool bce_loss) except +
        vector[double] compute_hessians(vector[double] &y, vector[double] &gradients) except +
        double compute_init_score(vector[double] &y, double threshold, bool use_dp, double privacy_budget) except +
        double compute_score(vector[double] &y, vector[double] &y_pred, classify_metric metric) except +
    cdef cppclass BinaryClassification(Task):
        BinaryClassification() except +
        vector[double] compute_gradients(vector[double] &y, vector[double] &y_pred, bool bce_loss) except +
        vector[double] compute_hessians(vector[double] &y, vector[double] &gradients) except +
        double compute_init_score(vector[double] &y, double threshold, bool use_dp, double privacy_budget) except +
        double compute_score(vector[double] &y, vector[double] &y_pred, classify_metric metric) except +


cdef extern from "data.h":
    cdef struct Scaler:
        double lower, upper;
        double minimum_y, maximum_y;
        double data_min, data_max
        double feature_min, feature_max
        double scale, min_
        bool scaling_required
        #Scaler()
        Scaler(double min_val, double max_val, double fmin, double fmax, bool scaling_required) except +
    cdef cppclass DataSet:
        #DataSet()
        DataSet(VVD X, vector[double] y, string name) except +

        VVD X
        vector[double] y
        vector[int] cluster_ids
        vector[double] gradients
        vector[double] hessians
        vector[int] identifiers
        int length, num_x_cols
        bool empty
        Scaler scaler
        string name

        void add_row(vector[double] xrow, double yval) except +
        void scale_y(ModelParams &params, double lower, double upper) except +
        void scale_y_with_scaler(ModelParams &params, Scaler scaler) except +
        void shuffle_dataset() except +
        DataSet copy() except +
        DataSet get_subset(vector[int] &indices) except +
        DataSet remove_rows(vector[int] &indices) except +
    void inverse_scale_y(ModelParams &params, Scaler &scaler, vector[double] &vec) except +


cdef extern from "dp_ensemble.h":
    cdef cppclass DPEnsemble:
        DPEnsemble(ModelParams *params) except +

        void train(DataSet *dataset) except +
        vector[double] predict(VVD &X) except +
        vector[double] predict(VVD &X, size_t treestop_early) except +
        void getTrees(VVI &split_attr, VVD &split_val, vector[VVI] &next_node, VVD &leafs) except +
        void exportTrees(VVD &value, VVI &children_left, VVI &children_right,
                         VVD &threshold, vector[vector[double complex]] &impurity, VVI &feature, VVI &n_node_samples) except +

        double init_score
        double real_eps
        double alpha
        double max_rho
        double leaf_sigma

cdef extern from "logging.h":
    void setup_logging(unsigned int verbosity) except +