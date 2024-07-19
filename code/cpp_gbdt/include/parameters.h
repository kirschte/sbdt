#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <map>
#include <memory>
#include <vector>
#include <cmath>
#include <functional>
#include "loss.h"

#define SGN(x) (static_cast<int>(x > 0) - static_cast<int>(x < 0))


typedef enum {RAND, ONE, LOG2, SQRT, ALL} max_feature_type;
typedef enum {XGD_MSE, XGBOOST} criterion_type;
typedef enum {LAPLACE, GAUSS} noise_type;
typedef enum {GRADIENTS, RENYI} pf_accountant_type;
typedef enum {MAX, ADD} lambda_reg_type;

inline std::string mft_to_str(max_feature_type t) {
    switch (t) {
        case RAND: return "random";
        case ONE: return "one";
        case LOG2: return "log2";
        case SQRT: return "sqrt";
        case ALL: return "all";
    }
    return "undef";
}

inline std::string ct_to_str(criterion_type t) {
    switch (t) {
        case XGD_MSE: return "xgdboost_mse";
        case XGBOOST: return "xgboost";
    }
    return "undef";
}

inline std::string nt_to_str(noise_type t) {
    switch (t) {
        case LAPLACE: return "laplace";
        case GAUSS: return "gauss";
    }
    return "undef";
}

inline std::string lrt_to_str(lambda_reg_type t) {
    switch (t) {
        case MAX: return "max";
        case ADD: return "add";
    }
    return "undef";
}

struct ModelParams {
    // tree ensemble
    int nb_trees = 150;
    double subsampling_ratio = 0.1;
    double learning_rate = 0.1;
    int max_depth = 2;
    bool newton_boosting = false;
    bool balance_partition = true;
    lambda_reg_type lambda_reg_mode = ADD;
    double l2_lambda = 15;
    double reg_delta = 2.0;
    // splits
    int min_samples_split = 0;
    bool ignore_split_constraints = true;
    max_feature_type max_features = RAND;
    max_feature_type max_feature_values = RAND;
    criterion_type criterion = XGD_MSE;
    bool reuse_attr = true;
    // privacy
    bool use_dp = true;
    double privacy_budget = 0.1;
    double privacy_budget_init_score_ratio = 0.1;
    double privacy_budget_gain_ratio = 0.5;
    double leaf_denom_noise_weight = 0.2;
    double l2_threshold = 0.1;
    double hess_l2_threshold = 1.0;
    double init_score_threshold = 1.0;
    double numeric_feature_weight = 1.0;
    noise_type leaf_noise = GAUSS;
    // individual privacy filter
    bool use_privacy_filter = false;
    bool approximate_privacy_filter = false;
    int pf_additional_nb_trees = 0;
    double pf_l2_threshold = 0.1;
    double pf_hess_l2_threshold = 1.0;
    double pf_subsampling_ratio_factor = 1.0;
    // stream baseline
    int additional_nb_trees = 0;
    // other configs
    int refine_splits_rounds = 0;
    int num_split_candidates = 32;
    bool gradient_filtering = false;
    bool leaf_clipping = false;
    bool cyclical_feature_interactions = true;
    bool refine_splits = false;
    bool random_splits_from_candidates = true;
    double refine_splits_subsample = 1.0;
    bool cut_off_leaf_denom = true;
    // debugging
    double custom_noise_scale = -1.0;
    int verbosity = -1;
    // dataset
    std::shared_ptr<Task> task;
    std::pair<double, double> feature_val_border{0., .5};
    bool continuous_learning = false;
    bool scale_y = true;
    std::vector<int> cat_idx;
    std::vector<int> num_idx;
    std::vector<std::vector<double>> cat_values;
};

// each tree has these additional parameters
struct TreeParams {
    double delta_g = 0.0;
    double tree_privacy_budget;
    const double gain_privacy_share;
    double active_threshold;
    double hess_active_threshold;
    double active_subsampling_ratio;
    double active_noise_scale;
    double leaf_eps = 0.0;
    double rdp_alpha = 0.0;
    std::vector<std::vector<double>> split_candidates;
    std::map<std::tuple<const double, const double, const double>, const double> factor_table;
    std::map<std::tuple<const double, const double, const double, const double>, const double> leaf_rho_table;
    std::map<std::tuple<const double, const double, const double>, const double> leaf_sigma_table;
    std::map<std::pair<const double, const double>, const double> rho_approx_table;
    std::vector<double> saved_predictions;
    std::vector<double> feature_weights = std::vector<double>();

    TreeParams(
        double gain_eps_share
    ) : gain_privacy_share( gain_eps_share )
    {};
};

struct Accounting {
    double alpha;
    double eps;
    double delta;
    double max_rho;
    double noise_scale;
};

struct HyperParams {
    double g;
    double h;
    int nb;
    int d;
    double Q;
    double r1;
    bool isc;
    bool cfi;
    bool rsc;
    int lrm;
    double lam;
    double lr;
    bool rs;
    int rs_r;
    bool is;
    double isr;
    double ist;
    bool use_pf;
    int pf_trees;
    int add_trees;
    
    bool operator==(const HyperParams& other) const {
        return std::tie(g, h, nb, d, Q, r1, isc, cfi, rsc, lrm, lam, lr, rs, rs_r, is, isr, ist, use_pf, pf_trees, add_trees) == std::tie(other.g, other.h, other.nb, other.d, other.Q, other.r1, other.isc, other.cfi, other.rsc, other.lrm, other.lam, other.lr, other.rs, other.rs_r, other.is, other.isr, other.ist, other.use_pf, other.pf_trees, other.add_trees);
    }
};

#ifndef DELTA
#define DELTA (5e-8)  // for Gauss noise
#endif

#ifndef MAX_ALPHA
#define MAX_ALPHA (2000)  // for RDP
#endif

#endif /* PARAMETERS_H */
